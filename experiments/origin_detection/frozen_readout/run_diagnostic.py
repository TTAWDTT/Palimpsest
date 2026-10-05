"""Fifth iteration: two fixed readouts of existing signed caches, no image decode."""

import argparse
from dataclasses import asdict
import json
from time import perf_counter
import tomllib

import joblib
import numpy as np
import sklearn

from palimpsest.contracts import Origin, Prediction
from palimpsest.evaluation.classification import evaluate
from palimpsest.evaluation.detection import DetectionObservation, evaluate_detection
from palimpsest.evaluation.features import feature_views
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from palimpsest.detection.algorithms.ordinal_statistics.features import FEATURE_NAMES as ORDINAL_NAMES
from palimpsest.detection.algorithms.phase_statistics.features import FEATURE_NAMES as PHASE_NAMES, DECISION_INDICES
from experiments.origin_detection.ordinal_statistics.fit_rules import choose_threshold
from experiments.origin_detection.ordinal_statistics.run_iteration import load_cache, CONFIG as ORDINAL_CONFIG
from experiments.origin_detection.phase_statistics.run_iteration import signed_features
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals
from .fit_readout import fit_readout

CONFIG = REPO_ROOT / "configs/evaluation/frozen_readout.toml"
OUTPUT = WORK_DIR / "robust_statistics/frozen_readout"


def evaluate_readout(rows, names, config, *, shuffled=False):
    fit = feature_views(rows, names, "fit", processed=True)
    threshold = feature_views(rows, names, "threshold", processed=True)
    if len(fit) != 8 or len(threshold) != 8:
        raise ValueError("Registered eight fit/threshold views required")
    start = perf_counter()
    rule = choose_threshold(fit_readout(fit, seed=config['seed'], shuffled=shuffled), threshold)
    fit_threshold_s = perf_counter() - start
    strata = {}
    for processed in (False, True):
        selection = feature_views(rows, names, "selection", processed=processed)
        if len(selection) != (8 if processed else 4):
            raise ValueError("Registered selection view count differs")
        for key, (records, values, labels) in selection.items():
            scores = rule.score(values)
            strata[key] = {**evaluate([{**r, 'score': float(s-rule.threshold)} for r, s in zip(records, scores)], 'score'),
                           'auc_ci95': auc_intervals({key: scores}, labels, config)[key]}
    aggregate = {}
    for domain in ('rr', 'chimera'):
        selected = [r for r in rows if r['role'] == 'selection' and r['domain'] == domain and r['variant'] == 'raw']
        values = np.array([[float(r[n]) for n in names] for r in selected])
        aggregate[domain] = evaluate_detection([
            DetectionObservation(r['src'], r['condition'], Origin.AI if r['label']=='FAKE' else Origin.NATURAL,
                                 Prediction('frozen-readout', float(s), rule.threshold, 'statistical_score'))
            for r,s in zip(selected,rule.score(values))])
    processed = [v for k,v in strata.items() if not k.endswith('/original')]
    criteria = {
        'eight_auc_lower_bounds': all(v['auc_ci95'][0] > config['selection_auc_lower_gate'] for v in processed),
        'eight_both_class_accuracies': all(v[k] >= config['minimum_processed_class_accuracy'] for v in processed
                                         for k in ('real_accuracy_at_zero','fake_accuracy_at_zero')),
        'both_original_class_accuracies': all(d['conditions']['original'][k] >= config['minimum_original_class_accuracy']
                                            for d in aggregate.values() for k in ('real_accuracy_at_zero','fake_accuracy_at_zero')),
    }
    return {'threshold':rule.threshold, 'criteria':criteria, 'accuracy_gate_passed':all(criteria.values()),
            'selection_views':strata, 'selection_aggregate':aggregate,
            'fit_threshold_s':fit_threshold_s, 'shuffled_fit_labels':shuffled}, rule


def code_pins():
    files = [CONFIG, REPO_ROOT/'src/palimpsest/evaluation/features.py']
    files += list((REPO_ROOT/'experiments/origin_detection/frozen_readout').glob('*.py'))
    files.append(REPO_ROOT/'experiments/origin_detection/frozen_readout/README.md')
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in sorted(files)}


def run(caches, config, output):
    if output.exists():
        raise FileExistsError("Preserve frozen-readout results")
    output.mkdir(parents=True)
    results = {}
    start = perf_counter()
    for mode, (rows, names, lineage) in caches.items():
        results[mode] = {'lineage':lineage, 'feature_names':list(names)}
        for shuffled in (False, True):
            label = 'source_shuffled_null' if shuffled else 'labeled'
            result, rule = evaluate_readout(rows, names, config, shuffled=shuffled)
            model = output/f'{mode}_{label}.joblib'
            joblib.dump(asdict(rule), model)
            result['model_sha256'] = file_sha256(model)
            # Reload local artifact and verify numerical equality, not an independent science check.
            from .fit_readout import Readout
            restored = Readout(**joblib.load(model))
            probe = np.array([[float(r[n]) for n in names] for r in rows[:20]])
            np.testing.assert_array_equal(restored.score(probe), rule.score(probe))
            results[mode][label] = result
            print(json.dumps({'candidate':mode,'diagnostic':label,'accuracy_gate_passed':result['accuracy_gate_passed']}),flush=True)
    receipt = {'results':results,'elapsed_s':perf_counter()-start,'code_pins':code_pins(),
               'interpretation_refused':any(r['source_shuffled_null']['accuracy_gate_passed'] for r in results.values()),
               'sklearn_version':sklearn.__version__, 'config':config,
               'scope':'Fifth diagnostic on repeatedly exposed development sources; no new algorithm or blind validation'}
    (output/'diagnostic.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',action='store_true',help='Fit+threshold one candidate only; no selection evaluation')
    args=parser.parse_args()
    with CONFIG.open('rb') as stream:
        config=tomllib.load(stream)
    with ORDINAL_CONFIG.open('rb') as stream:
        ordinal, _, ordinal_receipt = load_cache(tomllib.load(stream))
    phase, phase_receipt = signed_features()
    names=tuple(PHASE_NAMES[i] for i in DECISION_INDICES)
    caches={'ordinal':([{**r,'variant':'raw'} for r in ordinal], ORDINAL_NAMES,
                       {'features_receipt_sha256':file_sha256(WORK_DIR/'robust_statistics/joint_ordinal_statistics/features.json'),
                        'cache_sha256':ordinal_receipt['outputs']['features.csv']}),
            'phase':(phase,names,{'features_receipt_sha256':file_sha256(WORK_DIR/'robust_statistics/joint_phase_statistics/features.json'),
                                 'cache_sha256':phase_receipt['csv_sha256']})}
    if args.pilot:
        start=perf_counter();rows,names,_=caches['ordinal']
        rule=fit_readout(feature_views(rows,names,'fit',processed=True),seed=config['seed'])
        choose_threshold(rule,feature_views(rows,names,'threshold',processed=True))
        print(json.dumps({'pilot':'ordinal fit+threshold','elapsed_s':perf_counter()-start,'no_selection':True}))
    else:
        run(caches,config,OUTPUT)


if __name__=='__main__':
    main()
