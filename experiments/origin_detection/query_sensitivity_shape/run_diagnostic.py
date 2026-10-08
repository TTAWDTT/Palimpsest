"""Fit-only held source rankings, comparing fixed linear and tree readouts."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.source_bagged_forest import fit_source_forest
from palimpsest.detection.algorithms.source_view_risk import fit_source_risk
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.classification import auc
from palimpsest.evaluation.source_crossfit import source_folds
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.query_sensitivity.protocol import OUTPUT as PARENT, NAMES, SEED, write_json

OUTPUT = WORK_DIR/'robust_statistics/query_sensitivity_shape'
METHODS = ('linear', 'source_forest', 'row_forest')


def code_pins():
    files = list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md',
        REPO_ROOT/'src/palimpsest/detection/algorithms/source_bagged_forest.py',
        REPO_ROOT/'src/palimpsest/detection/algorithms/source_view_risk.py',
        REPO_ROOT/'src/palimpsest/evaluation/source_training.py',
        REPO_ROOT/'src/palimpsest/evaluation/source_crossfit.py',
        REPO_ROOT/'src/palimpsest/evaluation/balanced_null.py',
        REPO_ROOT/'src/palimpsest/evaluation/classification.py']
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(files)}


def fit(method, x, labels, weights, sources, domains):
    if method == 'linear':
        return fit_source_risk(x, labels, weights, sources, feature_names=NAMES)
    return fit_source_forest(x, labels, sources, domains, feature_names=NAMES, seed=SEED,
        source_level=method == 'source_forest', trees=64, maximum_depth=6, minimum_leaf=12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT/('pilot.json' if args.pilot else 'diagnostic.json')
    if destination.exists():
        raise FileExistsError('Preserve diagnostic')
    parent = json.loads((PARENT/'features.json').read_text())
    if not parent['passed'] or parent['csv_sha256'] != file_sha256(PARENT/'features.csv'):
        raise ValueError('Sensitivity cache changed')
    rows = read_rows(PARENT/'features.csv')
    records = sorted([r for r in rows if r['role'] == 'fit'],
        key=lambda r: tuple(r[k] for k in ('domain', 'src', 'condition', 'variant')))
    x, truth, weights, sources = weighted_source_arrays(rows, NAMES, ('raw',),
        expected_source_counts={'rr': 540, 'chimera': 720}, seed=SEED)
    domains = np.array([r['domain'] for r in records])
    folds = source_folds(records)
    ids = np.array([folds[r['domain'], r['src']] for r in records])
    assignment, null = balanced_source_null(records, seed=SEED)
    sham = np.array([assignment[r['domain'], r['src']] for r in records])
    if args.pilot:
        controls = subprocess.run([sys.executable, '-m', 'pytest', '-q',
            'tests/detection/test_source_view_risk.py', 'tests/detection/test_source_bagged_forest.py'],
            capture_output=True, text=True)
        if controls.returncode:
            raise ValueError(controls.stdout+controls.stderr)
        write_json(OUTPUT/'software_controls.json', {'passed': True, 'stdout': controls.stdout,
            'code_pins': code_pins(), 'scope': 'Existing component tests only'})
    else:
        pilot = json.loads((OUTPUT/'pilot.json').read_text())
        if (not pilot['passed'] or pilot['code_pins'] != code_pins()
                or pilot['parent_receipt_sha256'] != file_sha256(PARENT/'features.json')):
            raise ValueError('Diagnostic pilot changed')
    labels_list = (('truth', truth),) if args.pilot else (('truth', truth), ('null', sham))
    scores, diagnostics, times = {}, {}, {}
    start = perf_counter()
    with threadpool_limits(limits=1):
        for mode, labels in labels_list:
            for method in METHODS:
                name = mode+'/'+method
                predictions = np.full(len(x), np.nan)
                entries, elapsed = [], []
                for held in ((0,) if args.pilot else range(5)):
                    train, test = ids != held, ids == held
                    if int(train.sum()) != 3024 or int(test.sum()) != 756:
                        raise ValueError('Diagnostic fold row count changed')
                    _, groups = np.unique(sources[train], return_inverse=True)
                    tick = perf_counter()
                    rule, diagnostic = fit(method, x[train], labels[train], weights[train],
                        groups, domains[train])
                    values = rule.score(x[test])
                    elapsed.append(perf_counter()-tick)
                    if values.shape != (756,) or not np.isfinite(values).all():
                        raise ValueError('Invalid diagnostic held scores')
                    predictions[test] = values
                    entries.append({'held_fold': held, 'train_sources': len(np.unique(sources[train])),
                        'held_sources': len(np.unique(sources[test])), 'fit': diagnostic})
                    print({'mode': mode, 'method': method, 'held_fold': held, 'elapsed_s': elapsed[-1]}, flush=True)
                times[name] = elapsed
                diagnostics[name] = entries
                if not args.pilot:
                    if not np.isfinite(predictions).all():
                        raise ValueError('Missing diagnostic OOF')
                    scores[name] = [{**{k: r[k] for k in ('domain', 'scene', 'condition', 'variant', 'src', 'role')},
                        'label': 'FAKE' if label else 'REAL', 'score': float(score), 'fold': int(fold)}
                        for r, label, score, fold in zip(records, labels, predictions, ids)]
    elapsed = perf_counter()-start
    common = {'passed': True, 'elapsed_s': elapsed, 'fit_score_s': times, 'diagnostics': diagnostics,
        'code_pins': code_pins(), 'parent_receipt_sha256': file_sha256(PARENT/'features.json')}
    if args.pilot:
        projection = 10*sum(v[0] for v in times.values())
        write_json(destination, {**common, 'conditional_30fit_projection_s': projection,
            'assumption': 'Same-size five folds times truth/null;label-specific cost may change;input/tests/write excluded',
            'passed': projection < 7200})
        return
    groups, metrics = {}, {}
    for name, scored in scores.items():
        groups[name] = {}
        for row in scored:
            key = '/'.join(row[k] for k in ('domain', 'scene', 'condition'))
            groups[name].setdefault(key, []).append((row['score'], int(row['label'] == 'FAKE')))
            if row['scene'] != 'all':
                groups[name].setdefault(row['domain']+'/all/'+row['condition'], []).append(
                    (row['score'], int(row['label'] == 'FAKE')))
        metrics[name] = {key: {'auc': auc(pairs), 'rows': len(pairs), 'fake_rows': sum(y for _, y in pairs)}
            for key, pairs in groups[name].items()}
    write_json(OUTPUT/'oof_scores.json', scores)
    write_json(destination, {**common, 'metrics': metrics, 'null_assignment': null,
        'fold_sources': {'/'.join(k): v for k, v in folds.items()},
        'oof_sha256': file_sha256(OUTPUT/'oof_scores.json'),
        'scope': 'Fit-only source-held ranking;no calibration/outer scores/image latency/independent acceptance'})


if __name__ == '__main__':
    main()
