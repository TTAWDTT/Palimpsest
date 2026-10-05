"""Six preregistered threshold calibrations; frozen eighth readout weights."""

import json
from time import perf_counter
import tomllib

import cv2
import numpy as np

from palimpsest.detection.algorithms.paired_stability import StableRule, StableDetector
from palimpsest.detection.algorithms.residual_statistics.features import FEATURE_NAMES
from palimpsest.evaluation.features import feature_views
from palimpsest.evaluation.file_benchmark import benchmark_files
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.phase_statistics.run_iteration import write_json, image_path
from experiments.origin_detection.residual_statistics.run_iteration import screen_rule, aggregate
from experiments.origin_detection.paired_stability.run_iteration import (
    OUTPUT as PARENT, CONFIG, INTERVENTION, load_inputs, code_pins as parent_pins,
)
from .fit_threshold import class_threshold

OUTPUT = WORK_DIR/'robust_statistics/threshold_calibration'


def code_pins():
    directory = REPO_ROOT/'experiments/origin_detection/threshold_calibration'
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p)
            for p in sorted([directory/'README.md', *directory.glob('*.py')])}


def main():
    if (OUTPUT/'iteration.json').exists(): raise FileExistsError('Preserve ninth evaluation')
    cv2.setNumThreads(1)
    rows, held, _ = load_inputs()
    parent = json.loads((PARENT/'iteration.json').read_text(encoding='utf-8'))
    if parent['code_pins'] != parent_pins(): raise ValueError('Eighth readout code changed')
    if set(parent['rule_files']) != {'lambda0', 'lambda1', 'lambda10'}:
        raise ValueError('Registered ninth candidate set changed')
    config = tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    output, rules = {}, {}
    start = perf_counter()
    for mode, digest in parent['rule_files'].items():
        path = PARENT/f'{mode}_rule.json'
        if file_sha256(path) != digest: raise ValueError('Frozen eighth rule changed')
        base_rule = StableRule.load(path)
        for policy in ('raw', 'raw90'):
            variants = ('raw',) if policy == 'raw' else ('raw', 'jpeg90_444_after_resize256')
            views = {}
            for variant in variants:
                for processed in (False, True):
                    views.update({k+'/'+variant: v for k, v in feature_views(
                        rows, FEATURE_NAMES, 'threshold', processed=processed, variant=variant).items()})
            rule, calibration = class_threshold(base_rule, views)
            result = screen_rule(rows, FEATURE_NAMES, rule, config)
            encoded = aggregate(held, FEATURE_NAMES, rule, INTERVENTION)
            changes = [(result['selection_aggregate'][d]['conditions'][c]['balanced_accuracy_at_zero'],
                        v['balanced_accuracy_at_zero'])
                       for d, z in encoded.items() for c, v in z['conditions'].items() if c != 'original']
            result['criteria'].update({
                'held_encoding_ba_floor': all(b >= config['minimum_reencoded_ba'] for _, b in changes),
                'held_encoding_ba_drop': all(a-b <= config['maximum_reencoded_ba_drop'] for a, b in changes),
                'both_encoded_class_accuracies': all(v[k] >= .55 for z in (result['recompression_aggregate'], encoded)
                                                     for d in z.values() for v in d['conditions'].values()
                                                     for k in ('real_accuracy_at_zero', 'fake_accuracy_at_zero')),
            })
            result['gate_passed'] = all(result['criteria'].values())
            result['held_encoding_aggregate'] = encoded
            result['calibration'] = calibration
            result['parent_threshold'] = base_rule.threshold
            for variant, data in [('raw', rows), ('jpeg90_444_after_resize256', rows), (INTERVENTION, held)]:
                selected = [r for r in data if r['role'] == 'selection' and r['variant'] == variant]
                x = np.array([[float(r[n]) for n in FEATURE_NAMES] for r in selected])
                if not np.array_equal(rule.score(x), [rule.score([v])[0] for v in x]):
                    raise ValueError('Single/batch stable scores differ')
            output[mode+'/'+policy] = result; rules[mode+'/'+policy] = rule
            print(json.dumps({'candidate': mode+'/'+policy, 'calibration': calibration, 'gate': result['gate_passed']}), flush=True)
    passing = [m for m, z in output.items() if z['gate_passed']]
    # Threshold changes cannot repair the parent's failed null AUC gate.
    refused = parent['candidates']['source_shuffled_null']['criteria']['eight_auc_lower_bounds']
    chosen = max(passing, key=lambda m: output[m]['worst_processed_auc']) if passing and not refused else None
    result = {'candidates': output, 'chosen': chosen, 'code_pins': code_pins(), 'elapsed_s': perf_counter()-start,
              'parent_iteration_sha256': file_sha256(PARENT/'iteration.json'), 'single_batch_scores_exact': True,
              'interpretation_refused': refused, 'null_control': 'Inherited failed eighth null AUC; threshold cannot change ranking',
              'scope': 'Threshold-only repeated development; no independently trained new classifier'}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for mode, rule in rules.items(): rule.save(OUTPUT/(mode.replace('/', '_')+'_rule.json'))
    result['rule_files'] = {m: file_sha256(OUTPUT/(m.replace('/', '_')+'_rule.json')) for m in rules}
    write_json(OUTPUT/'iteration.json', result)
    if chosen:
        prior = WORK_DIR/'robust_statistics/rr_first_iteration/benchmark.csv'
        signature = json.loads(prior.with_suffix('.json').read_text(encoding='utf-8'))
        if file_sha256(prior) != signature['csv_sha256']: raise ValueError('Benchmark selection changed')
        names = {'rr/'+r['filename'] for r in read_rows(prior)}
        selected = [r for r in rows if r['variant'] == 'raw' and r['filename'] in names]
        if len(selected) != 60: raise ValueError('Benchmark denominator differs')
        rule = rules[chosen]
        items = [{'filename': r['filename'], 'path': image_path(r), 'sha256': r['sha256'],
                  'expected_score': float(rule.score([[float(r[n]) for n in FEATURE_NAMES]])[0])} for r in selected]
        write_json(OUTPUT/'benchmark.json', benchmark_files(StableDetector(rule), items))


if __name__ == '__main__': main()
