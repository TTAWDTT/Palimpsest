"""Six fixed covariance readouts on signed, previously exposed features."""

import argparse
import json
from time import perf_counter
import tomllib

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.gaussian_readout import fit_gaussian_rule
from palimpsest.detection.algorithms.residual_statistics.features import FEATURE_NAMES as OLD_NAMES
from palimpsest.evaluation.features import feature_views
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.conditional_residual.run_iteration import final_target_screen
from experiments.origin_detection.paired_stability.run_iteration import load_inputs, INTERVENTION
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.residual_statistics.run_iteration import screen_rule, aggregate
from experiments.origin_detection.residual_training.fit_rules import training_arrays, TRAINING_VARIANT
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.threshold_diagnostics.audit_rate_boundaries import audit
from experiments.origin_detection.wavelet_envelopes.run_iteration import (
    OUTPUT as ENVELOPE_DIR, MODES, audit_new, code_pins as envelope_pins, join_features,
)
from fractions import Fraction

CONFIG = REPO_ROOT/'configs/evaluation/gaussian_readout.toml'
OUTPUT = WORK_DIR/'robust_statistics/gaussian_readout'
NAMES = {'residual': OLD_NAMES, 'hybrid': MODES['hybrid']}


def code_pins():
    files = [CONFIG, REPO_ROOT/'src/palimpsest/detection/algorithms/gaussian_readout.py',
             REPO_ROOT/'tests/detection/test_gaussian_readout.py',
             REPO_ROOT/'experiments/origin_detection/threshold_diagnostics/audit_rate_boundaries.py']
    directory = REPO_ROOT/'experiments/origin_detection/gaussian_readout'
    files.extend(directory.glob('*.py')); files.append(directory/'README.md')
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(files)}


def signed_inputs():
    old, old_held, receipt = load_inputs()
    feature_receipt = json.loads((ENVELOPE_DIR/'features.json').read_text(encoding='utf-8'))
    if (feature_receipt['code_pins'] != envelope_pins()
            or feature_receipt['csv_sha256'] != file_sha256(ENVELOPE_DIR/'features.csv')
            or feature_receipt['inventory_sha256'] != receipt['inventory_sha256']):
        raise ValueError('Frozen envelope feature lineage changed')
    inventory = [r for r in old if r['variant'] == 'raw']
    ordinary, held = audit_new(read_rows(ENVELOPE_DIR/'features.csv'), inventory)
    return join_features(old, ordinary), join_features(old_held, held), receipt


def fit_rule(rows, names, mode, config, manifest_sha, *, shuffled=False, calibrate=True):
    _, labels, weights, identities = training_arrays(rows, 'equal_domain_augmented', config['seed'], shuffled=shuffled)
    values = []; ordered_ids = []
    for variant in ('raw', TRAINING_VARIANT):
        views = {**feature_views(rows, names, 'fit', processed=False, variant=variant),
                 **feature_views(rows, names, 'fit', processed=True, variant=variant)}
        for records, x, _ in views.values():
            values.append(x)
            ordered_ids.extend((r['src'], r['domain'], variant) for r in records)
    if identities != ordered_ids:
        raise ValueError('Gaussian training rows/labels traversal differs')
    rule, fit = fit_gaussian_rule(np.concatenate(values), labels, weights, feature_names=names, mode=mode,
                                 ridge=config['ridge'], shrinkage=config['shrinkage'],
                                 scale_floor=config['scale_floor'], manifest_sha=manifest_sha)
    fit['sources'] = len({s for s, _, _ in identities})
    if calibrate:
        views = {k+'/'+variant: v for variant in ('raw', TRAINING_VARIANT) for processed in (False, True)
                 for k, v in feature_views(rows, names, 'threshold', processed=processed, variant=variant).items()}
        rule, calibration = class_threshold(rule, views)
        fit['calibration'] = calibration
    return rule, fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    path = OUTPUT/('pilot.json' if args.pilot else 'iteration.json')
    if path.exists():
        raise FileExistsError('Preserve Gaussian evaluation')
    config = tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    controls = json.loads((OUTPUT/'controls.json').read_text(encoding='utf-8'))
    if controls['returncode'] != 0 or controls['test_sha256'] != file_sha256(REPO_ROOT/'tests/detection/test_gaussian_readout.py'):
        raise ValueError('Gaussian planted controls failed or changed')
    rows, held, receipt = signed_inputs()
    start = perf_counter()
    with threadpool_limits(limits=1):
        if args.pilot:
            _, diagnostic = fit_rule(rows, NAMES['hybrid'], 'pooled', config, receipt['inventory_sha256'], calibrate=False)
            write_json(path, {'elapsed_s': perf_counter()-start, 'fit_diagnostics': diagnostic, 'code_pins': code_pins(),
                              'scope': 'One fixed head cost pilot; no selection screening'})
            print(json.dumps({'pilot_s': perf_counter()-start}), flush=True)
            return
        candidates = {}; rules = {}
        specifications = [(rep, head, False) for rep in config['representations'] for head in config['readouts']]
        specifications.append(('hybrid', 'class_full', True))
        for representation, head, shuffled in specifications:
            name = representation+'/'+head+('_source_shuffled_null' if shuffled else '')
            names = NAMES[representation]
            rule, diagnostic = fit_rule(rows, names, head, config, receipt['inventory_sha256'], shuffled=shuffled)
            result = screen_rule(rows, names, rule, config)
            encoded = aggregate(held, names, rule, INTERVENTION)
            changes = [(v['balanced_accuracy_at_zero'], encoded[d]['conditions'][c]['balanced_accuracy_at_zero'])
                       for d, z in result['selection_aggregate'].items() for c, v in z['conditions'].items() if c != 'original']
            result['criteria'].update({
                'held_encoding_ba_floor': all(b >= config['minimum_reencoded_ba'] for _, b in changes),
                'held_encoding_ba_drop': all(a-b <= config['maximum_reencoded_ba_drop'] for a, b in changes),
                'both_encoded_class_accuracies': all(v[k] >= .55 for data in (result['recompression_aggregate'], encoded)
                                                     for z in data.values() for v in z['conditions'].values()
                                                     for k in ('real_accuracy_at_zero', 'fake_accuracy_at_zero'))})
            result.update({'held_encoding_aggregate': encoded, 'fit_diagnostics': diagnostic,
                           'final_target': final_target_screen(result, encoded, config)})
            exact = audit({'candidates': {name: result}}, Fraction(str(config['maximum_reencoded_ba_drop'])))[name]
            result['floating_point_criteria'] = result['criteria']
            result['criteria'] = exact['corrected_criteria']
            result['gate_passed'] = exact['corrected_gate_passed']
            result['rate_boundary_audit'] = exact['comparisons']
            for variant, data in [('raw', rows), (TRAINING_VARIANT, rows), (INTERVENTION, held)]:
                selected = [r for r in data if r['role'] == 'selection' and r['variant'] == variant]
                x = np.array([[float(r[n]) for n in names] for r in selected])
                if not np.array_equal(rule.score(x), [rule.score([v])[0] for v in x]):
                    raise ValueError('Gaussian single/batch scoring differs')
            candidates[name] = result; rules[name] = rule
            print(json.dumps({'candidate': name, 'gate': result['gate_passed'], 'target': result['final_target']}), flush=True)
    null_name = 'hybrid/class_full_source_shuffled_null'
    refused = candidates[null_name]['criteria']['eight_auc_lower_bounds']
    passing = [k for k, v in candidates.items() if k != null_name and v['gate_passed']]
    chosen = max(passing, key=lambda k: candidates[k]['worst_processed_auc']) if passing and not refused else None
    for name, rule in rules.items():
        rule.save(OUTPUT/(name.replace('/', '_')+'_rule.json'))
    write_json(path, {'candidates': candidates, 'development_chosen': chosen, 'interpretation_refused': refused,
                      'elapsed_s': perf_counter()-start, 'single_batch_scores_exact': True,
                      'inventory_sha256': receipt['inventory_sha256'], 'controls_sha256': file_sha256(OUTPUT/'controls.json'),
                      'envelope_receipt_sha256': file_sha256(ENVELOPE_DIR/'features.json'), 'code_pins': code_pins(),
                      'rule_files': {k: file_sha256(OUTPUT/(k.replace('/', '_')+'_rule.json')) for k in rules},
                      'scope': 'Gaussian mechanism diagnostic on repeatedly exposed development; no independent validation'})


if __name__ == '__main__':
    main()
