"""Fixed four-way convex objective ablation, wrong pairs, and shuffled truth."""

import argparse
from collections import defaultdict
from fractions import Fraction
import json
from time import perf_counter
import tomllib

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.group_margin import fit_group_margin
from palimpsest.evaluation.features import feature_views
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.conditional_residual.run_iteration import final_target_screen
from experiments.origin_detection.gaussian_readout.run_iteration import signed_inputs, NAMES
from experiments.origin_detection.paired_stability.fit_rules import paired_deltas
from experiments.origin_detection.paired_stability.run_iteration import INTERVENTION
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.residual_statistics.run_iteration import screen_rule, aggregate
from experiments.origin_detection.residual_training.fit_rules import training_arrays, TRAINING_VARIANT
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.threshold_diagnostics.audit_rate_boundaries import audit

CONFIG = REPO_ROOT/'configs/evaluation/group_margin.toml'
OUTPUT = WORK_DIR/'robust_statistics/group_margin'
FEATURE_NAMES = NAMES['hybrid']


def code_pins():
    files = [CONFIG, REPO_ROOT/'src/palimpsest/detection/algorithms/group_margin.py',
             REPO_ROOT/'tests/detection/test_group_margin.py',
             REPO_ROOT/'experiments/origin_detection/gaussian_readout/run_iteration.py',
             REPO_ROOT/'experiments/origin_detection/paired_stability/fit_rules.py',
             REPO_ROOT/'experiments/origin_detection/threshold_diagnostics/audit_rate_boundaries.py']
    directory = REPO_ROOT/'experiments/origin_detection/group_margin'
    files.extend(directory.glob('*.py')); files.append(directory/'README.md')
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(files)}


def wrong_pairs(rows, names, seed):
    fitted = [r for r in rows if r['role'] == 'fit']
    reference = {r['src']: r for r in fitted if r['condition'] == 'original' and r['variant'] == 'raw'}
    buckets = defaultdict(list)
    counts = {d: sum(r['domain'] == d for r in reference.values()) for d in ('rr', 'chimera')}
    for r in fitted:
        if r['condition'] == 'original' and r['variant'] == 'raw':
            continue
        key = tuple(r[k] for k in ('domain', 'scene', 'label', 'condition', 'variant'))
        buckets[key].append(r)
    rng = np.random.default_rng(seed)
    deltas, weights = [], []
    for key, records in sorted(buckets.items()):
        records.sort(key=lambda r: r['src'])
        offset = int(rng.integers(1, len(records)))
        for index, r in enumerate(records):
            donor = reference[records[(index+offset) % len(records)]['src']]
            if donor['src'] == r['src'] or any(donor[k] != r[k] for k in ('domain', 'scene', 'label')):
                raise ValueError('Wrong-pair control failed marginal/source constraints')
            deltas.append([float(r[n])-float(donor[n]) for n in names])
            weights.append(.5/(5*counts[r['domain']]))
    if len(deltas) != 5*len(reference):
        raise ValueError('Wrong-pair coverage differs')
    return np.asarray(deltas), np.asarray(weights), {'pairs': len(deltas), 'same_source_pairs': 0,
                                                    'buckets': len(buckets), 'source_marginals_preserved': True}


def fit_rule(rows, mode, config, manifest_sha, *, calibrate=True):
    shuffled = mode == 'source_shuffled_null'
    _, y, weights, expected_ids = training_arrays(rows, 'equal_domain_augmented', config['seed'], shuffled=shuffled)
    matrices, records, ids = [], [], []
    for variant in ('raw', TRAINING_VARIANT):
        views = {**feature_views(rows, FEATURE_NAMES, 'fit', processed=False, variant=variant),
                 **feature_views(rows, FEATURE_NAMES, 'fit', processed=True, variant=variant)}
        for selected, x, _ in views.values():
            matrices.append(x); records.extend(selected)
            ids.extend((r['src'], r['domain'], variant) for r in selected)
    if ids != expected_ids:
        raise ValueError('Margin fit traversal differs')
    keys = [tuple(r[k] for k in ('domain', 'scene', 'condition', 'variant'))+(int(label),) for r, label in zip(records, y)]
    unique = sorted(set(keys)); group_index = {key: index for index, key in enumerate(unique)}
    groups = np.array([group_index[key] for key in keys])
    if len(unique) != 48:
        raise ValueError('Expected 48 class/condition/encoding groups')
    if mode == 'wrong_pair_both':
        delta, pw, pair_control = wrong_pairs(rows, FEATURE_NAMES, config['seed'])
    else:
        delta, pw = paired_deltas(rows, 'fit', ('raw', TRAINING_VARIANT), FEATURE_NAMES)
        pair_control = {'pairs': len(delta), 'same_source_pairs': len(delta)}
    strength = config['pair_strength'] if mode in ('pair', 'both', 'wrong_pair_both') else 0.
    temperature = config['group_temperature'] if mode in ('group', 'both', 'wrong_pair_both') else 0.
    rule, diagnostic = fit_group_margin(np.concatenate(matrices), y, weights, groups, delta, pw,
                                        feature_names=FEATURE_NAMES, ridge=config['ridge'], strength=strength,
                                        temperature=temperature, scale_floor=config['scale_floor'],
                                        maximum_iterations=config['maximum_iterations'],
                                        gradient_tolerance=config['gradient_tolerance'], manifest_sha=manifest_sha)
    diagnostic.update({'sources': len({s for s, _, _ in ids}), 'group_keys': unique, 'pair_control': pair_control})
    if calibrate:
        views = {key+'/'+variant: view for variant in ('raw', TRAINING_VARIANT) for processed in (False, True)
                 for key, view in feature_views(rows, FEATURE_NAMES, 'threshold', processed=processed, variant=variant).items()}
        rule, diagnostic['calibration'] = class_threshold(rule, views)
    return rule, diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    path = OUTPUT/('pilot.json' if args.pilot else 'iteration.json')
    if path.exists():
        raise FileExistsError('Preserve group-margin evaluation')
    config = tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    controls = json.loads((OUTPUT/'controls.json').read_text(encoding='utf-8'))
    if controls['returncode'] != 0 or controls['test_sha256'] != file_sha256(REPO_ROOT/'tests/detection/test_group_margin.py'):
        raise ValueError('Group margin controls failed or changed')
    rows, held, receipt = signed_inputs()
    start = perf_counter()
    with threadpool_limits(limits=1):
        if args.pilot:
            _, diagnostic = fit_rule(rows, 'erm', config, receipt['inventory_sha256'], calibrate=False)
            write_json(path, {'elapsed_s': perf_counter()-start, 'fit_diagnostics': diagnostic, 'code_pins': code_pins(),
                              'scope': 'Fixed ERM full-fit cost pilot; no selection screening'})
            print(json.dumps({'pilot_s': perf_counter()-start}), flush=True)
            return
        results, rules = {}, {}
        for mode in ('erm', 'pair', 'group', 'both', 'wrong_pair_both', 'source_shuffled_null'):
            rule, diagnostic = fit_rule(rows, mode, config, receipt['inventory_sha256'])
            value = screen_rule(rows, FEATURE_NAMES, rule, config)
            encoded = aggregate(held, FEATURE_NAMES, rule, INTERVENTION)
            changes = [(v['balanced_accuracy_at_zero'], encoded[d]['conditions'][c]['balanced_accuracy_at_zero'])
                       for d, z in value['selection_aggregate'].items() for c, v in z['conditions'].items() if c != 'original']
            value['criteria'].update({
                'held_encoding_ba_floor': all(b >= config['minimum_reencoded_ba'] for _, b in changes),
                'held_encoding_ba_drop': all(a-b <= config['maximum_reencoded_ba_drop'] for a, b in changes),
                'both_encoded_class_accuracies': all(v[k] >= .55 for data in (value['recompression_aggregate'], encoded)
                                                     for z in data.values() for v in z['conditions'].values()
                                                     for k in ('real_accuracy_at_zero', 'fake_accuracy_at_zero'))})
            value.update({'held_encoding_aggregate': encoded, 'fit_diagnostics': diagnostic,
                          'final_target': final_target_screen(value, encoded, config)})
            exact = audit({'candidates': {mode: value}}, Fraction(str(config['maximum_reencoded_ba_drop'])))[mode]
            value['floating_point_criteria'] = value['criteria']
            value['criteria'] = exact['corrected_criteria']; value['gate_passed'] = exact['corrected_gate_passed']
            value['rate_boundary_audit'] = exact['comparisons']
            for variant, data in [('raw', rows), (TRAINING_VARIANT, rows), (INTERVENTION, held)]:
                x = np.asarray([[float(r[n]) for n in FEATURE_NAMES] for r in data if r['role'] == 'selection' and r['variant'] == variant])
                if not np.array_equal(rule.score(x), [rule.score([v])[0] for v in x]):
                    raise ValueError('Margin single/batch scoring differs')
            results[mode] = value; rules[mode] = rule
            print(json.dumps({'candidate': mode, 'gate': value['gate_passed'], 'target': value['final_target']}), flush=True)
    refused = results['source_shuffled_null']['criteria']['eight_auc_lower_bounds']
    passing = [key for key in ('erm', 'pair', 'group', 'both') if results[key]['gate_passed']]
    chosen = max(passing, key=lambda key: results[key]['worst_processed_auc']) if passing and not refused else None
    for key, rule in rules.items():
        rule.save(OUTPUT/f'{key}_rule.json')
    write_json(path, {'candidates': results, 'development_chosen': chosen, 'interpretation_refused': refused,
                      'elapsed_s': perf_counter()-start, 'single_batch_scores_exact': True,
                      'inventory_sha256': receipt['inventory_sha256'], 'controls_sha256': file_sha256(OUTPUT/'controls.json'),
                      'code_pins': code_pins(), 'rule_files': {key: file_sha256(OUTPUT/f'{key}_rule.json') for key in rules},
                      'scope': 'Four fixed objectives plus two controls on exposed development; no independent validation'})


if __name__ == '__main__':
    main()
