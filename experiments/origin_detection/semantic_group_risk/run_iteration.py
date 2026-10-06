"""Registered group/paired convex objectives on the signed CLIP representation."""

from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.group_margin import fit_group_margin
from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.features import feature_views
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.semantic_kernel.run_iteration import inputs
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS
from experiments.origin_detection.paired_stability.fit_rules import paired_deltas
from experiments.origin_detection.group_margin.run_iteration import wrong_pairs
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals

OUTPUT = WORK_DIR / 'robust_statistics/semantic_group_risk'
CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}


def code_pins():
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent / 'README.md']
    paths += [REPO_ROOT / p for p in (
        'src/palimpsest/detection/algorithms/group_margin.py',
        'src/palimpsest/detection/algorithms/paired_stability.py',
        'src/palimpsest/evaluation/source_training.py',
        'src/palimpsest/evaluation/balanced_null.py',
        'src/palimpsest/evaluation/robust_views.py',
        'experiments/origin_detection/semantic_kernel/run_iteration.py',
        'experiments/origin_detection/paired_stability/fit_rules.py',
        'experiments/origin_detection/group_margin/run_iteration.py',
        'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        'tests/detection/test_group_margin.py')]
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)}


def fit_rule(rows, mode, manifest_sha):
    records = sorted([r for r in rows if r['role'] == 'fit' and r['variant'] in VARIANTS[:2]],
                     key=lambda r: tuple(r[k] for k in ('domain', 'src', 'condition', 'variant')))
    x, y, weights, _ = weighted_source_arrays(rows, FEATURE_NAMES, VARIANTS[:2],
        expected_source_counts={'rr': 540, 'chimera': 720}, seed=CONFIG['seed'])
    traversal = np.array([[float(r[n]) for n in FEATURE_NAMES] for r in records])
    if not np.array_equal(traversal, x) or not np.array_equal(y, [int(r['label'] == 'FAKE') for r in records]):
        raise ValueError('Group metadata/feature traversal differs')
    keys = [tuple(r[k] for k in ('domain', 'scene', 'condition', 'variant', 'label')) for r in records]
    unique = sorted(set(keys))
    if len(unique) != 48:
        raise ValueError('Expected48 class-conditioned groups')
    mapping = {k: i for i, k in enumerate(unique)}
    groups = np.array([mapping[k] for k in keys])
    if mode == 'wrong_pair_both':
        delta, pw, pair_control = wrong_pairs(rows, FEATURE_NAMES, CONFIG['seed'])
    else:
        delta, pw = paired_deltas(rows, 'fit', VARIANTS[:2], FEATURE_NAMES)
        pair_control = {'pairs': len(delta), 'same_source_pairs': len(delta)}
    if len(delta) != 6300:
        raise ValueError('Expected6300 weak source pairs')
    temperature = .1 if mode in ('group', 'both', 'wrong_pair_both', 'balanced_null_both') else 0
    strength = .1 if mode in ('pair', 'both', 'wrong_pair_both', 'balanced_null_both') else 0
    rule, diagnostic = fit_group_margin(x, y, weights, groups, delta, pw,
        feature_names=FEATURE_NAMES, ridge=.01, strength=strength, temperature=temperature,
        scale_floor=.001, maximum_iterations=500, gradient_tolerance=1e-5, manifest_sha=manifest_sha)
    views = {k + '/' + v: view for v in VARIANTS[:2] for processed in (False, True)
             for k, view in feature_views(rows, FEATURE_NAMES, 'threshold', processed=processed, variant=v).items()}
    if len(views) != 24:
        raise ValueError('Expected24 weak calibration views')
    rule, diagnostic['calibration'] = class_threshold(rule, views)
    diagnostic.update({'group_keys': unique, 'pair_control': pair_control})
    return rule, diagnostic


def check_mean_reference(scores):
    parent = WORK_DIR / 'robust_statistics/source_view_risk'
    receipt = json.loads((parent / 'iteration.json').read_text())
    path = parent / 'selection_scores.json'
    if receipt['selection_scores_sha256'] != file_sha256(path):
        raise ValueError('Signed mean reference changed')
    reference = json.loads(path.read_text())['weak/mean']
    key = lambda r: tuple(r[k] for k in ('domain', 'src', 'condition', 'variant'))
    left, right = {key(r): r for r in scores}, {key(r): r for r in reference}
    if len(left) != 5040 or set(left) != set(right):
        raise ValueError('Mean reference coverage changed')
    mismatches = sum((left[k]['score'] > 0) != (right[k]['score'] > 0) for k in left)
    if mismatches:
        raise ValueError('Equivalent mean solver changed registered decisions')
    return {'decisions': len(left), 'mismatches': mismatches,
            'maximum_margin_discrepancy': max(abs(left[k]['score'] - right[k]['score']) for k in left),
            'reference_scores_sha256': file_sha256(path), 'shared': 'same signed features/objective/evaluation'}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'iteration.json').exists():
        raise FileExistsError('Preserve semantic group-risk campaign')
    controls = json.loads((OUTPUT / 'software_controls.json').read_text())
    if not controls['passed'] or any(file_sha256(REPO_ROOT / k) != v for k, v in controls['test_pins'].items()):
        raise ValueError('Group-risk known control changed')
    rows, parent = inputs()
    assignment, control = balanced_source_null([r for r in rows if r['role'] == 'fit'], seed=CONFIG['seed'])
    prior = json.loads((WORK_DIR / 'robust_statistics/balanced_null/assignment.json').read_text())
    if control['assignment_sha256'] != prior['assignment_sha256']:
        raise ValueError('Balanced assignment changed')
    results, rules, scores = {}, {}, {}
    start = perf_counter()
    with threadpool_limits(limits=1):
        for mode in ('mean', 'group', 'pair', 'both', 'wrong_pair_both', 'balanced_null_both'):
            selected = [{**r, 'label': 'FAKE' if assignment[r['domain'], r['src']] else 'REAL'}
                        if r['role'] == 'fit' else r for r in rows] if mode == 'balanced_null_both' else rows
            rule, diagnostic = fit_rule(selected, mode, parent['inventory_sha256'])
            margins, result = score_rule(rows, FEATURE_NAMES, rule, CONFIG)
            if mode == 'mean':
                reference_gate = check_mean_reference(margins)
            result.update({'fit_diagnostics': diagnostic, 'threshold': rule.threshold})
            results[mode], rules[mode], scores[mode] = result, rule, margins
            print(json.dumps({'candidate': mode, 'minimum_ba': result['minimum_domain_ba'],
                'minimum_scene_ba': result['minimum_scene_ba'],
                'maximum_scene_drop': result['maximum_any_scene_drop']}), flush=True)
    groups = defaultdict(list)
    for r in scores['balanced_null_both']:
        if r['variant'] == 'raw' and r['condition'] != 'original':
            groups[r['domain'] + '/' + r['condition']].append(r)
    intervals = {}
    for key, records in groups.items():
        records.sort(key=lambda r: r['src'])
        intervals[key] = auc_intervals({key: np.array([r['score'] for r in records])},
            np.array([int(r['label'] == 'FAKE') for r in records]), CONFIG)[key]
    for k, rule in rules.items():
        rule.save(OUTPUT / (k + '_rule.json'))
    write_json(OUTPUT / 'selection_scores.json', scores)
    write_json(OUTPUT / 'iteration.json', {'candidates': results, 'code_pins': code_pins(),
        'balanced_assignment_sha256': control['assignment_sha256'], 'mean_reference_gate': reference_gate,
        'null_raw_processed_auc_ci95': intervals, 'elapsed_s': perf_counter()-start,
        'inventory_sha256': parent['inventory_sha256'],
        'selection_scores_sha256': file_sha256(OUTPUT / 'selection_scores.json'),
        'rule_files': {k: file_sha256(OUTPUT / (k + '_rule.json')) for k in rules},
        'goal_achieved': False, 'scope': 'Frozen CLIP group/paired conventional objectives;exposed development'})


if __name__ == '__main__':
    main()
