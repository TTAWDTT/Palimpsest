"""Fixed CuRe representation with conventional processing-conditioned experts."""

from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.source_view_risk import fit_source_risk
from palimpsest.detection.algorithms.condition_mixture import ConditionMixtureRule
from palimpsest.detection.representations.frozen_cure import FULL_NAMES
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.evaluation.features import feature_views
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.semantic_gaussian.run_iteration import cure_inputs, null_intervals
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold

OUTPUT = WORK_DIR / 'robust_statistics/cure_condition_risk'
CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}


def code_pins():
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent / 'README.md']
    paths += [REPO_ROOT / p for p in (
        'src/palimpsest/detection/algorithms/condition_mixture.py',
        'src/palimpsest/detection/algorithms/source_view_risk.py',
        'src/palimpsest/detection/algorithms/paired_stability.py',
        'src/palimpsest/evaluation/source_training.py', 'src/palimpsest/evaluation/balanced_null.py',
        'src/palimpsest/evaluation/features.py', 'src/palimpsest/evaluation/robust_views.py',
        'experiments/origin_detection/semantic_gaussian/run_iteration.py',
        'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        'tests/detection/test_condition_mixture.py')]
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)}


def fit_head(x, y, weights, groups, manifest_sha, temperature):
    return fit_source_risk(x, y, weights, groups, feature_names=FULL_NAMES, temperature=temperature,
        ridge=.01, scale_floor=.001, maximum_iterations=500, gradient_tolerance=1e-5,
        manifest_sha=manifest_sha)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / 'iteration.json'
    if destination.exists():
        raise FileExistsError('Preserve conditional CuRe campaign')
    controls_path = OUTPUT / 'software_controls.json'
    controls = json.loads(controls_path.read_text())
    if not controls['passed'] or any(file_sha256(REPO_ROOT / k) != v for k, v in controls['test_pins'].items()):
        raise ValueError('Probability pool known controls changed')
    rows, parent = cure_inputs()
    ordered = sorted([r for r in rows if r['role'] == 'fit' and r['variant'] in VARIANTS[:2]],
                     key=lambda r: tuple(r[k] for k in ('domain', 'src', 'condition', 'variant')))
    x, y, weights, sources = weighted_source_arrays(rows, FULL_NAMES, VARIANTS[:2],
        expected_source_counts={'rr': 540, 'chimera': 720}, seed=CONFIG['seed'])
    if not np.array_equal(x, [[float(r[n]) for n in FULL_NAMES] for r in ordered]):
        raise ValueError('Processing mask and feature traversal differ')
    processed = np.array([r['condition'] != 'original' for r in ordered])
    for source in range(1260):
        if np.sum((sources == source) & ~processed) != 2 or np.sum((sources == source) & processed) != 4:
            raise ValueError('Original/processed source views changed')
    gate_weights = weights*np.where(processed, .75, 1.5)
    if not np.allclose(np.bincount(processed.astype(int), weights=gate_weights), [.5, .5], atol=1e-12, rtol=0):
        raise ValueError('Gate class mass changed')
    assignment, control = balanced_source_null(ordered, seed=CONFIG['seed'])
    prior = json.loads((WORK_DIR / 'robust_statistics/balanced_null/assignment.json').read_text())
    if control['assignment_sha256'] != prior['assignment_sha256']:
        raise ValueError('Balanced source assignment changed')
    results, rules, scores, intervals, diagnostics = {}, {}, {}, {}, {}
    start = perf_counter()
    with threadpool_limits(limits=1):
        gate, diagnostics['gate'] = fit_head(x, processed.astype(int), gate_weights,
            np.arange(len(x)), parent['inventory_sha256'], 0)
        for null in (False, True):
            labels = np.array([assignment[r['domain'], r['src']] for r in ordered]) if null else y
            original, d0 = fit_head(x[~processed], labels[~processed], weights[~processed], sources[~processed],
                parent['inventory_sha256'], .1)
            altered, d1 = fit_head(x[processed], labels[processed], weights[processed], sources[processed],
                parent['inventory_sha256'], .1)
            diagnostics['null_experts' if null else 'experts'] = {'original': d0, 'processed': d1}
            base = ConditionMixtureRule(original, altered, gate)
            for name, rule in [('soft_gate', base), ('uniform_half', replace(base, uniform=True))]:
                key = name + ('_null' if null else '')
                views = {k + '/' + v: view for v in VARIANTS[:2] for p in (False, True)
                    for k, view in feature_views(rows, FULL_NAMES, 'threshold', processed=p, variant=v).items()}
                if len(views) != 24:
                    raise ValueError('Mixture weak calibration count changed')
                rule, calibration = class_threshold(rule, views)
                margins, result = score_rule(rows, FULL_NAMES, rule, CONFIG)
                result.update({'threshold': rule.threshold, 'calibration': calibration})
                rules[key], scores[key], results[key] = rule, margins, result
                if null:
                    intervals[key] = null_intervals(margins)
                print(json.dumps({'candidate': key, 'minimum_ba': result['minimum_domain_ba'],
                    'minimum_scene_ba': result['minimum_scene_ba'],
                    'maximum_scene_drop': result['maximum_any_scene_drop']}), flush=True)
        selected = [r for r in rows if r['role'] == 'selection' and r['variant'] == 'raw']
        matrix = np.array([[float(r[n]) for n in FULL_NAMES] for r in selected])
        estimates = gate.score(matrix) > 0
    gate_controls = {}
    for domain in ('rr', 'chimera'):
        indices = np.array([i for i, r in enumerate(selected) if r['domain'] == domain])
        labels = np.array([selected[i]['condition'] != 'original' for i in indices])
        gate_controls[domain] = {'images': len(indices),
            'original_accuracy': float(np.mean(~estimates[indices][~labels])),
            'processed_accuracy': float(np.mean(estimates[indices][labels])),
            'scope': 'processing classification only,not origin accuracy or calibrated wild posterior'}
    for key, rule in rules.items():
        rule.save(OUTPUT / (key+'_rule.json'))
    write_json(OUTPUT / 'selection_scores.json', scores)
    write_json(destination, {'candidates': results, 'code_pins': code_pins(), 'gate_controls': gate_controls,
        'software_controls_sha256': file_sha256(controls_path), 'fit_diagnostics': diagnostics,
        'null_raw_processed_auc_ci95': intervals, 'elapsed_s': perf_counter()-start,
        'inventory_sha256': parent['inventory_sha256'], 'balanced_assignment_sha256': control['assignment_sha256'],
        'selection_scores_sha256': file_sha256(OUTPUT / 'selection_scores.json'),
        'rule_files': {k: file_sha256(OUTPUT / (k+'_rule.json')) for k in rules},
        'goal_achieved': False, 'scope': 'Single frozen CuRe encoder;feature-only conventional processing experts;exposed development'})


if __name__ == '__main__':
    main()
