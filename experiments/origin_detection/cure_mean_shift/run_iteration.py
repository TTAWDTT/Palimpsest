"""Finite conditional means with collapsed single-image linear readouts."""

from collections import defaultdict
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.mean_shift_projection import fit_mean_shift
from palimpsest.detection.algorithms.source_view_risk import fit_source_risk
from palimpsest.detection.representations.frozen_cure import FULL_NAMES
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.features import feature_views
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.semantic_gaussian.run_iteration import cure_inputs, null_intervals
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold

OUTPUT = WORK_DIR/'robust_statistics/cure_mean_shift'
CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}
EXPECTED_VIEWS = tuple(c+'/'+v for c in ('original', 'processed1', 'processed2') for v in VARIANTS[:2])


def code_pins():
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent/'README.md']
    paths += [REPO_ROOT/p for p in (
        'src/palimpsest/detection/algorithms/mean_shift_projection.py',
        'src/palimpsest/detection/algorithms/source_view_risk.py',
        'src/palimpsest/detection/algorithms/paired_stability.py',
        'src/palimpsest/evaluation/source_training.py', 'src/palimpsest/evaluation/balanced_null.py',
        'src/palimpsest/evaluation/robust_views.py', 'src/palimpsest/evaluation/features.py',
        'experiments/origin_detection/semantic_gaussian/run_iteration.py',
        'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        'tests/detection/test_mean_shift_projection.py', 'tests/detection/test_mean_shift_campaign.py',
        'tests/detection/test_source_view_risk.py')]
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)}


def panel_views(rows, *, corrupt=False):
    # Exactly the same ordering as weighted_source_arrays; no hidden row shuffle.
    fit = sorted([r for r in rows if r['role'] == 'fit' and r['variant'] in VARIANTS[:2]],
                 key=lambda r: tuple(r[k] for k in ('domain', 'src', 'condition', 'variant')))
    conditions = defaultdict(set)
    for row in fit:
        conditions[row['domain'], row['scene']].add(row['condition'])
    lookup = {}
    for panel, actual in conditions.items():
        processed = sorted(actual-{'original'})
        if len(processed) != 2 or 'original' not in actual:
            raise ValueError('Fixed three-condition panel changed')
        lookup[panel] = {'original': 'original', **dict(zip(processed, ('processed1', 'processed2')))}
    panels = [r['domain']+'/'+r['scene'] for r in fit]
    views = [lookup[r['domain'], r['scene']][r['condition']]+'/'+r['variant'] for r in fit]
    if corrupt:
        rng = random.Random(CONFIG['seed'])
        sources = defaultdict(list)
        for i, r in enumerate(fit):
            sources[r['domain'], r['src']].append(i)
        for indices in sources.values():
            shuffled = [views[i] for i in indices]
            rng.shuffle(shuffled)
            for i, v in zip(indices, shuffled):
                views[i] = v
    return fit, panels, views


def main():
    if (OUTPUT/'iteration.json').exists():
        raise FileExistsError('Preserve finite mean-shift campaign')
    controls_path = OUTPUT/'software_controls.json'
    controls = json.loads(controls_path.read_text())
    if not controls['passed'] or controls['code_pins'] != code_pins():
        raise ValueError('Mean-shift planted controls changed')
    rows, parent = cure_inputs()
    assignment, control = balanced_source_null([r for r in rows if r['role'] == 'fit'], seed=CONFIG['seed'])
    prior = json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if control['assignment_sha256'] != prior['assignment_sha256']:
        raise ValueError('Balanced mean-shift assignment changed')
    start = perf_counter()
    results, rules, maps, scores = {}, {}, {}, {}
    selected = [r for r in rows if r['role'] == 'selection']
    native = np.array([[float(r[n]) for n in FULL_NAMES] for r in selected])
    with threadpool_limits(limits=1):
        for key, temperature, sham, corrupt in [('source', .1, False, False), ('mean', 0, False, False),
                                               ('source_null', .1, True, False), ('wrong_processing', .1, False, True)]:
            fitting = [{**r, 'label': 'FAKE' if assignment[r['domain'], r['src']] else 'REAL'}
                       if r['role'] == 'fit' else r for r in rows] if sham else rows
            x, y, weights, sources = weighted_source_arrays(fitting, FULL_NAMES, VARIANTS[:2],
                expected_source_counts={'rr': 540, 'chimera': 720}, seed=CONFIG['seed'])
            fit, panels, views = panel_views(fitting, corrupt=corrupt)
            if not np.array_equal(x, [[float(r[n]) for n in FULL_NAMES] for r in fit]):
                raise ValueError('Panel labels and fit values out of order')
            mapper, diagnostic = fit_mean_shift(x, y, weights, sources, panels, views,
                feature_names=FULL_NAMES, reference_view='original/raw', expected_views=EXPECTED_VIEWS)
            if diagnostic['shift_vectors'] != 40:
                raise ValueError('Fixed40 class mean shifts changed')
            mapped = mapper.transform(x)
            head, fit_diagnostic = fit_source_risk(mapped, y, weights, sources,
                feature_names=mapper.output_names, temperature=temperature, ridge=.01, scale_floor=.001,
                maximum_iterations=500, gradient_tolerance=1e-5, manifest_sha=parent['inventory_sha256'])
            rule = mapper.collapse(head)
            fit_error = float(np.max(np.abs(rule.score(x)-head.score(mapped))))
            calibration = {k+'/'+v: view for v in VARIANTS[:2] for p in (False, True)
                for k, view in feature_views(rows, FULL_NAMES, 'threshold', processed=p, variant=v).items()}
            if len(calibration) != 24:
                raise ValueError('Mean-shift threshold groups changed')
            rule, fit_diagnostic['calibration'] = class_threshold(rule, calibration)
            before = head.score(mapper.transform(native))
            after = rule.score(native)
            selection_error = float(np.max(np.abs(before-after)))
            if max(fit_error, selection_error) > 1e-9 or not np.array_equal(before > rule.threshold, after > rule.threshold):
                raise ValueError('Collapsed mean-shift margins/decisions differ')
            margins, result = score_rule(rows, FULL_NAMES, rule, CONFIG)
            result.update({'map_diagnostics': diagnostic, 'fit_diagnostics': fit_diagnostic,
                           'maximum_collapse_difference': max(fit_error, selection_error),
                           'collapse_decisions_exact': True, 'threshold': rule.threshold})
            results[key], rules[key], maps[key], scores[key] = result, rule, mapper, margins
            print(json.dumps({'candidate': key, 'minimum_ba': result['minimum_domain_ba'],
                'minimum_scene_ba': result['minimum_scene_ba'], 'maximum_scene_drop': result['maximum_any_scene_drop'],
                'removed_rank': diagnostic['removed_rank']}), flush=True)
    for key, rule in rules.items():
        rule.save(OUTPUT/(key+'_rule.json'))
        maps[key].save(OUTPUT/(key+'_map.json'))
    write_json(OUTPUT/'selection_scores.json', scores)
    write_json(OUTPUT/'iteration.json', {'candidates': results, 'code_pins': code_pins(),
        'software_controls_sha256': file_sha256(controls_path), 'elapsed_s': perf_counter()-start,
        'inventory_sha256': parent['inventory_sha256'], 'balanced_assignment_sha256': control['assignment_sha256'],
        'null_raw_processed_auc_ci95': null_intervals(scores['source_null']),
        'selection_scores_sha256': file_sha256(OUTPUT/'selection_scores.json'),
        'rule_files': {k: file_sha256(OUTPUT/(k+'_rule.json')) for k in rules},
        'map_files': {k: file_sha256(OUTPUT/(k+'_map.json')) for k in maps},
        'goal_achieved': False, 'scope': 'Finite conditional means;collapsed linear score;exposed development'})


if __name__ == '__main__':
    main()
