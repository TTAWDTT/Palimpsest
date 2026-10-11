"""Classical covariance heads on two signed frozen semantic representations."""

from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.gaussian_readout import fit_gaussian_rule
from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES as CLIP_NAMES
from palimpsest.detection.representations.frozen_cure import FULL_NAMES as CURE_NAMES
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.features import feature_views
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.semantic_kernel.run_iteration import inputs as clip_inputs
from experiments.origin_detection.cure_readout.prepare_features import OUTPUT as CURE, code_pins as cure_pins, audit_cache
from experiments.origin_detection.source_view_risk.run_iteration import parent_data, VARIANTS
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals

OUTPUT = WORK_DIR / 'robust_statistics/semantic_gaussian'
CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}


def code_pins():
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent / 'README.md']
    paths += [REPO_ROOT / p for p in (
        'src/palimpsest/detection/algorithms/gaussian_readout.py',
        'src/palimpsest/evaluation/source_training.py', 'src/palimpsest/evaluation/balanced_null.py',
        'src/palimpsest/evaluation/features.py', 'src/palimpsest/evaluation/robust_views.py',
        'experiments/origin_detection/semantic_kernel/run_iteration.py',
        'experiments/origin_detection/cure_readout/prepare_features.py',
        'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        'tests/detection/test_gaussian_readout.py')]
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)}


def cure_inputs():
    old, inventory, parent, _ = parent_data()
    del old
    receipt = json.loads((CURE / 'features.json').read_text())
    if (receipt['code_pins'] != cure_pins() or receipt['csv_sha256'] != file_sha256(CURE / 'features.csv')
            or receipt['inventory_sha256'] != parent['inventory_sha256']
            or not receipt['original_probability_exact'] or receipt['matched_original_probabilities'] != 5040
            or not receipt['wrong_probability_rejected'] or receipt['records'] != 15120):
        raise ValueError('Signed frozen CuRe representation changed')
    rows = read_rows(CURE / 'features.csv')
    audit_cache(rows, inventory)
    return rows, parent


def fit_rule(rows, names, mode, manifest_sha):
    x, y, weights, sources = weighted_source_arrays(rows, names, VARIANTS[:2],
        expected_source_counts={'rr': 540, 'chimera': 720}, seed=CONFIG['seed'])
    if len(x) != 7560 or len(set(sources.tolist())) != 1260:
        raise ValueError('Fixed weak support count changed')
    rule, diagnostic = fit_gaussian_rule(x, y, weights, feature_names=names, mode=mode,
        ridge=.1, shrinkage=.5, scale_floor=.001, manifest_sha=manifest_sha)
    views = {k + '/' + v: view for v in VARIANTS[:2] for processed in (False, True)
             for k, view in feature_views(rows, names, 'threshold', processed=processed, variant=v).items()}
    if len(views) != 24:
        raise ValueError('Fixed weak calibration count changed')
    rule, diagnostic['calibration'] = class_threshold(rule, views)
    diagnostic['sources'] = 1260
    return rule, diagnostic


def null_intervals(records):
    groups = defaultdict(list)
    for row in records:
        if row['variant'] == 'raw' and row['condition'] != 'original':
            groups[row['domain'] + '/' + row['condition']].append(row)
    intervals = {}
    for key, selected in groups.items():
        selected.sort(key=lambda r: r['src'])
        intervals[key] = auc_intervals({key: np.array([r['score'] for r in selected])},
            np.array([int(r['label'] == 'FAKE') for r in selected]), CONFIG)[key]
    return intervals


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / 'iteration.json').exists():
        raise FileExistsError('Preserve semantic Gaussian campaign')
    controls_path = OUTPUT / 'software_controls.json'
    controls = json.loads(controls_path.read_text())
    if not controls['passed'] or any(file_sha256(REPO_ROOT / k) != v for k, v in controls['test_pins'].items()):
        raise ValueError('Gaussian known controls changed')
    prior = json.loads((WORK_DIR / 'robust_statistics/balanced_null/assignment.json').read_text())
    results, rules, scores, intervals, receipts = {}, {}, {}, {}, {}
    manifest_sha = None
    start = perf_counter()
    with threadpool_limits(limits=1):
        for prefix, names, loader in [('clip', CLIP_NAMES, clip_inputs), ('cure', CURE_NAMES, cure_inputs)]:
            rows, parent = loader()
            if manifest_sha is not None and manifest_sha != parent['inventory_sha256']:
                raise ValueError('Representation source inventory differs')
            manifest_sha = parent['inventory_sha256']
            assignment, control = balanced_source_null([r for r in rows if r['role'] == 'fit'], seed=CONFIG['seed'])
            if control['assignment_sha256'] != prior['assignment_sha256']:
                raise ValueError('Balanced source assignment changed')
            receipts[prefix] = {'inventory_sha256': manifest_sha, 'balanced_assignment_sha256': control['assignment_sha256']}
            for mode in ('pooled', 'diagonal', 'class_full', 'class_full_null'):
                key = prefix + '/' + mode
                selected = [{**r, 'label': 'FAKE' if assignment[r['domain'], r['src']] else 'REAL'}
                            if r['role'] == 'fit' else r for r in rows] if mode.endswith('_null') else rows
                rule, diagnostic = fit_rule(selected, names, mode.removesuffix('_null'), manifest_sha)
                margins, result = score_rule(rows, names, rule, CONFIG)
                result.update({'fit_diagnostics': diagnostic, 'threshold': rule.threshold})
                results[key], rules[key], scores[key] = result, rule, margins
                if mode.endswith('_null'):
                    intervals[key] = null_intervals(margins)
                print(json.dumps({'candidate': key, 'minimum_ba': result['minimum_domain_ba'],
                    'minimum_scene_ba': result['minimum_scene_ba'],
                    'maximum_scene_drop': result['maximum_any_scene_drop']}), flush=True)
            del rows, selected
    for key, rule in rules.items():
        rule.save(OUTPUT / (key.replace('/', '_') + '_rule.json'))
    write_json(OUTPUT / 'selection_scores.json', scores)
    write_json(OUTPUT / 'iteration.json', {'candidates': results, 'code_pins': code_pins(),
        'software_controls_sha256': file_sha256(controls_path), 'input_receipts': receipts,
        'input_feature_digests': {
            'cure': file_sha256(CURE / 'features.csv'),
            'clip_original': file_sha256(WORK_DIR / 'robust_statistics/frozen_clip/features.csv'),
            'clip_extra': file_sha256(WORK_DIR / 'robust_statistics/source_view_risk/features.csv')},
        'null_raw_processed_auc_ci95': intervals, 'elapsed_s': perf_counter()-start,
        'inventory_sha256': manifest_sha, 'selection_scores_sha256': file_sha256(OUTPUT / 'selection_scores.json'),
        'rule_files': {k: file_sha256(OUTPUT / (k.replace('/', '_') + '_rule.json')) for k in rules},
        'goal_achieved': False, 'scope': 'Fixed Gaussian moments on signed frozen semantic features;exposed development'})


if __name__ == '__main__':
    main()
