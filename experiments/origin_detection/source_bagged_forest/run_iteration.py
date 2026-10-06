"""Fixed source versus row bootstrap forests on signed frozen representations."""

from collections import defaultdict
import json
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.source_bagged_forest import fit_source_forest
from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES as CLIP_NAMES
from palimpsest.detection.representations.frozen_dinov2_small import FEATURE_NAMES as DINO_NAMES
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.signed_feature_inputs import load_signed_columns
from palimpsest.evaluation.pixel_features import join_features
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.evaluation.features import feature_views
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.source_view_risk.run_iteration import parent_data, VARIANTS, Q60
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals

OUTPUT = WORK_DIR / 'robust_statistics/source_bagged_forest'
NAMES = CLIP_NAMES + DINO_NAMES
CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}


def code_pins():
    paths = [REPO_ROOT / 'src/palimpsest/detection/algorithms/source_bagged_forest.py',
             REPO_ROOT / 'tests/detection/test_source_bagged_forest.py']
    paths += [REPO_ROOT / 'src/palimpsest/evaluation' / n for n in
              ('balanced_null.py', 'signed_feature_inputs.py', 'source_training.py', 'robust_views.py')]
    paths += [REPO_ROOT / 'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
              REPO_ROOT / 'experiments/origin_detection/threshold_calibration/fit_threshold.py']
    paths += [REPO_ROOT / 'experiments/origin_detection/source_bagged_forest' / n for n in
              ('README.md', 'run_iteration.py')]
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in paths}


def main():
    destination = OUTPUT / 'iteration.json'
    if destination.exists():
        raise FileExistsError('Preserve source forest campaign')
    controls = json.loads((OUTPUT / 'software_controls.json').read_text())
    if controls['returncode'] != 0 or any(file_sha256(REPO_ROOT / k) != v for k, v in controls['test_pins'].items()):
        raise ValueError('Forest controls changed')
    old, inventory, parent, _ = parent_data()
    del old
    parts, receipts = [], {}
    for slug, names in [('intermediate_encoder', CLIP_NAMES), ('compact_frozen_encoder', DINO_NAMES)]:
        part, sha = load_signed_columns(WORK_DIR / 'robust_statistics' / slug, names, inventory,
            repository_root=REPO_ROOT, inventory_sha=parent['inventory_sha256'],
            ordinary_variants=VARIANTS[:3], selection_variant=Q60)
        parts.append(part)
        receipts[slug] = sha
    rows = join_features(*parts, DINO_NAMES)
    del parts
    records = sorted([r for r in rows if r['role'] == 'fit' and r['variant'] in VARIANTS[:2]],
                     key=lambda r: tuple(r[k] for k in ('domain', 'src', 'condition', 'variant')))
    assignment, assignment_receipt = balanced_source_null(records, seed=CONFIG['seed'])
    prior = json.loads((WORK_DIR / 'robust_statistics/balanced_null/assignment.json').read_text())
    if assignment_receipt['assignment_sha256'] != prior['assignment_sha256']:
        raise ValueError('Forest sham assignment differs')
    x, y, _, sources = weighted_source_arrays(rows, NAMES, VARIANTS[:2],
        expected_source_counts={'rr': 540, 'chimera': 720}, seed=CONFIG['seed'])
    if len(x) != 7560 or len(records) != len(x) or np.any(np.bincount(sources) != 6):
        raise ValueError('Forest weakfit source coverage differs')
    domains = np.array([r['domain'] for r in records])
    sham = np.array([assignment[(r['domain'], r['src'])] for r in records])
    views = {k + '/' + v: view for v in VARIANTS[:2] for processed in (False, True)
             for k, view in feature_views(rows, NAMES, 'threshold', processed=processed, variant=v).items()}
    if len(views) != 24:
        raise ValueError('Forest threshold coverage differs')
    results, scores, files = {}, {}, {}
    start = perf_counter()
    with threadpool_limits(limits=1):
        for key, labels, source_level in [('source_bagged', y, True), ('row_bagged', y, False),
                                          ('balanced_null', sham, True)]:
            candidate_start = perf_counter()
            rule, diagnostic = fit_source_forest(x, labels, sources, domains, feature_names=NAMES,
                seed=CONFIG['seed'], source_level=source_level, trees=128, maximum_depth=12, minimum_leaf=12)
            rule, calibration = class_threshold(rule, views)
            margins, result = score_rule(rows, NAMES, rule, CONFIG)
            result.update({'threshold': rule.threshold, 'calibration': calibration,
                           'fit_diagnostics': diagnostic, 'cached_fit_evaluation_s': perf_counter() - candidate_start})
            path = OUTPUT / (key + '_rule.json')
            rule.save(path)
            files[key] = file_sha256(path)
            results[key], scores[key] = result, margins
            print(json.dumps({'candidate': key, 'minimum_domain_ba': result['minimum_domain_ba'],
                'minimum_scene_ba': result['minimum_scene_ba'],
                'maximum_drop': result['maximum_any_scene_drop']}), flush=True)
    groups = defaultdict(list)
    for row in scores['balanced_null']:
        if row['variant'] == 'raw' and row['condition'] != 'original':
            groups[row['domain'] + '/' + row['condition']].append(row)
    intervals = {}
    for key, records in groups.items():
        records.sort(key=lambda r: r['src'])
        intervals[key] = auc_intervals({key: np.array([r['score'] for r in records])},
            np.array([int(r['label'] == 'FAKE') for r in records]), CONFIG)[key]
    write_json(OUTPUT / 'selection_scores.json', scores)
    write_json(destination, {'candidates': results, 'null_raw_processed_auc_ci95': intervals,
        'elapsed_s': perf_counter() - start, 'parent_receipts': receipts, 'code_pins': code_pins(),
        'balanced_assignment_sha256': assignment_receipt['assignment_sha256'],
        'inventory_sha256': parent['inventory_sha256'],
        'selection_scores_sha256': file_sha256(OUTPUT / 'selection_scores.json'), 'rule_files': files,
        'controls_sha256': file_sha256(OUTPUT / 'software_controls.json'),
        'goal_achieved': False, 'scope': 'Conventional randomized forests on frozen neural features;exposed development'})


if __name__ == '__main__':
    main()
