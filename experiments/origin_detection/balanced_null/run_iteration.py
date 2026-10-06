"""One fixed truth-orthogonal sham head; preserve the stage27 method unchanged."""

from collections import defaultdict
import json
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.source_view_risk import fit_source_risk
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

OUTPUT = WORK_DIR / 'robust_statistics/balanced_null'
NAMES = CLIP_NAMES + DINO_NAMES
CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}


def code_pins():
    paths = [REPO_ROOT / 'src/palimpsest/evaluation' / n for n in
             ('balanced_null.py', 'signed_feature_inputs.py', 'source_training.py', 'robust_views.py')]
    paths += [REPO_ROOT / 'src/palimpsest/detection/algorithms' / n for n in
              ('source_view_risk.py', 'paired_stability.py')]
    paths += [REPO_ROOT / 'experiments/origin_detection/balanced_null' / n for n in
              ('README.md', 'run_iteration.py')]
    paths += [REPO_ROOT / 'tests/evaluation/test_balanced_null.py',
              REPO_ROOT / 'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
              REPO_ROOT / 'experiments/origin_detection/threshold_calibration/fit_threshold.py']
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in paths}


def main():
    destination = OUTPUT / 'iteration.json'
    if destination.exists():
        raise FileExistsError('Preserve balanced-null campaign')
    controls = json.loads((OUTPUT / 'software_controls.json').read_text())
    if controls['returncode'] != 0 or any(file_sha256(REPO_ROOT / k) != v
                                       for k, v in controls['test_pins'].items()):
        raise ValueError('Balanced-null software controls changed')
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
    fit_rows = sorted([r for r in rows if r['role'] == 'fit' and r['variant'] in VARIANTS[:2]],
                      key=lambda r: tuple(r[k] for k in ('domain', 'src', 'condition', 'variant')))
    assignments, assignment_receipt = balanced_source_null(fit_rows, seed=CONFIG['seed'])
    x, truth, weights, sources = weighted_source_arrays(rows, NAMES, VARIANTS[:2],
        expected_source_counts={'rr': 540, 'chimera': 720}, seed=CONFIG['seed'])
    labels = np.array([assignments[(r['domain'], r['src'])] for r in fit_rows])
    if len(x) != 7560 or len(labels) != len(x) or assignment_receipt['sources'] != 1260:
        raise ValueError('Balanced-null coverage changed')
    for source in range(1260):
        if len(set(labels[sources == source])) != 1:
            raise ValueError('Sham targets differ across source views')
    write_json(OUTPUT / 'assignment.json', assignment_receipt)
    start = perf_counter()
    with threadpool_limits(limits=1):
        rule, diagnostics = fit_source_risk(x, labels, weights, sources, feature_names=NAMES,
            temperature=.1, ridge=.01, scale_floor=.001, maximum_iterations=500,
            gradient_tolerance=1e-5, manifest_sha=parent['inventory_sha256'])
        views = {k + '/' + v: view for v in VARIANTS[:2] for processed in (False, True)
                 for k, view in feature_views(rows, NAMES, 'threshold',
                                             processed=processed, variant=v).items()}
        if len(views) != 24:
            raise ValueError('Balanced-null threshold coverage changed')
        rule, calibration = class_threshold(rule, views)
        margins, result = score_rule(rows, NAMES, rule, CONFIG)
        result.update({'threshold': rule.threshold, 'calibration': calibration})
    groups = defaultdict(list)
    for row in margins:
        if row['variant'] == 'raw' and row['condition'] != 'original':
            groups[row['domain'] + '/' + row['condition']].append(row)
    intervals = {}
    for key, records in groups.items():
        records.sort(key=lambda r: r['src'])
        intervals[key] = auc_intervals({key: np.array([r['score'] for r in records])},
            np.array([int(r['label'] == 'FAKE') for r in records]), CONFIG)[key]
    rule.save(OUTPUT / 'balanced_null_rule.json')
    write_json(OUTPUT / 'selection_scores.json', {'balanced_null': margins})
    write_json(destination, {'candidates': {'balanced_null': result}, 'null_raw_processed_auc_ci95': intervals,
        'fit_diagnostics': diagnostics, 'elapsed_s': perf_counter() - start,
        'assignment_sha256': file_sha256(OUTPUT / 'assignment.json'),
        'assignment_content_sha256': assignment_receipt['assignment_sha256'], 'parent_receipts': receipts,
        'code_pins': code_pins(), 'inventory_sha256': parent['inventory_sha256'],
        'selection_scores_sha256': file_sha256(OUTPUT / 'selection_scores.json'),
        'rule_files': {'balanced_null': file_sha256(OUTPUT / 'balanced_null_rule.json')},
        'goal_achieved': False, 'real_method_refitted': False,
        'scope': 'Fit truth-orthogonal constrained sham;true-label threshold exposed;not a permutation p-value'})
    print(json.dumps({'minimum_domain_ba': result['minimum_domain_ba'],
                     'null_intervals': intervals}), flush=True)


if __name__ == '__main__':
    main()
