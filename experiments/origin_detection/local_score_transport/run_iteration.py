"""Matched local score corrections around a reused original-only expert."""

from collections import defaultdict
import json
import random
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.condition_mixture import ConditionMixtureRule
from palimpsest.detection.algorithms.local_score_transport import LocalScoreTransport, matched_source_anchors
from palimpsest.detection.algorithms.readouts.source_view_risk import fit_source_risk
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

OUTPUT = WORK_DIR / 'robust_statistics/local_score_transport'
NAMES = CLIP_NAMES + DINO_NAMES
WEIGHTS = (.5,) * 768 + (.25,) * 768
CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}


def code_pins():
    paths = [REPO_ROOT / 'src/palimpsest/detection/algorithms' / n for n in
             ('local_score_transport.py', 'source_view_risk.py', 'paired_stability.py', 'condition_mixture.py')]
    paths += [REPO_ROOT / 'src/palimpsest/evaluation' / n for n in
              ('balanced_null.py', 'signed_feature_inputs.py', 'source_training.py', 'robust_views.py')]
    paths += [REPO_ROOT / 'tests/detection/test_local_score_transport.py',
              REPO_ROOT / 'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
              REPO_ROOT / 'experiments/origin_detection/threshold_calibration/fit_threshold.py']
    paths += [REPO_ROOT / 'experiments/origin_detection/local_score_transport' / n for n in
              ('README.md', 'run_iteration.py')]
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in paths}


def main():
    destination = OUTPUT / 'iteration.json'
    if destination.exists():
        raise FileExistsError('Preserve local transport campaign')
    controls = json.loads((OUTPUT / 'software_controls.json').read_text())
    if controls['returncode'] != 0 or any(file_sha256(REPO_ROOT / k) != v
                                       for k, v in controls['test_pins'].items()):
        raise ValueError('Local transport software controls changed')
    origin_directory = WORK_DIR / 'robust_statistics/condition_mixture'
    origin_receipt = json.loads((origin_directory / 'iteration.json').read_text())
    base_path = origin_directory / 'soft_gate_rule.json'
    if (file_sha256(base_path) != origin_receipt['rule_files']['soft_gate']
            or any(file_sha256(REPO_ROOT / k) != v for k, v in origin_receipt['code_pins'].items())):
        raise ValueError('Reused original expert changed')
    base = ConditionMixtureRule.load(base_path).original
    if base.feature_names != NAMES:
        raise ValueError('Reused original expert feature schema differs')
    old, inventory, parent, _ = parent_data()
    del old
    if origin_receipt['inventory_sha256'] != parent['inventory_sha256']:
        raise ValueError('Reused original expert inventory differs')
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
    assignment, assignment_receipt = balanced_source_null(fit_rows, seed=CONFIG['seed'])
    prior = json.loads((WORK_DIR / 'robust_statistics/balanced_null/assignment.json').read_text())
    if assignment_receipt['assignment_sha256'] != prior['assignment_sha256']:
        raise ValueError('Local transport sham assignment changed')
    x, _, weights, sources = weighted_source_arrays(rows, NAMES, VARIANTS[:2],
        expected_source_counts={'rr': 540, 'chimera': 720}, seed=CONFIG['seed'])
    original = np.array([r['condition'] == 'original' for r in fit_rows])
    if len(x) != 7560 or len(fit_rows) != len(x):
        raise ValueError('Local transport fit coverage changed')
    start = perf_counter()
    bags, anchors = matched_source_anchors(base, x, sources, original)
    # Shuffle anchor identities only within fit domain/scene, preserving the index.
    metadata = {}
    for r, source in zip(fit_rows, sources):
        value = (r['domain'], r['scene'])
        if source in metadata and metadata[source] != value:
            raise ValueError('Local transport source metadata conflicts')
        metadata[source] = value
    strata = defaultdict(list)
    for source, value in sorted(metadata.items()):
        strata[value].append(source)
    shuffled = list(anchors)
    rng = random.Random(CONFIG['seed'])
    for keys in strata.values():
        values = [anchors[k] for k in keys]
        rng.shuffle(values)
        for k, value in zip(keys, values):
            shuffled[k] = value
    sham = np.array([assignment[(r['domain'], r['src'])] for r in fit_rows])
    with threadpool_limits(limits=1):
        null_base, diagnostics = fit_source_risk(x[original], sham[original], weights[original], sources[original],
            feature_names=NAMES, temperature=.1, ridge=.01, scale_floor=.001, maximum_iterations=500,
            gradient_tolerance=1e-5, manifest_sha=parent['inventory_sha256'])
    null_bags, null_anchors = matched_source_anchors(null_base, x, sources, original)
    candidates = [('original_head', base), ('matched_correction', LocalScoreTransport(base, bags, anchors, WEIGHTS)),
                  ('shuffled_anchor_correction', LocalScoreTransport(base, bags, tuple(shuffled), WEIGHTS)),
                  ('balanced_null_correction', LocalScoreTransport(null_base, null_bags, null_anchors, WEIGHTS))]
    views = {k + '/' + v: view for v in VARIANTS[:2] for processed in (False, True)
             for k, view in feature_views(rows, NAMES, 'threshold', processed=processed, variant=v).items()}
    if len(views) != 24:
        raise ValueError('Local transport threshold coverage changed')
    results, scores, files = {}, {}, {}
    with threadpool_limits(limits=1):
        for key, rule in candidates:
            head_start = perf_counter()
            rule, calibration = class_threshold(rule, views)
            margins, result = score_rule(rows, NAMES, rule, CONFIG)
            result.update({'threshold': rule.threshold, 'calibration': calibration,
                           'cached_evaluation_s': perf_counter() - head_start})
            path = OUTPUT / (key + '_rule.json')
            rule.save(path)
            files[key] = {'json': file_sha256(path)}
            if isinstance(rule, LocalScoreTransport):
                files[key]['npz'] = file_sha256(path.with_suffix('.npz'))
            results[key], scores[key] = result, margins
            print(json.dumps({'candidate': key, 'minimum_domain_ba': result['minimum_domain_ba'],
                'minimum_scene_ba': result['minimum_scene_ba'],
                'maximum_drop': result['maximum_any_scene_drop']}), flush=True)
    groups = defaultdict(list)
    for row in scores['balanced_null_correction']:
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
        'reused_original_rule_sha256': file_sha256(base_path),
        'reused_original_component_fingerprint': base.fingerprint, 'null_fit_diagnostics': diagnostics,
        'balanced_assignment_sha256': assignment_receipt['assignment_sha256'],
        'inventory_sha256': parent['inventory_sha256'],
        'selection_scores_sha256': file_sha256(OUTPUT / 'selection_scores.json'), 'rule_files': files,
        'goal_achieved': False, 'scope': 'Finite matched local score correction;not recovered physical information'})


if __name__ == '__main__':
    main()
