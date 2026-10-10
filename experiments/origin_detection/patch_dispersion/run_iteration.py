"""One frozen DINO forward, fixed patch SD descriptor, conventional readouts."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from time import perf_counter

import cv2
import numpy as np

from palimpsest.detection.representations.frozen_patch_dispersion import (
    FEATURE_NAMES, DISPERSION_NAMES, FrozenPatchDispersion,
)
from palimpsest.detection.representations.frozen_dinov2_small import FEATURE_NAMES as BASE_NAMES
from palimpsest.detection.algorithms.residual_statistics.features import resize256
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.feature_prefix import validate_cached_prefix
from palimpsest.evaluation.signed_feature_inputs import load_signed_columns
from palimpsest.evaluation.pixel_features import extract_inventory, audit_variants
from palimpsest.evaluation.source_readout_campaign import fit_readouts
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR, MODELS_ROOT
from experiments.origin_detection.source_view_risk.run_iteration import parent_data, VARIANTS, Q60
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import write_json, image_path
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals

OUTPUT = WORK_DIR / 'robust_statistics/patch_dispersion'
SOURCE = WORK_DIR / 'robust_statistics/compact_encoder_review/runtime_receipt.json'
CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}


def code_pins():
    paths = [REPO_ROOT / 'src/palimpsest/detection/representations' / n for n in
             ('frozen_patch_dispersion.py', 'frozen_dinov2_small.py')]
    paths += [REPO_ROOT / 'src/palimpsest/evaluation' / n for n in
              ('feature_prefix.py', 'balanced_null.py', 'signed_feature_inputs.py',
               'pixel_features.py', 'source_readout_campaign.py', 'source_training.py', 'robust_views.py')]
    paths += [REPO_ROOT / 'src/palimpsest/detection/algorithms/readouts/source_view_risk.py',
              REPO_ROOT / 'tests/detection/test_frozen_patch_dispersion.py',
              REPO_ROOT / 'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
              REPO_ROOT / 'experiments/origin_detection/threshold_calibration/fit_threshold.py']
    paths += [REPO_ROOT / 'experiments/origin_detection/patch_dispersion' / n for n in
              ('README.md', 'run_iteration.py', 'audit_runtime.py')]
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in paths}


def parent_features(inventory, parent):
    return load_signed_columns(WORK_DIR / 'robust_statistics/compact_frozen_encoder', BASE_NAMES, inventory,
        repository_root=REPO_ROOT, inventory_sha=parent['inventory_sha256'],
        ordinary_variants=VARIANTS[:3], selection_variant=Q60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pilot', type=int, default=0)
    group.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.pilot < 0:
        parser.error('Pilot must be positive')
    controls = json.loads((OUTPUT / 'software_controls.json').read_text())
    runtime = json.loads((OUTPUT / 'runtime_controls.json').read_text())
    if (controls['returncode'] != 0 or not runtime['passed']
            or any(file_sha256(REPO_ROOT / k) != v for k, v in controls['test_pins'].items())
            or runtime['script_sha256'] != file_sha256(Path(__file__).with_name('audit_runtime.py'))
            or runtime['source_receipt_sha256'] != file_sha256(SOURCE)):
        raise ValueError('Patch dispersion controls changed')
    old, inventory, parent, _ = parent_data()
    del old
    if args.pilot or args.prepare:
        directory = OUTPUT.with_name('patch_dispersion_pilot') if args.pilot else OUTPUT
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / 'features.json').exists():
            raise FileExistsError('Preserve dispersion extraction')
        reference, parent_sha = parent_features(inventory, parent)
        if args.prepare:
            pilot_path = OUTPUT.with_name('patch_dispersion_pilot') / 'features.json'
            pilot = json.loads(pilot_path.read_text())
            pilot_csv = pilot_path.with_name('features.csv')
            if (pilot['code_pins'] != code_pins() or not pilot['prefix_control']['exact_equal']
                    or pilot['csv_sha256'] != file_sha256(pilot_csv)
                    or not pilot['wrong_prefix_rejected']):
                raise ValueError('Patch dispersion pilot changed')
        if args.pilot:
            random.Random(CONFIG['seed']).shuffle(inventory)
            inventory = inventory[:args.pilot]
        encoder = FrozenPatchDispersion(MODELS_ROOT / 'dinov2_small/dinov2_vits14_pretrain.pth', SOURCE)
        cv2.setNumThreads(1)
        result = extract_inventory(inventory, directory / 'features.csv', names=FEATURE_NAMES,
            extractor=encoder.extract, resolve_path=image_path, resize=resize256,
            ordinary_variants=VARIANTS[:3], selection_variant=Q60,
            jpeg_parameters={VARIANTS[1]: (90, 0), VARIANTS[2]: (70, 2), Q60: (60, 2)})
        fresh = read_rows(directory / 'features.csv')
        prefix = validate_cached_prefix(fresh, reference, BASE_NAMES, full_coverage=not args.pilot)
        bad = [{**fresh[0], BASE_NAMES[0]: float(fresh[0][BASE_NAMES[0]]) + .001}]
        rejected = False
        try:
            validate_cached_prefix(bad, reference, BASE_NAMES, full_coverage=False)
        except ValueError as error:
            if 'values differ' not in str(error):
                raise
            rejected = True
        if not rejected:
            raise ValueError('Wrong patch dispersion prefix not rejected')
        write_json(directory / 'features.json', {**result, 'prefix_control': prefix,
            'wrong_prefix_rejected': True, 'encoder': encoder.provenance, 'code_pins': code_pins(),
            'inventory_sha256': parent['inventory_sha256'], 'parent_features_receipt_sha256': parent_sha,
            'runtime_controls_sha256': file_sha256(OUTPUT / 'runtime_controls.json'),
            'software_controls_sha256': file_sha256(OUTPUT / 'software_controls.json')})
        return
    destination = OUTPUT / 'iteration.json'
    if destination.exists():
        raise FileExistsError('Preserve dispersion head evaluation')
    receipt = json.loads((OUTPUT / 'features.json').read_text())
    if (receipt['code_pins'] != code_pins() or receipt['csv_sha256'] != file_sha256(OUTPUT / 'features.csv')
            or receipt['inventory_sha256'] != parent['inventory_sha256']
            or not receipt['prefix_control']['exact_equal'] or not receipt['wrong_prefix_rejected']):
        raise ValueError('Patch dispersion cache changed')
    rows = read_rows(OUTPUT / 'features.csv')
    audit_variants(rows, inventory, FEATURE_NAMES, ordinary_variants=VARIANTS[:3], selection_variant=Q60)
    start = perf_counter()
    candidates = [('dispersion/source', DISPERSION_NAMES, .1, False),
                  ('joint/source', FEATURE_NAMES, .1, False), ('joint/mean', FEATURE_NAMES, 0, False)]
    results, rules, scores = fit_readouts(rows, candidates, fit_variants=VARIANTS[:2],
        manifest_sha=parent['inventory_sha256'], config=CONFIG, calibrate=class_threshold, score=score_rule)
    assignment, control = balanced_source_null([r for r in rows if r['role'] == 'fit'], seed=CONFIG['seed'])
    old_control = json.loads((WORK_DIR / 'robust_statistics/balanced_null/assignment.json').read_text())
    if control['assignment_sha256'] != old_control['assignment_sha256']:
        raise ValueError('Dispersion sham assignment changed')
    sham_rows = [{**r, 'label': 'FAKE' if assignment[r['domain'], r['src']] else 'REAL'}
                 if r['role'] == 'fit' else r for r in rows]
    extra_results, extra_rules, extra_scores = fit_readouts(sham_rows,
        [('joint/source_null', FEATURE_NAMES, .1, False)], fit_variants=VARIANTS[:2],
        manifest_sha=parent['inventory_sha256'], config=CONFIG, calibrate=class_threshold, score=score_rule)
    results.update(extra_results)
    rules.update(extra_rules)
    scores.update(extra_scores)
    groups = defaultdict(list)
    for r in scores['joint/source_null']:
        if r['variant'] == 'raw' and r['condition'] != 'original':
            groups[r['domain'] + '/' + r['condition']].append(r)
    intervals = {}
    for key, records in groups.items():
        records.sort(key=lambda r: r['src'])
        intervals[key] = auc_intervals({key: np.array([r['score'] for r in records])},
            np.array([int(r['label'] == 'FAKE') for r in records]), CONFIG)[key]
    for key, rule in rules.items():
        rule.save(OUTPUT / (key.replace('/', '_') + '_rule.json'))
    write_json(OUTPUT / 'selection_scores.json', scores)
    write_json(destination, {'candidates': results, 'null_raw_processed_auc_ci95': intervals,
        'elapsed_s': perf_counter() - start, 'code_pins': code_pins(),
        'inventory_sha256': parent['inventory_sha256'], 'features_sha256': receipt['csv_sha256'],
        'selection_scores_sha256': file_sha256(OUTPUT / 'selection_scores.json'),
        'balanced_assignment_sha256': control['assignment_sha256'],
        'rule_files': {k: file_sha256(OUTPUT / (k.replace('/', '_') + '_rule.json')) for k in rules},
        'goal_achieved': False,
        'null_scope': 'Fit labels changed on a copy after cache validation;threshold/selection truth unchanged',
        'scope': 'Frozen diagonal patch moments,conventional heads;exposed development'})


if __name__ == '__main__':
    main()
