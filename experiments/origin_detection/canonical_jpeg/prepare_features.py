"""Signed fixed normalization cache with raw-to-existing-Q70 parity gates."""

import argparse
import json
from pathlib import Path
import random

import cv2
import numpy as np

from palimpsest.detection.representations.canonical_clip import CanonicalClip, canonical_pixels
from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES
from palimpsest.detection.algorithms.residual_statistics.features import resize256
from palimpsest.evaluation.feature_prefix import validate_cached_prefix
from palimpsest.evaluation.signed_feature_inputs import load_signed_columns
from palimpsest.evaluation.pixel_features import extract_inventory
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR, MODELS_ROOT
from experiments.origin_detection.source_view_risk.run_iteration import parent_data, VARIANTS, Q60
from experiments.origin_detection.phase_statistics.run_iteration import write_json, image_path

OUTPUT = WORK_DIR / 'robust_statistics/canonical_jpeg'


def code_pins():
    paths = [REPO_ROOT / 'src/palimpsest/detection/representations' / n for n in
             ('canonical_clip.py', 'frozen_clip.py')]
    paths += [REPO_ROOT / 'src/palimpsest/evaluation' / n for n in
              ('feature_prefix.py', 'balanced_null.py', 'signed_feature_inputs.py',
               'pixel_features.py', 'source_readout_campaign.py', 'source_training.py', 'robust_views.py')]
    paths += [REPO_ROOT / 'src/palimpsest/detection/algorithms/source_view_risk.py',
              REPO_ROOT / 'src/palimpsest/detection/algorithms/residual_statistics/features.py',
              REPO_ROOT / 'tests/detection/test_canonical_clip.py',
              REPO_ROOT / 'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
              REPO_ROOT / 'experiments/origin_detection/threshold_calibration/fit_threshold.py']
    paths += [Path(__file__).parent / n for n in ('README.md', 'prepare_features.py', 'run_iteration.py')]
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--pilot', type=int)
    group.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.pilot is not None and args.pilot <= 0:
        parser.error('Pilot must be positive')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    controls = json.loads((OUTPUT / 'software_controls.json').read_text())
    if controls['returncode'] or any(file_sha256(REPO_ROOT / k) != v for k, v in controls['test_pins'].items()):
        raise ValueError('Canonical software controls changed')
    old, inventory, parent, _ = parent_data()
    del old
    reference, parent_sha = load_signed_columns(WORK_DIR / 'robust_statistics/intermediate_encoder',
        FEATURE_NAMES, inventory, repository_root=REPO_ROOT, inventory_sha=parent['inventory_sha256'],
        ordinary_variants=VARIANTS[:3], selection_variant=Q60)
    reference = [r for r in reference if r['variant'] == VARIANTS[2]]
    directory = OUTPUT.with_name('canonical_jpeg_pilot') if args.pilot else OUTPUT
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'features.json').exists():
        raise FileExistsError('Preserve canonical extraction')
    if args.prepare:
        pilot_path = OUTPUT.with_name('canonical_jpeg_pilot') / 'features.json'
        pilot = json.loads(pilot_path.read_text())
        if (pilot['code_pins'] != code_pins() or not pilot['raw_to_q70']['exact_equal']
                or not pilot['wrong_value_rejected']
                or pilot['csv_sha256'] != file_sha256(pilot_path.with_name('features.csv'))):
            raise ValueError('Canonical pilot changed')
    else:
        random.Random(20261006).shuffle(inventory)
        inventory = inventory[:args.pilot]
    encoder = CanonicalClip(MODELS_ROOT / 'd3/ViT-L-14.pt')
    cv2.setNumThreads(1)
    synthetic = []
    for h, w in ((256, 256), (301, 417), (417, 301)):
        y, x = np.indices((h, w))
        pixels = np.stack([(x + y) % 256, (3*x + y) % 256, (x + 7*y) % 256], axis=2).astype(np.uint8)
        actual = encoder.extract(pixels).values
        expected = encoder.base.extract(canonical_pixels(pixels)).values
        if not np.array_equal(actual, expected) or not np.array_equal(actual, encoder.extract(pixels).values):
            raise ValueError('Canonical encoder runtime parity differs')
        synthetic.append({'height': h, 'width': w, 'exact_equal': True})
    result = extract_inventory(inventory, directory / 'features.csv', names=FEATURE_NAMES,
        extractor=encoder.extract, resolve_path=image_path, resize=resize256,
        ordinary_variants=VARIANTS[:3], selection_variant=Q60,
        jpeg_parameters={VARIANTS[1]: (90, 0), VARIANTS[2]: (70, 2), Q60: (60, 2)})
    fresh = read_rows(directory / 'features.csv')
    mapped = [{**r, 'variant': VARIANTS[2]} for r in fresh if r['variant'] == 'raw']
    parity = validate_cached_prefix(mapped, reference, FEATURE_NAMES, full_coverage=not args.pilot)
    bad = [{**mapped[0], FEATURE_NAMES[0]: float(mapped[0][FEATURE_NAMES[0]]) + .001}]
    rejected = False
    try:
        validate_cached_prefix(bad, reference, FEATURE_NAMES, full_coverage=False)
    except ValueError as error:
        if 'values differ' not in str(error):
            raise
        rejected = True
    if not rejected:
        raise ValueError('Canonical wrong feature did not fail')
    write_json(directory / 'features.json', {**result, 'raw_to_q70': parity, 'wrong_value_rejected': True,
        'synthetic_runtime': synthetic, 'encoder': encoder.provenance, 'code_pins': code_pins(),
        'inventory_sha256': parent['inventory_sha256'], 'parent_features_receipt_sha256': parent_sha,
        'software_controls_sha256': file_sha256(OUTPUT / 'software_controls.json')})


if __name__ == '__main__':
    main()
