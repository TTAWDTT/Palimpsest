"""Extract only missing stronger CuRe support, gating on unchanged old views."""

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from palimpsest.detection.algorithms.residual_statistics.features import resize256
from palimpsest.detection.representations.frozen_cure import FEATURE_NAMES, FrozenCure
from palimpsest.evaluation.features import validate_feature_cache
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR, MODELS_ROOT
from experiments.origin_detection.cure_baseline.run_iteration import VENDOR, source_pins, runtime
from experiments.origin_detection.cure_readout.prepare_features import OUTPUT as PARENT, code_pins as parent_pins
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json
from experiments.origin_detection.source_view_risk.run_iteration import parent_data, VARIANTS

OUTPUT = WORK_DIR / 'robust_statistics/cure_encoding_risk'
VARIANT = VARIANTS[2]


def code_pins():
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent / 'README.md']
    paths += [REPO_ROOT / p for p in (
        'src/palimpsest/detection/representations/frozen_cure.py',
        'src/palimpsest/detection/baselines/cure.py',
        'src/palimpsest/detection/algorithms/residual_statistics/features.py',
        'src/palimpsest/evaluation/features.py', 'src/palimpsest/evaluation/source_training.py',
        'src/palimpsest/evaluation/balanced_null.py', 'src/palimpsest/evaluation/robust_views.py',
        'src/palimpsest/detection/algorithms/readouts/source_view_risk.py',
        'src/palimpsest/detection/algorithms/paired_stability.py',
        'experiments/origin_detection/cure_readout/prepare_features.py',
        'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        'tests/detection/test_frozen_cure.py')]
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)}


def exact_reference(row, expected):
    reference = expected[row['filename']]
    if (float(row['probability_fake']) != float(reference['probability_fake'])
            or not np.array_equal([float(row[n]) for n in FEATURE_NAMES], [float(reference[n]) for n in FEATURE_NAMES])):
        raise ValueError('Repeated CuRe vector or author probability differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pilot', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    output = OUTPUT.with_name(OUTPUT.name+'_pilot') if args.pilot else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'features.csv').exists() or (output / 'features.json').exists():
        raise FileExistsError('Preserve stronger CuRe support extraction')
    controls = json.loads((OUTPUT / 'software_controls.json').read_text())
    if not controls['passed'] or any(file_sha256(REPO_ROOT / k) != v for k, v in controls['test_pins'].items()):
        raise ValueError('Raw vector software gate changed')
    old, inventory, parent, _ = parent_data()
    del old
    receipt = json.loads((PARENT / 'features.json').read_text())
    if (receipt['code_pins'] != parent_pins() or receipt['csv_sha256'] != file_sha256(PARENT / 'features.csv')
            or receipt['inventory_sha256'] != parent['inventory_sha256'] or not receipt['original_probability_exact']):
        raise ValueError('Signed weak CuRe parent changed')
    packages = runtime()
    expected = None
    if args.pilot:
        signed = PARENT.with_name(PARENT.name+'_pilot')
        pilot = json.loads((signed / 'features.json').read_text())
        if pilot['csv_sha256'] != file_sha256(signed / 'features.csv'):
            raise ValueError('Original60-file pilot changed')
        expected = {r['filename']: r for r in read_rows(signed / 'features.csv') if r['variant'] == VARIANT}
        inventory = [r for r in inventory if r['filename'] in expected]
        if len(expected) != 60 or len(inventory) != 60 or any(r['role'] != 'selection' for r in inventory):
            raise ValueError('Expected60 pre-exposed selection witnesses')
    else:
        signed = OUTPUT.with_name(OUTPUT.name+'_pilot')
        pilot = json.loads((signed / 'features.json').read_text())
        if (pilot['code_pins'] != code_pins() or pilot['runtime'] != packages or not pilot['reference_exact']
                or not pilot['changed_coordinate_rejected'] or pilot['records'] != 60
                or pilot['csv_sha256'] != file_sha256(signed / 'features.csv')):
            raise ValueError('Stronger support pilot differs or failed')
        inventory = [r for r in inventory if r['role'] in ('fit', 'threshold')]
        if len(inventory) != 5040:
            raise ValueError('Additional support denominator changed')
    cv2.setNumThreads(1)
    encoder = FrozenCure(VENDOR, VENDOR / 'weights/cure_adapter.pt', MODELS_ROOT / 'cure/PE-Core-L14-336.pt',
                         source_pins=source_pins())
    scratch = output / 'query.jpg'
    if scratch.exists():
        raise FileExistsError('Unfinished query file requires audit')
    start = perf_counter()
    rows = []
    for i, row in enumerate(inventory, 1):
        path = image_path(row)
        if file_sha256(path) != row['sha256']:
            raise ValueError('Native additional support digest changed')
        with Image.open(path) as image:
            if image.size != (int(row['width']), int(row['height'])):
                raise ValueError('Additional support dimensions changed')
            pixels = np.asarray(image.convert('RGB'))
        Image.fromarray(resize256(pixels)).save(scratch, format='JPEG', quality=70, subsampling=2)
        feature = encoder.extract_file(scratch)
        actual = {**row, 'variant': VARIANT, **dict(zip(FEATURE_NAMES, feature.values.tolist())),
                  'probability_fake': feature.probability_fake}
        if args.pilot:
            exact_reference(actual, expected)
        rows.append(actual)
        if i % 120 == 0 or i == len(inventory):
            print(json.dumps({'images': i, 'total': len(inventory), 'elapsed_s': round(perf_counter()-start, 3)}), flush=True)
    scratch.unlink()
    validate_feature_cache(rows, inventory, FEATURE_NAMES, variants=(VARIANT,), bounds=(-np.inf, np.inf))
    changed_rejected = None
    if args.pilot:
        wrong = {**rows[0], FEATURE_NAMES[0]: float(rows[0][FEATURE_NAMES[0]])+.001}
        try:
            exact_reference(wrong, expected)
        except ValueError:
            changed_rejected = True
        else:
            raise ValueError('Changed coordinate passed equality gate')
    with (output / 'features.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'features.json', {'records': len(rows), 'images': len(inventory),
        'elapsed_s': perf_counter()-start, 'csv_sha256': file_sha256(output / 'features.csv'),
        'code_pins': code_pins(), 'runtime': packages, 'encoder': encoder.provenance,
        'inventory_sha256': parent['inventory_sha256'], 'parent_receipt_sha256': file_sha256(PARENT / 'features.json'),
        'reference_exact': True if args.pilot else None, 'changed_coordinate_rejected': changed_rejected,
        'scope': 'Pilot repeats old60Q70;full only fit/threshold missingQ70;no new selection inference'})


if __name__ == '__main__':
    main()
