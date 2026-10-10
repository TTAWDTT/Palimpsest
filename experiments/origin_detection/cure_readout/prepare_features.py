"""Extract fixed released CuRe vectors, requiring exact old scalar parity."""

import argparse
from collections import defaultdict
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
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json
from experiments.origin_detection.source_view_risk.run_iteration import parent_data, VARIANTS

OUTPUT = WORK_DIR / 'robust_statistics/cure_readout'
BASELINE = WORK_DIR / 'robust_statistics/cure_baseline'


def code_pins():
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent / 'README.md',
        REPO_ROOT / 'src/palimpsest/detection/representations/frozen_cure.py',
        REPO_ROOT / 'tests/detection/test_frozen_cure.py',
        REPO_ROOT / 'src/palimpsest/detection/baselines/cure.py',
        REPO_ROOT / 'src/palimpsest/evaluation/features.py',
        REPO_ROOT / 'src/palimpsest/evaluation/source_readout_campaign.py',
        REPO_ROOT / 'src/palimpsest/evaluation/source_training.py',
        REPO_ROOT / 'src/palimpsest/evaluation/balanced_null.py',
        REPO_ROOT / 'src/palimpsest/evaluation/robust_views.py',
        REPO_ROOT / 'src/palimpsest/detection/algorithms/readouts/source_view_risk.py',
        REPO_ROOT / 'src/palimpsest/detection/algorithms/paired_stability.py',
        REPO_ROOT / 'experiments/origin_detection/cure_baseline/run_iteration.py',
        REPO_ROOT / 'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        REPO_ROOT / 'experiments/origin_detection/threshold_calibration/fit_threshold.py']
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)}


def baseline_probabilities():
    receipt = json.loads((BASELINE / 'iteration.json').read_text())
    if receipt['scores_sha256'] != file_sha256(BASELINE / 'scores.csv'):
        raise ValueError('Signed CuRe baseline changed')
    rows = read_rows(BASELINE / 'scores.csv')
    expected = {(r['filename'], r['variant']): float(r['probability_fake']) for r in rows}
    if len(rows) != 5040 or len(expected) != len(rows):
        raise ValueError('Original CuRe score coverage changed')
    return expected, receipt


def audit_cache(rows, inventory):
    for role in ('fit', 'threshold', 'selection'):
        selected = [r for r in inventory if r['role'] == role]
        actual = [r for r in rows if r['role'] == role]
        if selected:
            validate_feature_cache(actual, selected, FEATURE_NAMES,
                variants=VARIANTS if role == 'selection' else VARIANTS[:2], bounds=(-np.inf, np.inf))
        elif actual:
            raise ValueError('Unexpected CuRe feature role')


def check_probability(row, expected):
    if row['role'] == 'selection' and float(row['probability_fake']) != expected[row['filename'], row['variant']]:
        raise ValueError('Original CuRe probability changed')


def synthetic_gate(encoder, output):
    witness = output / 'synthetic.png'
    if witness.exists():
        raise FileExistsError('Preserve CuRe feature witness')
    pixels = np.random.default_rng(20261006).integers(0, 256, (301, 417, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(witness)
    first = encoder.extract_file(witness)
    second = encoder.extract_file(witness)
    direct = encoder.base.predict_file(witness)
    if not np.array_equal(first.values, second.values) or first.probability_fake != direct.probability_fake:
        raise ValueError('CuRe feature wrapper/direct/repeat mismatch')
    return {'passed': True, 'feature_repeat_exact': True, 'author_probability_exact': True,
            'shape': list(first.values.shape), 'synthetic_sha256': file_sha256(witness)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pilot', action='store_true')
    mode.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    output = OUTPUT.with_name(OUTPUT.name + '_pilot') if args.pilot else OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'features.csv').exists() or (output / 'features.json').exists():
        raise FileExistsError('Preserve CuRe feature cache')
    control_path = OUTPUT / 'software_controls.json'
    controls = json.loads(control_path.read_text())
    if not controls['passed'] or any(file_sha256(REPO_ROOT / k) != v for k, v in controls['test_pins'].items()):
        raise ValueError('CuRe raw storage gate changed')
    expected, baseline = baseline_probabilities()
    packages = runtime()
    if not args.pilot:
        pilot_dir = OUTPUT.with_name(OUTPUT.name + '_pilot')
        pilot = json.loads((pilot_dir / 'features.json').read_text())
        if (pilot['code_pins'] != code_pins() or pilot['runtime'] != packages or not pilot['original_probability_exact']
                or not pilot['wrong_probability_rejected'] or pilot['csv_sha256'] != file_sha256(pilot_dir / 'features.csv')):
            raise ValueError('CuRe feature pilot changed/failed')
    old, inventory, parent, _ = parent_data()
    del old
    if parent['inventory_sha256'] != baseline['inventory_sha256']:
        raise ValueError('CuRe baseline/feature inventory differs')
    if args.pilot:
        selected = [r for r in inventory if r['role'] == 'selection']
        groups = defaultdict(list)
        for row in selected:
            groups[row['domain'], row['scene'], row['label'], row['condition']].append(row)
        kept = {r['filename']: r for g in groups.values() for r in g[:1]}
        for row in sorted(selected, key=lambda r: int(r['width']) * int(r['height']), reverse=True):
            if len(kept) >= 60:
                break
            kept[row['filename']] = row
        inventory = sorted(kept.values(), key=lambda r: r['filename'])
    cv2.setNumThreads(1)
    encoder = FrozenCure(VENDOR, VENDOR / 'weights/cure_adapter.pt', MODELS_ROOT / 'cure/PE-Core-L14-336.pt',
                         source_pins=source_pins())
    gate = synthetic_gate(encoder, output)
    rows, start = [], perf_counter()
    scratch = output / 'query.jpg'
    if scratch.exists():
        raise FileExistsError('Unfinished CuRe feature query needs audit')
    parameters = dict(zip(VARIANTS[1:], ((90, 0), (70, 2), (60, 2))))
    for index, row in enumerate(inventory, 1):
        native = image_path(row)
        if file_sha256(native) != row['sha256']:
            raise ValueError('CuRe feature pixel source changed')
        with Image.open(native) as image:
            if image.size != (int(row['width']), int(row['height'])):
                raise ValueError('CuRe feature pixel size changed')
            pixels = np.asarray(image.convert('RGB'))
        for variant in VARIANTS if row['role'] == 'selection' else VARIANTS[:2]:
            query = native
            if variant != 'raw':
                q, s = parameters[variant]
                Image.fromarray(resize256(pixels)).save(scratch, format='JPEG', quality=q, subsampling=s)
                query = scratch
            features = encoder.extract_file(query)
            actual = {**row, 'variant': variant, **dict(zip(FEATURE_NAMES, features.values.tolist())),
                'probability_fake': features.probability_fake, 'decode_preprocess_ms': features.decode_preprocess_ms,
                'encoder_ms': features.encoder_ms}
            check_probability(actual, expected)
            rows.append(actual)
        if index % 60 == 0 or index == len(inventory):
            print(json.dumps({'images': index, 'total': len(inventory), 'records': len(rows),
                              'elapsed_s': round(perf_counter()-start, 3)}), flush=True)
    scratch.unlink()
    audit_cache(rows, inventory)
    selection = [r for r in rows if r['role'] == 'selection']
    wrong = {**selection[0], 'probability_fake': float(selection[0]['probability_fake']) + .001}
    try:
        check_probability(wrong, expected)
    except ValueError:
        pass
    else:
        raise ValueError('Changed CuRe probability accepted')
    with (output / 'features.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'features.json', {'images': len(inventory), 'records': len(rows),
        'elapsed_s': perf_counter()-start, 'csv_sha256': file_sha256(output / 'features.csv'),
        'inventory_sha256': parent['inventory_sha256'], 'original_probability_exact': True,
        'matched_original_probabilities': len(selection), 'wrong_probability_rejected': True,
        'code_pins': code_pins(), 'runtime': packages, 'encoder': encoder.provenance,
        'synthetic_gate': gate, 'software_controls_sha256': file_sha256(control_path),
        'baseline_receipt_sha256': file_sha256(BASELINE / 'iteration.json'),
        'scope': 'Frozen released CuRe raw vectors;no representation fit;exposed development'})


if __name__ == '__main__':
    main()
