"""One-off Q60 acquisition budget: prewarm, repeat 60 signed numeric witnesses.

Problem-specific composition of the existing extraction helper, not a second
extractor. Discards model/CUDA startup before the measured streaming window.
"""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from palimpsest.detection.representations.frozen_cure_covariance import FrozenCureCovariance, FEATURE_NAMES
from palimpsest.evaluation.numeric_extension import check_prefix
from palimpsest.evaluation.numeric_view_increment import extract_numeric_view
from palimpsest.evaluation.calibrated_readout_campaign import write_json
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import MODELS_ROOT, WORK_DIR
from experiments.origin_detection.strong_token_views.prepare_features import (
    OUTPUT as SOURCE, PARENT, VARIANT, VENDOR, source_pins, runtime, parent_data, image_path, resize256, validate, code_pins)


def main():
    output = WORK_DIR/'robust_statistics/strong_token_views_warm_timing'
    if output.exists(): raise FileExistsError('Preserve warm timing')
    pilot_dir = SOURCE.with_name(SOURCE.name+'_pilot'); pilot = json.loads((pilot_dir/'features.json').read_text())
    if (not pilot['cost_passed'] or not pilot['complete_reference_exact'] or pilot['code_pins'] != code_pins()
            or file_sha256(pilot_dir/'vectors.npy') != pilot['vectors_sha256']
            or file_sha256(pilot_dir/'metadata.csv') != pilot['metadata_sha256']):
        raise ValueError('Warm timing requires unchanged pilot')
    metadata, values, inventory, parent = parent_data(); packages = runtime()
    if packages != parent['runtime']: raise ValueError('Warm runtime changed')
    filenames = {r['filename'] for r in read_rows(pilot_dir/'metadata.csv')}
    inventory = [r for r in inventory if r['filename'] in filenames]
    if len(inventory) != 60 or any(r['role'] != 'selection' for r in inventory): raise ValueError('Warm witness count differs')
    reference = {(r['filename'], VARIANT): (values[i], float(r['probability_fake']))
                 for i, r in enumerate(metadata) if r['filename'] in filenames and r['variant'] == VARIANT}
    output.mkdir(); cv2.setNumThreads(1)
    encoder = FrozenCureCovariance(VENDOR, VENDOR/'weights/cure_adapter.pt', MODELS_ROOT/'cure/PE-Core-L14-336.pt',
        source_pins=source_pins()); warm = output/'warm.jpg'
    try:
        if encoder.provenance != parent['encoder']: raise ValueError('Warm encoder changed')
        with threadpool_limits(limits=1):
            with Image.open(image_path(inventory[0])) as image: pixels = np.asarray(image.convert('RGB'))
            Image.fromarray(resize256(pixels)).save(warm, format='JPEG', quality=60, subsampling=2)
            feature = encoder.extract_file(warm)
            expected, probability = reference[inventory[0]['filename'], VARIANT]
            check_prefix(feature, expected, probability); warm.unlink()
            result = extract_numeric_view(inventory, output, FEATURE_NAMES, encoder, variant=VARIANT, quality=60,
                sampling=2, resolve_path=image_path, resize=resize256, validate=validate, reference=reference)
    finally:
        encoder.close()
    rows = read_rows(output/'metadata.csv'); times = np.array([float(r['elapsed_ms']) for r in rows])
    first, last = float(np.median(times[:30])), float(np.median(times[30:]))
    projected = 5040*result['elapsed_s']/60
    passed = (max(first, last)/min(first, last) < 2 and result['cost_median_ms'] <= 175 and projected <= 7200)
    write_json(output/'timing.json', {**result, 'passed': passed, 'prewarm_records': 1,
        'first30_feature_median_ms': first, 'last30_feature_median_ms': last,
        'projected5040_stream_s': projected, 'code_pins': code_pins(), 'script_sha256': file_sha256(Path(__file__)),
        'pilot_receipt_sha256': file_sha256(pilot_dir/'features.json'), 'parent_receipt_sha256': file_sha256(PARENT/'features.json'),
        'assumption': 'Hot60 streaming window,linear fixed-shape per-image cost;native size mix and thermal/load can differ',
        'scope': 'Measured stream includes native decode/materialization/validation and array/CSV write;input/model init/receipt hashes excluded'})
    print(json.dumps({'passed': passed, 'warm_stream_s': result['elapsed_s'], 'projected5040_s': projected}), flush=True)
    if not passed: raise ValueError('Warm timing failed;restructure before larger run')


if __name__ == '__main__': main()
