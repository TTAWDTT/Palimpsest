"""Engineering witness only; reuse signed numeric streaming and parent cache."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.representations.frozen_cure_quantiles import FrozenCureQuantiles, FEATURE_NAMES
from palimpsest.detection.algorithms.residual_statistics.features import resize256
from palimpsest.evaluation.numeric_extension import extract_extension
from palimpsest.evaluation.numeric_features import numeric_rows
from palimpsest.evaluation.features import validate_feature_cache
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import WORK_DIR, REPO_ROOT, MODELS_ROOT
from experiments.origin_detection.strong_token_views.prepare_features import parent_data, PARENT
from experiments.origin_detection.cure_token_covariance.prepare_features import code_pins as parent_pins
from experiments.origin_detection.cure_token_statistics.prepare_features import synthetic_gate
from experiments.origin_detection.cure_baseline.run_iteration import VENDOR, source_pins, runtime
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS
from experiments.origin_detection.phase_statistics.run_iteration import image_path
from experiments.origin_detection.query_sensitivity.protocol import write_json

OUTPUT = WORK_DIR/'robust_statistics/cure_token_quantiles_pilot'
TESTS = ('tests/detection/test_token_quantiles.py', 'tests/detection/test_token_covariance.py',
         'tests/evaluation/test_numeric_extension.py')


def code_pins():
    pins = parent_pins()
    paths = list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md']
    paths += [REPO_ROOT/p for p in (*TESTS,
        'src/palimpsest/detection/representations/token_quantiles.py',
        'src/palimpsest/detection/representations/frozen_cure_quantiles.py',
        'experiments/origin_detection/strong_token_views/prepare_features.py')]
    pins.update({str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)})
    return pins


def validate(metadata, matrix, inventory):
    rows = numeric_rows(metadata, matrix, FEATURE_NAMES)
    validate_feature_cache(rows, inventory, FEATURE_NAMES, variants=VARIANTS, bounds=(-np.inf, np.inf))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--record-controls', action='store_true')
    group.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.record_controls:
        if (OUTPUT/'software_controls.json').exists():
            raise FileExistsError('Preserve quantile controls')
        result = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'pytest', '-q', *TESTS],
            text=True, capture_output=True)
        if result.returncode:
            raise ValueError(result.stdout+result.stderr)
        write_json(OUTPUT/'software_controls.json', {'passed': True, 'code_pins': code_pins(),
            'stdout': result.stdout, 'scope': 'Component/known answer controls;not classification evidence'})
        return
    if (OUTPUT/'features.json').exists():
        raise FileExistsError('Preserve quantile pilot')
    controls = json.loads((OUTPUT/'software_controls.json').read_text())
    if not controls['passed'] or controls['code_pins'] != code_pins():
        raise ValueError('Quantile controls changed')
    metadata, values, inventory, parent = parent_data()
    old = PARENT.with_name(PARENT.name+'_pilot')
    receipt = json.loads((old/'features.json').read_text())
    if (file_sha256(old/'metadata.csv') != receipt['metadata_sha256']
            or file_sha256(old/'vectors.npy') != receipt['vectors_sha256']):
        raise ValueError('Old engineering witness changed')
    names = {r['filename'] for r in read_rows(old/'metadata.csv')}
    inventory = [r for r in inventory if r['filename'] in names]
    if len(inventory) != 60 or any(r['role'] != 'selection' for r in inventory):
        raise ValueError('Expected previous60exposed engineering queries')
    packages = runtime()
    cv2.setNumThreads(1)
    encoder = FrozenCureQuantiles(VENDOR, VENDOR/'weights/cure_adapter.pt',
        MODELS_ROOT/'cure/PE-Core-L14-336.pt', source_pins=source_pins())
    try:
        with threadpool_limits(limits=1):
            witnesses = synthetic_gate(encoder, OUTPUT)
            result = extract_extension(inventory, metadata, values, OUTPUT, FEATURE_NAMES, encoder,
                resolve_path=image_path, variants=VARIANTS, resize=resize256,
                jpeg_parameters={VARIANTS[1]: (90, 0), VARIANTS[2]: (70, 2), VARIANTS[3]: (60, 2)},
                repeat_raw=True, validate=validate)
    finally:
        encoder.close()
    write_json(OUTPUT/'features.json', {**result, 'code_pins': code_pins(), 'runtime': packages,
        'encoder': encoder.provenance, 'synthetic_witnesses': witnesses,
        'parent_receipt_sha256': file_sha256(PARENT/'features.json'),
        'inventory_sha256': parent['inventory_sha256'], 'cost_passed': result['cost_median_ms'] <= 200,
        'scope': '240 old exposed query variants;parent3600/probability exact;no new classifier result'})
    if result['cost_median_ms'] > 200:
        raise ValueError('Quantile pilot engineering budget refused')


if __name__ == '__main__':
    main()
