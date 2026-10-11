"""Extract only missing fit/threshold Q60, first repeating signed exposed views."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.representations.frozen_cure_covariance import FrozenCureCovariance, FEATURE_NAMES
from palimpsest.evaluation.numeric_view_increment import extract_numeric_view
from palimpsest.evaluation.numeric_features import numeric_rows
from palimpsest.evaluation.features import validate_feature_cache
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import WORK_DIR, REPO_ROOT, MODELS_ROOT
from palimpsest.detection.algorithms.residual_statistics.features import resize256
from experiments.origin_detection.cure_token_covariance.prepare_features import (
    OUTPUT as PARENT, code_pins as parent_pins, audit_cache as parent_audit)
from experiments.origin_detection.cure_baseline.run_iteration import VENDOR, source_pins, runtime
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS, audit_receipt
from experiments.origin_detection.phase_statistics.run_iteration import image_path
from palimpsest.evaluation.calibrated_readout_campaign import write_json

OUTPUT = WORK_DIR/'robust_statistics/strong_token_views'
VARIANT = VARIANTS[3]
TESTS = ('tests/evaluation/test_numeric_view_increment.py', 'tests/evaluation/test_numeric_extension.py',
         'tests/detection/test_token_covariance.py')


def code_pins():
    pins = parent_pins()
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent/'README.md']
    paths += [REPO_ROOT/p for p in (*TESTS, 'src/palimpsest/evaluation/numeric_view_increment.py',
                                 'src/palimpsest/evaluation/calibrated_readout_campaign.py')]
    pins.update({str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)})
    return pins


def parent_data():
    receipt = audit_receipt(PARENT/'features.json')
    if (receipt['records'] != 15120 or not receipt['all_parent_prefix_exact']
            or file_sha256(PARENT/'vectors.npy') != receipt['vectors_sha256']
            or file_sha256(PARENT/'metadata.csv') != receipt['metadata_sha256']):
        raise ValueError('Signed numeric parent changed')
    metadata = read_rows(PARENT/'metadata.csv'); values = np.load(PARENT/'vectors.npy', mmap_mode='r')
    inventory = [r for r in metadata if r['variant'] == 'raw']
    parent_audit(metadata, values, inventory)
    return metadata, values, inventory, receipt


def validate(metadata, values, inventory):
    rows = numeric_rows(metadata, values, FEATURE_NAMES)
    validate_feature_cache(rows, inventory, FEATURE_NAMES, variants=(VARIANT,), bounds=(-np.inf, np.inf))
    norms = np.linalg.norm(values[:, 3072:], axis=1)
    if np.any((norms != 0) & (np.abs(norms-1) > 1e-10)):
        raise ValueError('Incremental covariance root norm differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__); mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--record-controls', action='store_true'); mode.add_argument('--pilot', action='store_true')
    mode.add_argument('--prepare', action='store_true'); args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.record_controls:
        if (OUTPUT/'software_controls.json').exists(): raise FileExistsError('Preserve view controls')
        result = subprocess.run([sys.executable, '-X', 'utf8', '-m', 'pytest', '-q', *TESTS],
            cwd=REPO_ROOT, capture_output=True, text=True, encoding='utf-8')
        print(result.stdout, flush=True)
        if result.returncode or ' passed' not in result.stdout or 'skipped' in result.stdout:
            raise ValueError('View software controls failed: '+result.stderr)
        write_json(OUTPUT/'software_controls.json', {'passed': True, 'stdout': result.stdout, 'code_pins': code_pins(),
            'scope': 'Known synthetic view storage and covariance components;not full scientific pipeline'})
        return
    output = OUTPUT.with_name(OUTPUT.name+'_pilot') if args.pilot else OUTPUT
    if (output/'features.json').exists(): raise FileExistsError('Preserve incremental feature receipt')
    controls = json.loads((OUTPUT/'software_controls.json').read_text())
    if not controls['passed'] or controls['code_pins'] != code_pins(): raise ValueError('Incremental controls changed')
    metadata, values, inventory, parent = parent_data(); packages = runtime(); reference = None
    if packages != parent['runtime']: raise ValueError('Incremental runtime changed')
    if args.pilot:
        old = PARENT.with_name(PARENT.name+'_pilot'); p = audit_receipt(old/'features.json')
        if file_sha256(old/'metadata.csv') != p['metadata_sha256']: raise ValueError('Old exposed pilot changed')
        filenames = {r['filename'] for r in read_rows(old/'metadata.csv')}
        inventory = [r for r in inventory if r['filename'] in filenames]
        if len(inventory) != 60 or any(r['role'] != 'selection' for r in inventory):
            raise ValueError('Expected old exposed selection witnesses')
        reference = {(r['filename'], VARIANT): (values[i], float(r['probability_fake']))
                     for i, r in enumerate(metadata) if r['filename'] in filenames and r['variant'] == VARIANT}
    else:
        p = json.loads((OUTPUT.with_name(OUTPUT.name+'_pilot')/'features.json').read_text())
        if (not p['cost_passed'] or not p['complete_reference_exact'] or not p['changed_coordinate_rejected']
                or p['code_pins'] != code_pins() or p['runtime'] != packages
                or p['parent_receipt_sha256'] != file_sha256(PARENT/'features.json')
                or p['projected5040_stream_s'] > 7200):
            raise ValueError('Incremental pilot failed/changed/budget too large')
        pilot_dir = OUTPUT.with_name(OUTPUT.name+'_pilot')
        if (file_sha256(pilot_dir/'vectors.npy') != p['vectors_sha256']
                or file_sha256(pilot_dir/'metadata.csv') != p['metadata_sha256']):
            raise ValueError('Incremental pilot bytes changed')
        inventory = [r for r in inventory if r['role'] in ('fit', 'threshold')]
        if len(inventory) != 5040: raise ValueError('Missing strong view count differs')
    cv2.setNumThreads(1)
    encoder = FrozenCureCovariance(VENDOR, VENDOR/'weights/cure_adapter.pt', MODELS_ROOT/'cure/PE-Core-L14-336.pt',
        source_pins=source_pins())
    try:
        if encoder.provenance != parent['encoder']: raise ValueError('Incremental encoder/projection changed')
        with threadpool_limits(limits=1):
            result = extract_numeric_view(inventory, output, FEATURE_NAMES, encoder, variant=VARIANT, quality=60,
                sampling=2, resolve_path=image_path, resize=resize256, validate=validate, reference=reference)
    finally:
        encoder.close()
    changed = None
    if args.pilot:
        from palimpsest.evaluation.numeric_extension import check_prefix
        from types import SimpleNamespace
        first = next(iter(reference.values())); bad = np.array(first[0], copy=True); bad[0] += .001
        try: check_prefix(SimpleNamespace(values=bad, probability_fake=first[1]), first[0], first[1])
        except ValueError: changed = True
        else: raise ValueError('Changed coordinate accepted')
    passed = result['cost_median_ms'] <= 175 if args.pilot else None
    write_json(output/'features.json', {**result, 'cost_passed': passed, 'changed_coordinate_rejected': changed,
        'projected5040_stream_s': 5040*result['elapsed_s']/60 if args.pilot else None,
        'projection_assumption': 'Pilot same Q60 materialization;5040/60 linear count;loading/init/receipt hashes excluded',
        'runtime': packages, 'encoder': encoder.provenance, 'feature_names': FEATURE_NAMES, 'code_pins': code_pins(),
        'inventory_sha256': parent['inventory_sha256'], 'parent_receipt_sha256': file_sha256(PARENT/'features.json'),
        'scope': 'Only development Q60 acquisition;pilot repeats old vectors;new fit/threshold lack per-view old reference'})
    if args.pilot and not passed: raise ValueError('Incremental pilot engineering budget failed')


if __name__ == '__main__': main()
