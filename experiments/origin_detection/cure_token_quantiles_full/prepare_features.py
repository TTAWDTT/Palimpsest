"""Two signed numeric subcaches; reuse streaming helpers without source edits."""

import json
from pathlib import Path

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.representations.frozen_cure_quantiles import FrozenCureQuantiles, FEATURE_NAMES
from palimpsest.detection.algorithms.residual_statistics.features import resize256
from palimpsest.evaluation.numeric_extension import extract_extension
from palimpsest.evaluation.numeric_view_increment import extract_numeric_view
from palimpsest.evaluation.numeric_features import numeric_rows
from palimpsest.evaluation.features import validate_feature_cache
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import WORK_DIR, REPO_ROOT, MODELS_ROOT
from experiments.origin_detection.cure_token_quantiles.prepare_features import (
    OUTPUT as PILOT, code_pins as pilot_pins)
from experiments.origin_detection.strong_token_views.prepare_features import (
    parent_data, OUTPUT as STRONG, PARENT)
from experiments.origin_detection.cure_baseline.run_iteration import VENDOR, source_pins, runtime
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS, audit_receipt
from experiments.origin_detection.phase_statistics.run_iteration import image_path
from experiments.origin_detection.query_sensitivity.protocol import write_json

OUTPUT = WORK_DIR/'robust_statistics/cure_token_quantiles_full'


def code_pins():
    pins = pilot_pins()
    paths = list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md',
        REPO_ROOT/'src/palimpsest/evaluation/numeric_view_increment.py',
        REPO_ROOT/'tests/evaluation/test_quantile_extension.py']
    pins.update({str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)})
    return pins


def validate_base(metadata, matrix, inventory):
    rows = numeric_rows(metadata, matrix, FEATURE_NAMES)
    for role in ('fit', 'threshold', 'selection'):
        selected = [r for r in inventory if r['role'] == role]
        validate_feature_cache([r for r in rows if r['role'] == role], selected, FEATURE_NAMES,
            variants=VARIANTS if role == 'selection' else VARIANTS[:2], bounds=(-np.inf, np.inf))


def validate_q60_row(row, vector, old_row, old_vector):
    if (not np.array_equal(vector[:3600], old_vector)
            or row['probability_fake'] != float(old_row['probability_fake'])
            or row['query_sha256'] != old_row['query_sha256']):
        raise ValueError('Q60 prefix/probability/materialized query changed')


def main():
    if OUTPUT.exists():
        raise FileExistsError('Preserve full numeric cache/budget/partials')
    controls = json.loads((PILOT/'full_controls.json').read_text())
    if not controls['passed'] or controls['code_pins'] != code_pins():
        raise ValueError('Full-extension corruption controls changed')
    pilot = json.loads((PILOT/'features.json').read_text())
    if (pilot['code_pins'] != pilot_pins() or not pilot['cost_passed']
            or not pilot['all_parent_prefix_exact'] or pilot['repeated_raw_records'] != 60
            or pilot['runtime'] != runtime()
            or file_sha256(PILOT/'vectors.npy') != pilot['vectors_sha256']
            or file_sha256(PILOT/'metadata.csv') != pilot['metadata_sha256']):
        raise ValueError('Quantile engineering gate changed')
    metadata, values, inventory, parent = parent_data()
    strong = audit_receipt(STRONG/'features.json')
    if (strong['records'] != 5040 or file_sha256(STRONG/'vectors.npy') != strong['vectors_sha256']
            or file_sha256(STRONG/'metadata.csv') != strong['metadata_sha256']):
        raise ValueError('Signed Q60 parent changed')
    qmeta = read_rows(STRONG/'metadata.csv')
    qvalues = np.load(STRONG/'vectors.npy', mmap_mode='r')
    qindex = {(r['filename'], r['variant']): i for i, r in enumerate(qmeta)}
    if len(qindex) != 5040:
        raise ValueError('Duplicate Q60 parent')
    OUTPUT.mkdir(parents=True)
    write_json(OUTPUT/'budget.json', {'conditional_stream_proxy_s': pilot['elapsed_s']/300*20160,
        'pilot_extractor_calls': 300, 'planned_rows': 20160,
        'pilot_receipt_sha256': file_sha256(PILOT/'features.json'), 'code_pins': code_pins(),
        'assumption': 'Scale pilot stream including overhead;source-size/role distribution/thermals differ;init/sourceaudit excluded;not precise ETA'})
    cv2.setNumThreads(1)
    encoder = FrozenCureQuantiles(VENDOR, VENDOR/'weights/cure_adapter.pt',
        MODELS_ROOT/'cure/PE-Core-L14-336.pt', source_pins=source_pins())
    try:
        with threadpool_limits(limits=1):
            base = extract_extension(inventory, metadata, values, OUTPUT/'base', FEATURE_NAMES, encoder,
                resolve_path=image_path, variants=VARIANTS, resize=resize256,
                jpeg_parameters={VARIANTS[1]: (90, 0), VARIANTS[2]: (70, 2), VARIANTS[3]: (60, 2)},
                validate=validate_base)
            write_json(OUTPUT/'base/features.json', {**base, 'code_pins': code_pins(),
                'parent_receipt_sha256': file_sha256(PARENT/'features.json')})
            targets = [r for r in inventory if r['role'] in ('fit', 'threshold')]
            def validate_q60(new_metadata, matrix, native):
                rows = numeric_rows(new_metadata, matrix, FEATURE_NAMES)
                validate_feature_cache(rows, native, FEATURE_NAMES, variants=(VARIANTS[3],), bounds=(-np.inf, np.inf))
                for index, row in enumerate(new_metadata):
                    old = qindex[row['filename'], row['variant']]
                    validate_q60_row(row, matrix[index], qmeta[old], qvalues[old])
            extra = extract_numeric_view(targets, OUTPUT/'q60', FEATURE_NAMES, encoder,
                variant=VARIANTS[3], quality=60, sampling=2, resolve_path=image_path, resize=resize256,
                validate=validate_q60)
            write_json(OUTPUT/'q60/features.json', {**extra, 'code_pins': code_pins(),
                'all_parent_prefix_exact_in_final_validation': True,
                'parent_receipt_sha256': file_sha256(STRONG/'features.json')})
    finally:
        encoder.close()
    if base['records'] != 15120 or extra['records'] != 5040:
        raise ValueError('Quantile full counts changed')
    write_json(OUTPUT/'features.json', {'passed': True, 'records': 20160, 'dimensions': 3920,
        'parts': {'base': base, 'q60': extra}, 'code_pins': code_pins(),
        'inventory_sha256': parent['inventory_sha256'], 'encoder': encoder.provenance,
        'scope': 'Exposed development only;exact parent numeric extension;no classifier/independent result'})


if __name__ == '__main__':
    main()
