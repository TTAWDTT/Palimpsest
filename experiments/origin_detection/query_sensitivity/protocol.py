"""Fixed two-coordinate sensitivity schema and cache identities."""

import json
from pathlib import Path

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR

PARENT = WORK_DIR/'robust_statistics/intermediate_encoder'
OUTPUT = WORK_DIR/'robust_statistics/query_sensitivity'
NAMES = ('query_sensitivity/final768/jpeg90', 'query_sensitivity/midpoint12/jpeg90')
QUERY_VARIANT = 'jpeg90_444_after_resize256'
SEED = 20261006


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def code_pins():
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent/'README.md',
        REPO_ROOT/'src/palimpsest/detection/representations/query_sensitivity.py',
        REPO_ROOT/'src/palimpsest/detection/algorithms/source_view_risk.py',
        REPO_ROOT/'src/palimpsest/detection/algorithms/paired_stability.py',
        REPO_ROOT/'src/palimpsest/evaluation/source_training.py',
        REPO_ROOT/'src/palimpsest/evaluation/source_crossfit.py',
        REPO_ROOT/'src/palimpsest/evaluation/robust_views.py',
        REPO_ROOT/'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        REPO_ROOT/'tests/detection/test_query_sensitivity.py']
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)}


def parent_receipt():
    receipt = json.loads((PARENT/'features.json').read_text(encoding='utf-8'))
    if not receipt['passed'] or not receipt['final_parity']['passed']:
        raise ValueError('Parent did not pass')
    for name, expected in receipt['code_pins'].items():
        if file_sha256(REPO_ROOT/name) != expected:
            raise ValueError(f'Parent source changed: {name}')
    if receipt['csv_sha256'] != file_sha256(PARENT/'features.csv'):
        raise ValueError('Parent cache changed')
    return receipt
