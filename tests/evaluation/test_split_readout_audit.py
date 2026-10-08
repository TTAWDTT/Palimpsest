"""Planted stage-member corruption must fail the independent saved-role check."""

from copy import deepcopy

import pytest

from palimpsest.detection.algorithms.source_partition import source_half_split
from tools.audit_typed_source_panel_cv import verify_stage_partition
from tests.detection.test_rate_penalty import fixture


def test_known_stage_members_and_bad_record():
    records, y, _, _ = fixture()
    mask = source_half_split(records, y)
    basis = sorted({(r['domain'], r['src']) for r, m in zip(records, mask) if m})
    head = sorted({(r['domain'], r['src']) for r, m in zip(records, mask) if not m})
    rows = [{**r, 'label': 'FAKE' if label else 'REAL'} for r, label in zip(records, y)]
    folds = {'/'.join(k): 2 for k in basis+head}
    for name, fold in (('held', 0), ('cal', 1)):
        row = {**rows[0], 'src': name}; rows.append(row); folds['d/'+name] = fold
    d = {'basis_source_keys': basis, 'readout_source_keys': head, 'partition_seed': 20261008,
        'input_source_count': 4, 'basis_source_count': 2, 'basis_fit_records': 18, 'input_fit_records': 36}
    verify_stage_partition(rows, folds, d, 0)
    for field, value in (('basis_source_keys', head), ('readout_source_keys', basis),
                         ('input_source_count', 5), ('basis_fit_records', 36), ('partition_seed', 1)):
        bad = deepcopy(d); bad[field] = value
        with pytest.raises(ValueError, match='partition differs'):
            verify_stage_partition(rows, folds, bad, 0)
