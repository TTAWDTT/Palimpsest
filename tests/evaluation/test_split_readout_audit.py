"""Planted stage-member corruption must fail the independent saved-role check."""

from copy import deepcopy

import pytest

from tools.audit_typed_source_panel_cv import verify_stage_partition
from tests.detection.test_rate_penalty import fixture


def test_known_stage_members_and_bad_record():
    records, y, _, _ = fixture()
    # Literal preimage ranks independently checked with .NET SHA256, no producer.
    basis = [('d', '0'), ('d', '2')]
    head = [('d', '1'), ('d', '3')]
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


@pytest.mark.parametrize('labels,basis_ids', [
    ([0, 0, 0, 1, 1, 1], ('0', '4', '5')),
    ([1, 1, 1, 0, 0, 0], ('0', '1', '4'))])
def test_literal_odd_strata_and_supplied_null_members(labels, basis_ids):
    # .NET SHA256 on literal compact JSON ranks0<1<2 and4<5<3.
    # The second case reverses supplied labels, leaving metadata truth unchanged.
    rows = [dict(domain='d', scene='s', src=str(i), role='fit',
        label='FAKE' if labels[i] else 'REAL', truth_label='FAKE' if i >= 3 else 'REAL')
        for i in range(6) for _ in range(9)]
    folds = {'d/'+str(i): 2 for i in range(6)}
    d = {'basis_source_keys': [('d', i) for i in basis_ids],
        'readout_source_keys': [('d', str(i)) for i in range(6) if str(i) not in basis_ids],
        'partition_seed': 20261008, 'input_source_count': 6, 'basis_source_count': 3,
        'basis_fit_records': 27, 'input_fit_records': 54}
    verify_stage_partition(rows, folds, d, 0)
