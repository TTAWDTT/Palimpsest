"""Known source-role panels and planted cross-role byte duplicates."""

import copy
import hashlib

import pytest

from experiments.data_preparation.development_roles.audit_roles import audit_inventory


def panels():
    rows = []
    for domain, src, role, label, conditions in (
        ('rr', 'a', 'fit', 'REAL', ('original', 'transfer', 'redigital')),
        ('chimera', 'b', 'selection', 'FAKE', ('original', 'mac_iphone', 'lg_blackfly'))):
        for condition in conditions:
            rows.append({'domain': domain, 'src': src, 'role': role, 'label': label,
                'condition': condition, 'scene': 'all', 'width': '20', 'height': '30',
                'filename': src + condition, 'sha256': hashlib.sha256((src + condition).encode()).hexdigest()})
    return rows


def assert_count(result, scope, field, expected):
    if result[scope][field] != expected:
        raise ValueError('Planted identity count differs')


def test_cross_domain_role_label_duplicate_and_wrong_count():
    rows = panels()
    assert_count(audit_inventory(rows), 'originals', 'multi_record_groups', 0)
    rows[3]['sha256'] = rows[0]['sha256']
    result = audit_inventory(rows)
    for scope in ('originals', 'all_native'):
        for field in ('multi_record_groups', 'distinct_source_groups', 'cross_role_groups', 'conflicting_label_groups'):
            assert_count(result, scope, field, 1)
    assert len(result['originals']['duplicate_groups'][0]['records']) == 2
    with pytest.raises(ValueError, match='Planted identity'):
        assert_count(result, 'originals', 'cross_role_groups', 0)


def test_same_source_identical_view_is_not_cross_role():
    rows = panels()
    rows[1]['sha256'] = rows[0]['sha256']
    result = audit_inventory(rows)
    assert_count(result, 'all_native', 'within_source_only_groups', 1)
    assert_count(result, 'all_native', 'cross_role_groups', 0)
    assert_count(result, 'originals', 'multi_record_groups', 0)


def test_duplicate_identity_and_source_conflict_refused():
    rows = panels()
    with pytest.raises(ValueError, match='Duplicate native'):
        audit_inventory(rows + [copy.deepcopy(rows[0])])
    rows[1]['role'] = 'threshold'
    with pytest.raises(ValueError, match='conflicting source'):
        audit_inventory(rows)


def test_malformed_digest_and_incomplete_panel_refused():
    rows = panels()
    rows[0]['sha256'] = 'short'
    with pytest.raises(ValueError, match='Invalid cached byte'):
        audit_inventory(rows)
    with pytest.raises(ValueError, match='Incomplete'):
        audit_inventory(panels()[:-1])
