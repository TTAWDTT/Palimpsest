import pytest

from palimpsest.evaluation.balanced_null import balanced_source_null


def rows():
    return [{'domain': domain, 'src': f'{label}{i}', 'scene': 's',
             'role': 'fit', 'label': label} for domain in ('a', 'b')
            for label in ('REAL', 'FAKE') for i in range(4)]


def test_exact_orthogonality_and_view_consistency():
    records = rows()
    labels, receipt = balanced_source_null(records * 3, seed=7)
    assert len(labels) == 16
    assert all(c['sham_zero'] == c['sham_one'] == 2 for c in receipt['cells'])
    assert balanced_source_null(list(reversed(records)), seed=7) == (labels, receipt)
    for domain in ('a', 'b'):
        for truth in ('REAL', 'FAKE'):
            values = [labels[(domain, f'{truth}{i}')] for i in range(4)]
            assert sum(values) == 2


def test_odd_or_missing_class_refused():
    for records in (rows()[1:], [r for r in rows() if r['label'] == 'REAL'], []):
        with pytest.raises(ValueError):
            balanced_source_null(records, seed=7)


def test_nonfit_or_conflicting_metadata_refused():
    for modified in ({**rows()[0], 'role': 'selection'},
                     {**rows()[0], 'label': 'FAKE'},
                     {**rows()[0], 'scene': 'other'}):
        with pytest.raises(ValueError):
            balanced_source_null(rows() + [modified], seed=7)
