import pytest

from palimpsest.evaluation.features import IDENTITY_FIELDS
from palimpsest.evaluation.feature_prefix import validate_cached_prefix


def reference():
    row = {field: 'known' for field in IDENTITY_FIELDS}
    return [{**row, 'filename': 'a', 'variant': 'raw', 'feature': .125},
            {**row, 'filename': 'b', 'variant': 'raw', 'feature': .25}]


def test_exact_prefix_subset_and_full_coverage():
    rows = reference()
    assert validate_cached_prefix(rows[:1], rows, ('feature',), full_coverage=False)['exact_equal']
    assert validate_cached_prefix(rows, rows, ('feature',), full_coverage=True)['records'] == 2
    with pytest.raises(ValueError, match='coverage'):
        validate_cached_prefix(rows[:1], rows, ('feature',), full_coverage=True)


def test_wrong_value_identity_and_duplicate_refused():
    rows = reference()
    for bad in ([{**rows[0], 'feature': .126}], [{**rows[0], 'label': 'conflict'}], rows + rows[:1]):
        with pytest.raises(ValueError):
            validate_cached_prefix(bad, rows, ('feature',), full_coverage=False)
