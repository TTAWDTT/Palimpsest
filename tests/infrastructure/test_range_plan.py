"""Exact byte coverage and hand-counted request/gap cost,plus refusal cases."""

import pytest

from palimpsest.io.range_plan import coalesce_ranges


def test_coalescing_known_bytes_and_limits():
    original = [(0, 9), (13, 18), (40, 49)]
    merged = coalesce_ranges(original, maximum_gap=3, maximum_span=30)
    assert merged == [(0, 18), (40, 49)]
    before = {i for a, b in original for i in range(a, b+1)}
    after = {i for a, b in merged for i in range(a, b+1)}
    assert before <= after and after-before == {10, 11, 12} and len(after) == 29
    assert coalesce_ranges(original, maximum_gap=2, maximum_span=30) == original
    assert coalesce_ranges(original, maximum_gap=100, maximum_span=18) == original
    for spans in ([], [(0, 4), (4, 9)], [(-1, 2)], [(10, 9)], [(0, 31)], [(0., 2)]):
        with pytest.raises(ValueError): coalesce_ranges(spans, maximum_gap=3, maximum_span=30)
