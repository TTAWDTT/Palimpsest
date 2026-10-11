"""Explicit distinct pairs, diagonal whitening and portable source distances."""

from itertools import combinations

import numpy as np
import pytest

from palimpsest.detection.algorithms.paired_metric import (
    PairedMetricSupportRule, fit_query_map, paired_scatter, transform_rows,
)
from palimpsest.detection.algorithms.source_support import fit_source_support


def toy():
    x = np.array([[0., 0.], [2., 0.], [0., 2.], [4., 4.], [8., 4.], [4., 8.]])
    return x, np.array([1/9]*3 + [2/9]*3), np.repeat([0, 1], 3)


def test_explicit_unordered_pair_covariance():
    x, weights, sources = toy()
    expected = np.zeros((2, 2))
    for group, source_weight in ((0, 1/3), (1, 2/3)):
        pairs = [x[a] - x[b] for a, b in combinations(np.flatnonzero(sources == group), 2)]
        expected += source_weight * sum(np.outer(v, v) for v in pairs) / 3
    assert np.allclose(paired_scatter(x, weights, sources), expected, atol=1e-14, rtol=0)
    # Renaming whole source IDs does not change membership;mix actual members.
    uniform = np.ones(len(x))
    wrong = np.tile([0, 1], 3)
    assert not np.allclose(paired_scatter(x, uniform, wrong),
                           paired_scatter(x, uniform, sources), atol=1e-14, rtol=0)


def test_diagonal_inverse_root_and_bad_trace():
    x = np.array([[1., 0.], [-1., 0.], [0., 2.], [0., -2.]])
    center, scale, matrix, diagnostics = fit_query_map(x, np.ones(4), np.repeat([0, 1], 2), paired=True)
    assert np.array_equal(center, [0., 0.])
    # Standardized pair scatter is diag(4,4),nugget.04.
    assert np.allclose(matrix, np.eye(2) / np.sqrt(4.04), atol=1e-15, rtol=0)
    assert diagnostics['inverse_root_max_error'] < 1e-14
    assert np.array_equal(transform_rows(x, center, scale, matrix),
                          np.vstack([transform_rows([v], center, scale, matrix) for v in x]))
    with pytest.raises(ValueError, match='trace'):
        fit_query_map(np.zeros((4, 2)), np.ones(4), np.repeat([0, 1], 2), paired=True)


def test_rule_roundtrip_and_corrupt_map(tmp_path):
    x, weights, sources = toy()
    center, scale, matrix, _ = fit_query_map(x, weights, sources, paired=True)
    labels = np.repeat([0, 1], 3)
    support = fit_source_support(transform_rows(x, center, scale, matrix), labels, sources,
        feature_names=('a', 'b'), metric_weights=(1., 1.), neighbors=1)
    rule = PairedMetricSupportRule(('a', 'b'), center, scale, matrix, support)
    query = np.array([[.25, 1.], [5., 7.]])
    assert np.array_equal(rule.score(query), [rule.score([q])[0] for q in query])
    # Independent direct differences to every view,then nearest source per class.
    mapped = transform_rows(query, center, scale, matrix)
    expected = []
    for q in mapped:
        distance = np.linalg.norm(support.supports - q, axis=2).min(axis=1)
        expected.append(distance[0] - distance[1])
    assert np.allclose(rule.score(query), expected, atol=1e-14, rtol=0)
    path = tmp_path/'rule.json'
    rule.save(path)
    assert np.array_equal(rule.score(query), PairedMetricSupportRule.load(path).score(query))
    blob = path.with_name('rule_map.npz')
    blob.write_bytes(blob.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        PairedMetricSupportRule.load(path)


def test_bad_panel_and_nonfinite_query_refused():
    x, weights, sources = toy()
    with pytest.raises(ValueError, match='equal weights'):
        paired_scatter(x, np.arange(1., 7.), sources)
    x[0, 0] = np.nan
    with pytest.raises(ValueError, match='Invalid'):
        paired_scatter(x, weights, sources)
