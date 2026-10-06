"""Independent known boundaries, dependent bags, and nonlinear planted truth."""

import numpy as np
import pytest

from palimpsest.detection.algorithms.source_bagged_forest import (
    SourceBaggedForest, TreeNodes, bootstrap_weights, fit_source_forest,
)


def test_known_boundary_serialization_and_invalid_edges(tmp_path):
    tree = TreeNodes((1, -1, -1), (2, -1, -1), (0, -2, -2), (.5, -2., -2.), (.5, .25, .75))
    rule = SourceBaggedForest(('x',), (tree,))
    x = [[.49], [.5], [.51]]
    assert np.array_equal(rule.score(x), [-.25, -.25, .25])
    assert np.array_equal(rule.score(x), [rule.score([v])[0] for v in x])
    path = tmp_path / 'rule.json'
    rule.save(path)
    assert np.array_equal(SourceBaggedForest.load(path).score(x), rule.score(x))
    with pytest.raises(ValueError):
        SourceBaggedForest(('x',), (TreeNodes((0,), (0,), (0,), (.5,), (.5,)),))
    with pytest.raises(ValueError):
        rule.score([[np.nan]])
    with pytest.raises(ValueError):
        rule.score([[1e100]])


def test_whole_source_sampling_and_exact_stratum_mass():
    groups = np.repeat(np.arange(40), 6)
    labels = np.repeat(np.tile([0, 1], 20), 6)
    domains = np.repeat(np.repeat(['a', 'b'], 20), 6)
    w = bootstrap_weights(groups, labels, domains, seed=7)
    assert all(len(set(w[groups == g])) == 1 for g in set(groups))
    for d in ('a', 'b'):
        for label in (0, 1):
            assert w[(domains == d) & (labels == label)].sum() == pytest.approx(.25, abs=1e-15)
    row = bootstrap_weights(groups, labels, domains, seed=7, source_level=False)
    assert any(len(set(row[groups == g])) > 1 for g in set(groups))
    bad = labels.copy()
    bad[0] = 1
    with pytest.raises(ValueError, match='Conflicting'):
        bootstrap_weights(groups, bad, domains, seed=7)


def test_nonlinear_xor_planted_truth_and_export():
    corners = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    x = np.repeat(corners, 24, axis=0)
    y = np.repeat([0, 1, 1, 0], 24)
    groups = np.arange(len(x))
    rule, receipt = fit_source_forest(x, y, groups, ['a'] * len(x),
        feature_names=('x', 'y'), seed=7, trees=16, maximum_depth=3, minimum_leaf=2)
    assert np.array_equal(rule.score(corners) > 0, [False, True, True, False])
    assert receipt['native_export_max_error'] <= 1e-15
    # Incorrect labels cannot be accepted as the planted XOR answer.
    assert not np.array_equal(rule.score(corners) > 0, [True, False, False, True])
