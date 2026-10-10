import numpy as np
import pytest

from palimpsest.detection.algorithms.source_support import fit_source_support, SourceSupportRule


def example(centroid=False):
    # REAL sources have bags[0,2],[8,10];FAKE[4,6],[12,14].
    return fit_source_support(np.arange(0, 18, 2).reshape(-1, 1)[:8],
        [0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 2, 2, 3, 3],
        feature_names=('x',), metric_weights=(1,), neighbors=2, centroid=centroid)


def test_known_distances_and_distinct_source_counting():
    rule = example()
    # Query0: REAL source distances0,8;FAKE4,12 => score4-8=-4.
    assert rule.score([[0]])[0] == -4
    assert example(centroid=True).score([[0]])[0] == -4
    # Nearest single source:REAL0,FAKE4 =>-4, not nearest two views[0,2].
    k1 = fit_source_support([[0], [2], [4], [6], [10], [12], [14], [16]],
        [0, 0, 1, 1, 0, 0, 1, 1], [0, 0, 1, 1, 2, 2, 3, 3],
        feature_names=('x',), metric_weights=(1,), neighbors=1)
    assert k1.score([[2]])[0] == -2


def test_duplicate_view_invariance_batch_and_roundtrip(tmp_path):
    rule = example()
    duplicate = SourceSupportRule(rule.feature_names, np.repeat(rule.supports, 2, axis=1),
                                 rule.labels, rule.metric_weights, rule.neighbors)
    x = np.array([[0.0], [3.5], [17.0]])
    assert np.array_equal(rule.score(x), [rule.score([v])[0] for v in x])
    assert np.array_equal(rule.score(x), duplicate.score(x))
    path = tmp_path / 'rule.json'
    rule.save(path)
    assert np.array_equal(rule.score(x), SourceSupportRule.load(path).score(x))
    blob = path.with_suffix('.npz')
    blob.write_bytes(blob.read_bytes() + b'wrong')
    with pytest.raises(ValueError, match='changed'):
        SourceSupportRule.load(path)


def test_bad_groups_labels_metric_and_query_refused():
    with pytest.raises(ValueError):
        fit_source_support([[0], [1], [2]], [0, 0, 1], [0, 0, 1],
                           feature_names=('x',), metric_weights=(1,), neighbors=1)
    with pytest.raises(ValueError):
        fit_source_support([[0], [1], [2], [3]], [0, 1, 1, 1], [0, 0, 1, 1],
                           feature_names=('x',), metric_weights=(1,), neighbors=1)
    rule = example()
    for weights, k in [((0,), 2), ((np.nan,), 2), ((1,), 3)]:
        with pytest.raises(ValueError):
            SourceSupportRule(rule.feature_names, rule.supports, rule.labels, weights, k)
    for x in ([[np.nan]], [[0, 1]]):
        with pytest.raises(ValueError):
            rule.score(x)
