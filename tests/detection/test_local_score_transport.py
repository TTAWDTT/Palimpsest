import numpy as np
import pytest

from palimpsest.detection.algorithms.readouts.stable_rule import StableRule
from palimpsest.detection.algorithms.local_score_transport import LocalScoreTransport, matched_source_anchors


def base():
    return StableRule(('x',), (0,), (1,), (1,), 0, .1, .01)


def test_known_additive_offset_and_source_neighbor_selection():
    rule = LocalScoreTransport(base(), np.array([[[0], [2]], [[10], [12]]]),
                              (0, 10), (1,), neighbors=1)
    assert rule.score([[1.8]])[0] == pytest.approx(-.2)
    assert rule.score([[0], [10]]) == pytest.approx([0, 10])
    assert rule.score([[11.8]])[0] == pytest.approx(9.8)


def test_zero_offset_identity_batch_and_artifact(tmp_path):
    rule = LocalScoreTransport(base(), np.array([[[0]], [[10]]]), (0, 10), (1,), neighbors=1)
    x = [[0], [2.1], [11.5]]
    assert np.array_equal(rule.score(x), base().score(x))
    assert np.array_equal(rule.score(x), [rule.score([q])[0] for q in x])
    path = tmp_path / 'rule.json'
    rule.save(path)
    assert np.array_equal(rule.score(x), LocalScoreTransport.load(path).score(x))
    path.with_suffix('.npz').write_bytes(b'bad')
    with pytest.raises(ValueError, match='changed'):
        LocalScoreTransport.load(path)


def test_anchor_coverage_and_invalid_transport_refused():
    x = np.arange(12).reshape(-1, 1)
    mask = np.array([True, True, False, False, False, False] * 2)
    bags, anchors = matched_source_anchors(base(), x, np.repeat([0, 1], 6), mask)
    assert bags.shape == (2, 6, 1)
    assert anchors == (.5, 6.5)
    for bad_mask in (np.zeros(12, bool), np.ones(12, bool)):
        with pytest.raises(ValueError):
            matched_source_anchors(base(), x, np.repeat([0, 1], 6), bad_mask)
    for weights, k in [((0,), 1), ((np.nan,), 1), ((1,), 3)]:
        with pytest.raises(ValueError):
            LocalScoreTransport(base(), bags, anchors, weights, neighbors=k)
