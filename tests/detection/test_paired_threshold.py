"""Known matched threshold and independent brute-force exact-count selector."""

from fractions import Fraction

import numpy as np
import pytest

from palimpsest.detection.algorithms.paired_stability import StableRule
from palimpsest.detection.algorithms.paired_threshold import paired_threshold


def rule():
    return StableRule(('score',), (0.,), (1.,), (1.,), 0., 0., .01)


def test_known_perfect_processing_threshold_and_role_refusal():
    views = {}
    for condition, values in (('original', [-2., -1., 1., 2.]), ('transfer', [-1., -.5, .5, 1.])):
        rows = [dict(domain='d', scene='all', condition=condition, variant='raw', src=str(i), role='threshold') for i in range(4)]
        views[condition] = rows, np.array(values)[:, None], np.array([0, 0, 1, 1])
    fitted, diagnostic = paired_threshold(rule(), views, variants=('raw',))
    assert fitted.threshold == -.5 and diagnostic['joint_calibration_feasible']
    assert diagnostic['minimum_exact_calibration_ba'] == '1' and diagnostic['maximum_exact_calibration_drop'] == '0'
    bad = {k: ([{**r, 'role': 'selection'} for r in rows], values, y) for k, (rows, values, y) in views.items()}
    with pytest.raises(ValueError):
        paired_threshold(rule(), bad, variants=('raw',))


def test_random_ties_against_brute_force_integer_rank():
    rng = np.random.default_rng(20261008)
    for _ in range(8):
        labels = np.tile([0, 1], 4); scores = rng.integers(-3, 4, size=(3, 8))
        conditions = ('original', 'a', 'b'); views = {}
        for c, values in zip(conditions, scores):
            rows = [dict(domain='d', scene='all', condition=c, variant='raw', src=str(i), role='threshold') for i in range(8)]
            views[c] = rows, values.astype(float)[:, None], labels
        fitted, diagnostic = paired_threshold(rule(), views, variants=('raw',))
        best = None
        for threshold in [np.nextafter(float(scores.min()), -np.inf), *map(float, np.unique(scores))]:
            rates = [[Fraction(int(np.sum(((v>threshold)==labels.astype(bool))[labels==label])),4)
                      for label in (0, 1)] for v in scores]
            ba = [sum(r)/2 for r in rates]; low = min(ba); drop = max(ba[0]-ba[1], ba[0]-ba[2]); cls = min(min(r) for r in rates)
            if low >= Fraction(4, 5) and drop <= Fraction(1, 50): rank = (2, low, -drop, cls, -abs(threshold), -threshold)
            elif low >= Fraction(4, 5): rank = (1, -drop, low, cls, -abs(threshold), -threshold)
            else: rank = (0, low, -drop, cls, -abs(threshold), -threshold)
            if best is None or rank > best[0]: best = rank, threshold
        assert fitted.threshold == best[1]
        assert diagnostic['threshold_states'] == len(np.unique(scores))+1
