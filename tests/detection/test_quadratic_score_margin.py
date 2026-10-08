"""Analytic polynomial identity, nonlinear separation and full portable path."""

import numpy as np
import pytest

from palimpsest.detection.algorithms.quadratic_score_margin import (
    quadratic_terms, QUADRATIC_NAMES, QuadraticMarginFitter, QuadraticScoreRule)
from palimpsest.detection.algorithms.source_hinge import fit_source_hinge


def test_quadratic_identity_and_nonlinear_margin():
    a = np.array([[1., 2., 3.]])
    b = np.array([[-1., 0., 2.]])
    assert float(quadratic_terms(a)[0]@quadratic_terms(b)[0]) == pytest.approx(30.)
    broken = quadratic_terms(b).copy(); broken[:, 4] = 5
    assert not np.isclose(float(quadratic_terms(a)[0]@broken[0]), 30.)
    x = np.array([[-1., -1., 0.], [-1., 1., 0.], [1., -1., 0.], [1., 1., 0.]])
    y = np.array([1, 0, 0, 1])
    rule, d = fit_source_hinge(quadratic_terms(x), y, np.ones(4), np.arange(4),
        feature_names=QUADRATIC_NAMES, penalty=.001)
    assert d['certificate']['passed'] and np.array_equal(rule.score(quadratic_terms(x)) > 0, y.astype(bool))
    for bad in (np.ones((2, 2)), np.array([[np.inf, 0., 0.]]), np.full((1, 3), 1e200)):
        with pytest.raises(ValueError): quadratic_terms(bad)


def test_full_bank_mapping_and_portable_head(tmp_path):
    a = np.array([0., 0., .5, 1/np.sqrt(2), .5]); b = a.copy(); b[3] *= -1
    x = np.repeat([a, a, b, b], 3, axis=0); y = np.repeat([0, 0, 1, 1], 3)
    weights = np.ones(len(x)); sources = np.repeat(np.arange(4), 3)
    fitter = QuadraticMarginFitter(('c0', 'c1', 'r0', 'r1', 'r2'), channels=2, filter_count=2)
    zero, _ = fitter.fit(x, y, weights, sources, 0)
    rule, _ = fitter.fit(x, y, weights, sources, .001)
    assert fitter.audit()['banks'] == 1 and zero.bank is rule.bank
    assert np.array_equal(rule.score(x) > 0, y.astype(bool))
    assert np.array_equal(rule.score(x), [rule.score([row])[0] for row in x])
    path = tmp_path/'rule.json'; rule.save(path)
    assert np.array_equal(QuadraticScoreRule.load(path).score(x), rule.score(x))
    previous_center = rule.center; query = x.copy(); query[:, :2] += 2; rule.score(query)
    assert rule.center == previous_center
    flipped, _ = fitter.fit(x, 1-y, weights, sources, .001)
    assert fitter.audit()['banks'] == 2 and flipped.bank is not rule.bank
    assert np.array_equal(flipped.score(x) > 0, (1-y).astype(bool))
    with pytest.raises(ValueError, match='Conflicting'):
        fitter.fit(x, y, weights, sources, .001, np.zeros(len(sources), int))
