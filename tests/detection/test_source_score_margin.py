"""Constructed class separation, cache isolation and portable margin controls."""

import numpy as np
import pytest

from palimpsest.detection.models.frozen_features.cure.source_score_margin import ScoreMarginFitter
from palimpsest.detection.models.frozen_features.cure.source_score_subspace import ScoreSubspaceRule


def test_known_score_bank_margin_and_label_isolation(tmp_path):
    a = np.array([0., 0., .5, 1/np.sqrt(2), .5])
    b = a.copy(); b[3] *= -1
    x = np.repeat([a, a, b, b], 3, axis=0)
    y = np.repeat([0, 0, 1, 1], 3); w = np.ones(len(x)); s = np.repeat(np.arange(4), 3)
    names = ('c0', 'c1', 'r0', 'r1', 'r2')
    fitter = ScoreMarginFitter(names, channels=2, filter_count=2)
    mean, _ = fitter.fit(x, y, w, s, 0)
    maximum, diagnostic = fitter.fit(x, y, w, s, .001)
    assert fitter.provider.builds == 1 and mean.bank is maximum.bank
    assert diagnostic['certificate']['passed']
    assert diagnostic['weighted_source_max_hinge'] < 1e-7
    assert np.array_equal(maximum.score(x) > 0, y.astype(bool))
    assert np.array_equal(maximum.score(x), [maximum.score([row])[0] for row in x])
    path = tmp_path/'rule.json'; maximum.save(path)
    assert np.array_equal(maximum.score(x), ScoreSubspaceRule.load(path).score(x))
    flipped, _ = fitter.fit(x, 1-y, w, s, .001)
    assert fitter.provider.builds == 2 and flipped.bank is not maximum.bank
    assert np.array_equal(flipped.score(x) > 0, (1-y).astype(bool))
    with pytest.raises(ValueError, match='Conflicting'):
        fitter.fit(x, y, w, s, .001, np.zeros(len(s), int))


def test_zero_information_has_constant_hinge_one():
    # Equal classes and identical zero features cannot have below-one hinge.
    from palimpsest.detection.algorithms.readouts.source_hinge import fit_source_hinge
    rule, diagnostic = fit_source_hinge(np.zeros((4, 3)), np.array([0, 0, 1, 1]),
        np.ones(4), np.arange(4), feature_names=('a', 'b', 'c'))
    assert diagnostic['certificate']['primal_objective'] == pytest.approx(1., abs=1e-8)
    assert np.array_equal(rule.score(np.zeros((4, 3))), np.repeat(rule.bias, 4))
