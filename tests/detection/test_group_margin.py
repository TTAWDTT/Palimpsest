"""Independent finite-difference gradient and planted shortcut controls."""

import numpy as np
import pytest

from palimpsest.detection.algorithms.group_margin import fit_group_margin, margin_objective


def test_entropy_pair_gradient_and_constant_bound():
    rng = np.random.default_rng(34)
    z = rng.normal(size=(12, 3)); y = np.tile([-1, 1], 6)
    weights = np.arange(1, 13, dtype=float); weights /= weights.sum()
    groups = np.tile(np.arange(3), 4)
    delta = rng.normal(size=(5, 3)); gram = delta.T@delta/5
    theta = np.array([.3, -.2, .1, -.15])
    for temperature in (0., .1):
        kwargs = dict(ridge=.01, strength=.1, temperature=temperature)
        value, gradient, _, eta = margin_objective(theta, z, y, weights, groups, gram, **kwargs)
        epsilon = 1e-6; numerical = []
        for coordinate in range(4):
            perturb = np.eye(4)[coordinate]*epsilon
            a = margin_objective(theta+perturb, z, y, weights, groups, gram, **kwargs)[0]
            b = margin_objective(theta-perturb, z, y, weights, groups, gram, **kwargs)[0]
            numerical.append((a-b)/(2*epsilon))
        np.testing.assert_allclose(gradient, numerical, atol=1e-8, rtol=0)
        assert np.isfinite(value) and np.isclose(eta.sum(), 1)
        constant = margin_objective(np.zeros(4), z, y, weights, groups, gram, **kwargs)[0]
        assert np.isclose(constant, np.log(2))


def test_group_objective_improves_minority_margin():
    # Equal classes in both groups; group 0 has 9x sources and a tempting cue
    # reversed in group 1. The stable cue perfectly separates both groups.
    # ERM also classifies correctly here; its minority margin is smaller.
    positive = np.array([[.4, 2.]]*18+[[.4, -2.]]*2)
    x = np.r_[positive, -positive]
    labels = np.r_[np.ones(20, int), np.zeros(20, int)]
    groups = np.r_[np.r_[np.zeros(18, int), np.ones(2, int)], np.r_[np.zeros(18, int), np.ones(2, int)]]
    delta = np.array([[0., 4.], [0., -4.]])
    ordinary, ordinary_diag = fit_group_margin(x, labels, np.ones(40), groups, delta, np.ones(2), feature_names=('stable', 'shortcut'), ridge=.1)
    grouped, grouped_diag = fit_group_margin(x, labels, np.ones(40), groups, delta, np.ones(2), feature_names=('stable', 'shortcut'),
                                   ridge=.1, temperature=.01)
    assert np.all((ordinary.score(x) > 0) == labels)
    assert np.all((grouped.score(x) > 0) == labels)
    signed = 2*labels-1
    assert min(signed[groups == 1]*grouped.score(x)[groups == 1]) > min(signed[groups == 1]*ordinary.score(x)[groups == 1])
    assert grouped_diag['maximum_group_loss'] < ordinary_diag['maximum_group_loss']
    np.testing.assert_array_equal(grouped.score(x), [grouped.score([row])[0] for row in x])
    with pytest.raises(ValueError):
        fit_group_margin(x, labels, np.ones(40), groups+2, delta, np.ones(2), feature_names=('a', 'b'))
