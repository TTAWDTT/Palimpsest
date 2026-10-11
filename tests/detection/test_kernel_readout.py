"""Kernel approximation identities and a nonlinear known-answer example."""

import numpy as np
import pytest

from palimpsest.detection.algorithms.kernel_readout import fit_fourier_map, KernelRule
from palimpsest.detection.algorithms.readouts.stable_rule import fit_stable_rule


def test_kernel_norm_translation_and_fixed_pair_approximation():
    x = np.array([[0., 0.], [1., .5], [-1., .25], [.3, -.7]])
    mapper = fit_fourier_map(x, np.ones(4), feature_names=('x', 'y'), frequency_count=4096, seed=127)
    f = mapper.transform(x)
    np.testing.assert_array_equal(f, np.r_[tuple(mapper.transform([row]) for row in x)])
    np.testing.assert_allclose(np.sum(f*f, axis=1), 1, atol=1e-14, rtol=0)
    np.testing.assert_allclose(f@f.T, mapper.transform(x+2)@mapper.transform(x+2).T, atol=1e-13, rtol=0)
    z = (x-mapper.center)/mapper.scale
    exact = np.exp(-mapper.gamma*np.sum((z[:, None]-z[None])**2, axis=-1))
    assert np.max(np.abs(f@f.T-exact)) < .04
    with pytest.raises(ValueError):
        mapper.transform([[np.nan, 0.]])


def test_same_mean_and_covariance_nonlinearity_and_portability(tmp_path):
    # Both classes have mean 0 and variance 1. The AI class includes a spike
    # at zero plus symmetric tails; neither affine nor quadratic moments
    # separate these finite distributions, but a fixed kernel can do so.
    x = np.array([[-1.], [1.], [-1.], [1.], [0.], [0.], [-np.sqrt(2)], [np.sqrt(2)]])
    y = np.array([0]*4+[1]*4)
    assert all(np.isclose(x[y == k].mean(), 0) and np.isclose(x[y == k].var(), 1) for k in (0, 1))
    mapper = fit_fourier_map(x, np.ones(8), feature_names=('x',), frequency_count=128, seed=11)
    f = mapper.transform(x)
    rule, _ = fit_stable_rule(f, y, np.ones(8), np.zeros((1, f.shape[1])), np.ones(1),
                              feature_names=mapper.output_names, strength=0, ridge=.001)
    kernel = KernelRule(mapper, rule)
    np.testing.assert_array_equal(kernel.score(x) > 0, y.astype(bool))
    np.testing.assert_array_equal(kernel.score(x), [kernel.score([row])[0] for row in x])
    kernel.save(tmp_path/'rule.json')
    np.testing.assert_array_equal(KernelRule.load(tmp_path/'rule.json').score(x), kernel.score(x))
