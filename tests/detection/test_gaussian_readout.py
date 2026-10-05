"""Planted covariance discrimination and independent density arithmetic."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.stats import norm

from palimpsest.detection.algorithms.gaussian_readout import GaussianRule, fit_gaussian_rule


def test_variance_signal_and_density_oracle(tmp_path):
    x = np.array([[-.5], [.5], [-.5], [.5], [-2], [2], [-2], [2]])
    labels = np.array([0]*4+[1]*4)
    rules = {}
    for mode in ('pooled', 'diagonal', 'class_full'):
        rule, _ = fit_gaussian_rule(x, labels, np.ones(8), feature_names=('x',), mode=mode)
        rules[mode] = rule
        np.testing.assert_array_equal(rule.score(x), [rule.score([row])[0] for row in x])
        rule.save(tmp_path/f'{mode}.json')
        np.testing.assert_array_equal(GaussianRule.load(tmp_path/f'{mode}.json').score(x), rule.score(x))
    assert np.ptp(rules['pooled'].score(x)) == 0
    for mode in ('diagonal', 'class_full'):
        rule = rules[mode]
        z = x[:, 0]/rule.scale[0]
        expected = (norm.logpdf(z, scale=np.sqrt(4/rule.scale[0]**2+.1))
                    - norm.logpdf(z, scale=np.sqrt(.25/rule.scale[0]**2+.1)))
        np.testing.assert_allclose(rule.score(x), expected, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(rule.score(x) > 0, labels.astype(bool))


def test_gaussian_rule_rejects_invalid_state():
    x = np.array([[1., 2.], [2., 1.], [-1., 0.], [0., -1.]])
    rule, _ = fit_gaussian_rule(x, [0, 0, 1, 1], np.ones(4), feature_names=('a', 'b'), mode='class_full')
    with pytest.raises(ValueError):
        replace(rule, quadratic=((1., 2.), (0., 1.)))
    with pytest.raises(ValueError):
        rule.score([[np.nan, 0]])
    with pytest.raises(ValueError):
        fit_gaussian_rule(x, [0]*4, np.ones(4), feature_names=('a', 'b'), mode='diagonal')
    with pytest.raises(ValueError):
        fit_gaussian_rule(x, [0, 0, 1, 1], np.ones(4), feature_names=('a', 'b'), mode='pooled', ridge=0)
