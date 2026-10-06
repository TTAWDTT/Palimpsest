"""Known scalar densities and numerical derivative controls, before real fitting."""
from dataclasses import replace

import numpy as np
import pytest

from palimpsest.detection.algorithms.conditional_score import (
    ConditionalScoreRule, fit_conditional_score, gaussian_objective, radial_coordinate,
)
from palimpsest.detection.algorithms.paired_stability import StableRule


def base():
    return StableRule(('a', 'b'), (0., 0.), (1., 1.), (1., 0.), 0., 0., .01)


def test_known_gaussian_llr_and_wrong_value(tmp_path):
    rule = ConditionalScoreRule(base(), False, 0., 1., 0., 1., (-1., 0.), (1., 0.))
    x = np.array([[-3., 1.], [-1., 2.], [0., 1.], [2., 2.]])
    np.testing.assert_array_equal(rule.score(x), 2*x[:, 0])
    assert not np.array_equal(rule.score(x), x[:, 0])
    np.testing.assert_array_equal(rule.score(x), [rule.score(v[None, :])[0] for v in x])
    path = tmp_path/'rule.json';rule.save(path)
    np.testing.assert_array_equal(ConditionalScoreRule.load(path).score(x), rule.score(x))
    with pytest.raises(ValueError):
        replace(rule, score_scale=0.)


@pytest.mark.parametrize('var_bias', [0., 35., -35.])
def test_gradient_including_clipped_variance(var_bias):
    design = np.array([[-1., 1.], [.2, 1.], [1., 1.]])
    target = np.array([-.7, .4, .6]);weights=np.array([.2, .3, .5])
    theta = np.array([.3, -.1, .1, var_bias])
    _, grad = gaussian_objective(theta, design, target, weights, ridge=1e-4)
    numeric=[]
    for index in range(2 if var_bias == -35. else 4):
        step=np.zeros(4);step[index]=1e-4
        plus=gaussian_objective(theta+step, design, target, weights, ridge=1e-4)[0]
        minus=gaussian_objective(theta-step, design, target, weights, ridge=1e-4)[0]
        numeric.append((plus-minus)/2e-4)
    np.testing.assert_allclose(grad[:len(numeric)], numeric, rtol=1e-5, atol=1e-7)
    if abs(var_bias) == 35.:
        # The clipped likelihood is exactly constant in variance parameters.
        np.testing.assert_array_equal(grad[2:], 2e-4*theta[2:])


def test_known_fit_and_invalid_inputs():
    x=np.array([[-2.,1.],[-1.5,1.],[-1.,1.],[1.,1.],[1.5,1.],[2.,1.]])
    y=np.array([0,0,0,1,1,1]);weights=np.ones(6)
    rule, diagnostics=fit_conditional_score(x,y,weights,base(),conditional=False)
    np.testing.assert_array_equal(rule.score(x)>0,y.astype(bool))
    assert all(d['maximum_absolute_gradient']<=1e-5 for d in diagnostics)
    for bad in (np.zeros((1,2)), np.array([[np.nan,1.]])):
        with pytest.raises(ValueError):radial_coordinate(bad)
    with pytest.raises(ValueError):rule.score([[1.]])
    with pytest.raises(ValueError):fit_conditional_score(x,np.zeros(6),weights,base())
    with pytest.raises(ValueError):fit_conditional_score(x,y,weights,base(),radius_override=np.ones(5))


def test_conditional_query_parity_and_radius(tmp_path):
    np.testing.assert_array_equal(radial_coordinate([[3.,4.]]), [np.log(5.)])
    rule=ConditionalScoreRule(base(),True,.2,1.5,.3,.7, (.2,-1.,.1,.2),(-.3,1.,-.2,.3))
    x=np.array([[3.,4.],[-2.,1.],[1.,2.]])
    np.testing.assert_array_equal(rule.score(x), [rule.score(row[None,:])[0] for row in x])
    path=tmp_path/'conditional.json';rule.save(path)
    np.testing.assert_array_equal(rule.score(x),ConditionalScoreRule.load(path).score(x))
