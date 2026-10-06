"""Known probability mixture, extreme numerical control and explicit artifact."""

import numpy as np
import pytest
from scipy.special import expit,logit

from palimpsest.detection.algorithms.condition_mixture import ConditionMixtureRule,probability_pool_logits
from palimpsest.detection.algorithms.paired_stability import StableRule


def component(weight,bias,names=('x',)):
    return StableRule(names,(0.,),(1.,),(weight,),bias,0,.01)


def test_probability_pool_and_extreme_logits():
    a,b,g = np.array([2.,-1.]),np.array([-3.,4.]),np.array([-.7,.8])
    expected = logit((1-expit(g))*expit(a)+expit(g)*expit(b))
    np.testing.assert_allclose(probability_pool_logits(a,b,g),expected,rtol=0,atol=1e-14)
    np.testing.assert_array_equal(probability_pool_logits([1000.,-1000.],[1000.,-1000.],[0.,0.]),[1000.,-1000.])
    assert probability_pool_logits([1000.],[-1000.],[0.])[0] == 0


def test_feature_only_gate_uniform_and_artifact_roundtrip(tmp_path):
    rule = ConditionMixtureRule(component(0,2),component(0,-2),component(1,0))
    x = np.array([[-10.],[10.]])
    assert tuple(rule.score(x)>0) == (True,False)
    assert np.array_equal(rule.score(x),[rule.score([row])[0] for row in x])
    uniform = ConditionMixtureRule(rule.original,rule.processed,rule.gate,uniform=True)
    np.testing.assert_array_equal(uniform.score(x),[0.,0.])
    path = tmp_path/'rule.json';rule.save(path)
    np.testing.assert_array_equal(ConditionMixtureRule.load(path).score(x),rule.score(x))


def test_invalid_pool_or_expert_schema_refused():
    with pytest.raises(ValueError):
        probability_pool_logits([0],[0],[np.nan])
    with pytest.raises(ValueError):
        ConditionMixtureRule(component(0,0),component(0,0,names=('wrong',)),component(0,0))
