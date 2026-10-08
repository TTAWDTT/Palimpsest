"""Analytical Bernoulli JS, finite-difference gradients and label preservation."""

import numpy as np
import pytest

from palimpsest.detection.algorithms.probability_consistency import (
    bernoulli_js,probability_objective,fit_probability_consistency,
)
from palimpsest.detection.algorithms.source_view_risk import fit_source_risk
from palimpsest.detection.algorithms.paired_stability import StableRule


def test_known_js_and_extreme_logits():
    score=np.array([-np.log(3),np.log(3)])
    value,coefficient=bernoulli_js(score,[.5,.5],[0,0])
    expected=np.log(2)+.25*np.log(.25)+.75*np.log(.75)
    np.testing.assert_allclose(value,expected,atol=2e-16,rtol=0)
    np.testing.assert_allclose(coefficient,score*(3/32),atol=2e-16,rtol=0)
    assert not np.isclose(value,expected/2,atol=1e-12)
    value,gradient=bernoulli_js([-1000.,1000.],[.5,.5],[0,0])
    np.testing.assert_allclose(value,np.log(2),atol=1e-15,rtol=0)
    assert np.isfinite(gradient).all()
    zero,gradient=bernoulli_js([0.,0.],[.2,.8],[0,0])
    np.testing.assert_allclose([zero,*gradient],0.,atol=1e-15,rtol=0)


def test_probability_gradient_and_constant_escape():
    rng=np.random.default_rng(51);z=rng.normal(size=(12,3));signed=np.repeat([-1,1],6)
    weights=np.arange(1,13,dtype=float);weights/=weights.sum();s=np.repeat(np.arange(4),3)
    theta=np.array([.3,-.2,.1,.2]);kwargs=dict(ridge=.01,temperature=.1,strength=12.)
    for g in (s,np.tile(np.arange(4),3)):
        _,gradient,_,_=probability_objective(theta,z,signed,weights,s,g,**kwargs)
        for step in (1e-4,1e-5):
            numeric=[]
            for index in range(4):
                delta=np.eye(4)[index]*step
                a=probability_objective(theta+delta,z,signed,weights,s,g,**kwargs)[0]
                b=probability_objective(theta-delta,z,signed,weights,s,g,**kwargs)[0]
                numeric.append((a-b)/(2*step))
            np.testing.assert_allclose(gradient,numeric,atol=1e-8,rtol=0)
    # Uniform predictions have zero JS; a labelled class signal still has CE gradient.
    x=np.array([[-1.],[-1.],[1.],[1.]])
    _,gradient,_,divergence=probability_objective(np.zeros(2),x,np.array([-1,-1,1,1]),np.ones(4)/4,
        np.repeat([0,1],2),np.repeat([0,1],2),**kwargs)
    assert divergence==0. and gradient[0]!=0.


def test_known_fit_zero_parity_and_serialization(tmp_path):
    x=np.array([[-2.,-.4],[-2.,.4],[-1.,-.3],[-1.,.3],[1.,-.2],[1.,.2],[2.,-.1],[2.,.1]])
    y=np.repeat([0,0,1,1],2);s=np.repeat(np.arange(4),2);w=np.ones(8)/8
    kwargs=dict(feature_names=('class','view'))
    reference,_=fit_source_risk(x,y,w,s,**kwargs)
    zero,_=fit_probability_consistency(x,y,w,s,strength=0,**kwargs)
    assert zero==reference
    rule,d=fit_probability_consistency(x,y,w,s,strength=12.,**kwargs)
    np.testing.assert_array_equal(rule.score(x)>0,y.astype(bool))
    assert len(d['starts'])==2 and all(v['maximum_absolute_gradient']<=1e-5 for v in d['starts'])
    np.testing.assert_array_equal(rule.score(x),[rule.score(row[None,:])[0] for row in x])
    p=tmp_path/'rule.json';rule.save(p)
    np.testing.assert_array_equal(StableRule.load(p).score(x),rule.score(x))
    with pytest.raises(ValueError):bernoulli_js([0.,np.nan],w[:2],s[:2])
    with pytest.raises(ValueError):bernoulli_js([0.,0.],[1.,0.],[0,0])
    with pytest.raises(ValueError):fit_probability_consistency(x,y,w,s,strength=-1.,**kwargs)
