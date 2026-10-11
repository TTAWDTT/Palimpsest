"""Known loss/gradient, permutation, convex fit and source identity refusal."""

import math
import numpy as np
import pytest
from palimpsest.detection.algorithms.readouts.source_view_risk import source_objective, fit_source_risk


def test_source_risk_known_zero_and_hard_view():
    z=np.array([[1.],[2.],[-1.],[-2.]])
    signed=np.array([1,1,-1,-1]); weights=np.full(4,.25); sources=np.array([0,0,1,1])
    value,gradient,_=source_objective(np.array([0.,0.]),z,signed,weights,sources,ridge=.01,temperature=.1)
    assert abs(value-math.log(2))<1e-15
    np.testing.assert_allclose(gradient,[-.75,0],atol=1e-15,rtol=0)
    value,gradient,losses=source_objective(np.array([1.,0.]),z,signed,weights,sources,ridge=.01,temperature=.1)
    first,second=math.log1p(math.exp(-1)),math.log1p(math.exp(-2))
    probability=1/(1+math.exp((second-first)/.1))
    expected=.1*math.log((math.exp(first/.1)+math.exp(second/.1))/2)+.01
    derivative=-probability/(1+math.exp(1))-2*(1-probability)/(1+math.exp(2))+.02
    assert probability>.5 and second<=losses[0]<=first
    np.testing.assert_allclose([value,*gradient],[expected,derivative,0],atol=1e-14,rtol=0)
    for step in (1e-4,1e-6):
        theta=np.array([.3,-.2])
        _,analytical,_=source_objective(theta,z,signed,weights,sources,ridge=.01,temperature=.1)
        numerical=[]
        for axis in range(2):
            delta=np.eye(2)[axis]*step
            numerical.append((source_objective(theta+delta,z,signed,weights,sources,ridge=.01,temperature=.1)[0]-source_objective(theta-delta,z,signed,weights,sources,ridge=.01,temperature=.1)[0])/(2*step))
        np.testing.assert_allclose(numerical,analytical,atol=2e-8,rtol=0)
    order=[2,0,3,1]
    permuted=source_objective(np.array([1.,0.]),z[order],signed[order],weights[order],sources[order],ridge=.01,temperature=.1)
    np.testing.assert_allclose([permuted[0],*permuted[1]],[value,*gradient],atol=1e-15,rtol=0)


def test_source_risk_convex_fit_and_conflicting_labels():
    x=np.array([[-3.],[-2.],[-1.],[1.],[2.],[3.]])
    labels=np.array([0,0,0,1,1,1]); sources=np.array([0,0,0,1,1,1])
    for temperature in (0.,.1):
        rule,diagnostic=fit_source_risk(x,labels,np.ones(6),sources,feature_names=('known',),temperature=temperature)
        assert np.array_equal(rule.score(x)>0,labels.astype(bool))
        assert diagnostic['maximum_absolute_gradient']<=1e-5
        assert diagnostic['objective']<math.log(2)
    wrong=labels.copy();wrong[0]=1
    with pytest.raises(ValueError,match='Conflicting labels'):
        fit_source_risk(x,wrong,np.ones(6),sources,feature_names=('known',))
