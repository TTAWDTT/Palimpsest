"""Known paired variance and independently differenced objective gradient."""

import numpy as np
import pytest

from palimpsest.detection.algorithms.readouts.consistent_source_risk import source_variance_gram, consistent_objective, fit_consistent_source_risk
from palimpsest.detection.algorithms.readouts.source_view_risk import fit_source_risk


def test_known_variance_and_independent_gradient():
    z=np.array([[-2.,-1],[-2,1],[2,-2],[2,2]])
    weights=np.array([.1,.1,.4,.4]);groups=np.array([0,0,1,1]);signed=np.array([-1,-1,1,1])
    gram=source_variance_gram(z,weights,groups)
    np.testing.assert_allclose(gram,np.diag([0,3.4]),atol=1e-15,rtol=0)
    theta=np.array([.2,.3,-.1]);kwargs=dict(ridge=.01,temperature=.1,strength=.3)
    _,gradient,_=consistent_objective(theta,z,signed,weights,groups,gram,**kwargs)
    for step in (1e-5,1e-6):
        numerical=[]
        for i in range(3):
            d=np.eye(3)[i]*step
            numerical.append((consistent_objective(theta+d,z,signed,weights,groups,gram,**kwargs)[0]
                              -consistent_objective(theta-d,z,signed,weights,groups,gram,**kwargs)[0])/(2*step))
        np.testing.assert_allclose(gradient,numerical,atol=1e-8,rtol=0)


def test_zero_reference_and_finite_fit():
    x=np.array([[-2,-1],[-1,1],[1,-1],[2,1]],float);y=np.array([0,0,1,1]);w=np.ones(4)/4;s=np.repeat([0,1],2)
    kw=dict(feature_names=('class','view'))
    reference,_=fit_source_risk(x,y,w,s,**kw)
    zero,_=fit_consistent_source_risk(x,y,w,s,strength=0,**kw)
    assert zero==reference
    rule,d=fit_consistent_source_risk(x,y,w,s,strength=.1,**kw)
    assert d['maximum_absolute_gradient']<=1e-5
    assert np.array_equal(rule.score(x)>0,y==1)
    with pytest.raises(ValueError):fit_consistent_source_risk(x,y,w,s,strength=-1,**kw)
    with pytest.raises(ValueError):source_variance_gram(x,w,np.repeat([0,2],2))
