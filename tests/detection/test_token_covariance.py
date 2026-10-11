"""Known correlation, singular/zero branches and limited feature-bag properties."""

import numpy as np
import pytest

from palimpsest.detection.representations.token_covariance import fixed_projection,projected_covariance
from palimpsest.detection.representations.token_statistics import clipped_token_statistics


def test_same_old_statistics_distinct_known_covariance():
    a=np.array([[-1.,-1.],[-1.,-1.],[1.,1.],[1.,1.]])
    b=np.array([[-1.,1.],[-1.,1.],[1.,-1.],[1.,-1.]])
    ca,sa,_=clipped_token_statistics(a);cb,sb,_=clipped_token_statistics(b)
    assert np.array_equal(ca,cb) and np.array_equal(sa,sb)
    x,dx=projected_covariance(a,np.eye(2));y,dy=projected_covariance(b,np.eye(2))
    np.testing.assert_allclose(x,[.5,1/np.sqrt(2),.5],rtol=0,atol=1e-14)
    np.testing.assert_allclose(y,[.5,-1/np.sqrt(2),.5],rtol=0,atol=1e-14)
    assert dx['covariance_rank']==dy['covariance_rank']==1 and not np.array_equal(x,y)
    # An elementwise unsigned square root would erase this known signed relation.
    assert y[1]<0 and np.abs(y[1])>0


def test_zero_projection_and_limited_feature_transforms():
    p=fixed_projection(8,3)
    assert not p.flags.writeable and np.array_equal(p,fixed_projection(8,3))
    x=np.random.default_rng(123).normal(size=(24,8))
    base,_=projected_covariance(x,p)
    for changed in (x[::-1],3.25*x+7.5):
        actual,_=projected_covariance(changed,p)
        np.testing.assert_allclose(actual,base,rtol=0,atol=1e-12)
    zero,diagnostic=projected_covariance(np.full((24,8),1e20),p)
    assert np.array_equal(zero,np.zeros(6)) and diagnostic['covariance_trace']==0
    with pytest.raises(ValueError):projected_covariance(x[:1],p)
    with pytest.raises(ValueError):projected_covariance(np.full_like(x,np.nan),p)
    with pytest.raises(ValueError):projected_covariance(x,np.ones((8,3)))
