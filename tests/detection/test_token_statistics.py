"""Analytic clipping, information beyond a mean, and restricted equivariance."""

import numpy as np
import pytest

from palimpsest.detection.representations.token_statistics import clipped_token_statistics


def test_known_radial_clipping_and_mean_information():
    center,unit,d=clipped_token_statistics([[-1.,0.],[1.,0.],[100.,0.]])
    np.testing.assert_array_equal(center,[1.,0.]);np.testing.assert_array_equal(unit,[1.,0.])
    assert d['radius']==2. and d['clipped_fraction']==1/3
    np.testing.assert_allclose(d['spread_norm'],np.sqrt(8/3),atol=1e-15,rtol=0)
    a=clipped_token_statistics([[-1.],[1.]])[1];b=clipped_token_statistics([[0.],[0.]])[1]
    assert a[0]==1. and b[0]==0.  # Same mean, different spread.


def test_constant_majority_and_affine_feature_map():
    x=np.zeros((10,3));x[-1]=[1000.,-1000.,2000.]
    center,unit,d=clipped_token_statistics(x)
    np.testing.assert_array_equal(center,np.zeros(3));np.testing.assert_array_equal(unit,np.zeros(3))
    assert d['radius']==0. and d['clipped_fraction']==.1
    x=np.array([[-2.,1.],[0.,0.],[1.,2.],[20.,-4.]])
    c,u,_=clipped_token_statistics(x)
    for scale in (.5,2.,8.):
        shift=np.array([3.,-2.]);cc,uu,_=clipped_token_statistics(scale*x+shift)
        np.testing.assert_allclose(cc,scale*c+shift,atol=1e-12,rtol=0)
        np.testing.assert_allclose(uu,u,atol=1e-12,rtol=0)
    pc,pu,_=clipped_token_statistics(x[[3,1,2,0]])
    np.testing.assert_allclose(pc,c,atol=1e-12,rtol=0);np.testing.assert_allclose(pu,u,atol=1e-12,rtol=0)


@pytest.mark.parametrize('bad',[[],[[1.]],[[np.nan],[1.]],[[np.inf],[1.]]])
def test_invalid_tokens(bad):
    with pytest.raises(ValueError):clipped_token_statistics(bad)
