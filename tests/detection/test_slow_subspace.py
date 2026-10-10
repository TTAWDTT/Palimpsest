"""Independent diagonal covariance controls, including stable but classless data."""

import itertools

import numpy as np
import pytest

from palimpsest.detection.algorithms.readouts.stable_rule import StableRule
from palimpsest.detection.algorithms.slow_subspace import fit_slow_subspace,collapse_readout


def test_stability_can_discard_the_class_and_protection_does_not_remove_drift():
    rows=[];labels=[];sources=[]
    for source,(identity,label) in enumerate(itertools.product((-1,1),repeat=2)):
        for view in (-1,1):
            rows.append([identity,label+2*view]);labels.append((label+1)//2);sources.append(source)
    x=np.array(rows,float);w=np.ones(8)/8
    slow,d=fit_slow_subspace(x,labels,w,sources,dimension=1,mode='slow')
    assert d['mean_paired_energy']==0 and d['retained_class_mean_energy_fraction']==0
    protected,p=fit_slow_subspace(x,labels,w,sources,dimension=1,mode='protected_slow')
    assert p['retained_class_mean_energy_fraction']==pytest.approx(1)
    assert p['mean_paired_energy']==pytest.approx(16/5)
    assert abs(protected.basis[1,0])==pytest.approx(1/np.sqrt(5))
    head=StableRule(('slow/0',),(.2,),(2.,),(.7,),.3,0,.01,threshold=.1)
    rule=collapse_readout(protected,head,feature_names=('identity','class_and_view'))
    np.testing.assert_allclose(rule.score(x),head.score(protected.transform(x)),atol=1e-15,rtol=0)
    assert np.array_equal(slow.transform(x),np.array([slow.transform([v])[0] for v in x]))


def test_diagonal_invariant_label_and_wrong_source_control():
    x=np.array([[label,source+view] for source,label in enumerate((-1,-1,1,1)) for view in (-1,1)],float)
    y=np.repeat([0,0,1,1],2);s=np.repeat(np.arange(4),2);w=np.ones(8)/8
    mapping,d=fit_slow_subspace(x,y,w,s,dimension=1,mode='slow')
    assert d['mean_paired_energy']==pytest.approx(0,abs=1e-14)
    assert d['retained_class_mean_energy_fraction']>0.99
    value=mapping.transform(x)[:,0]
    assert len(np.unique(np.round(value,10)))==2
    with pytest.raises(ValueError,match='one class'):
        fit_slow_subspace(x,y,w,np.tile(np.arange(4),2),dimension=1,mode='slow')
    with pytest.raises(ValueError):fit_slow_subspace(x,y,w,s,dimension=3,mode='slow')
