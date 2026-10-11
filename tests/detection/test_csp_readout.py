"""Known off-diagonal classes, fold-local filters and portable calibration."""

from dataclasses import replace

import numpy as np
import pytest

from palimpsest.detection.algorithms.csp_readout import fit_csp_map,fit_csp_source,CSPRule,calibrate_csp_rule


def test_known_same_diagonal_classes_and_portable_source_rule(tmp_path):
    names=('constant0','constant1','root0','root1','root2')
    a=np.array([0.,0.,.5,1/np.sqrt(2),.5]);b=a.copy();b[3]*=-1
    x=np.repeat([a,a,b,b],3,axis=0);y=np.repeat([0,0,1,1],3);w=np.ones(len(x));s=np.repeat(np.arange(4),3)
    mapper,_=fit_csp_map(x,y,w,feature_names=names,channels=2,filter_count=2)
    mapped=mapper.transform(x);assert not np.array_equal(mapped[0],mapped[-1])
    rule,diagnostic=fit_csp_source(x,y,w,s,feature_names=names,strength=1,channels=2,filter_count=2)
    assert np.array_equal(rule.score(x)>0,y.astype(bool)) and diagnostic['maximum_absolute_gradient']<=1e-5
    assert np.array_equal(rule.score(x),[rule.score([row])[0] for row in x])
    path=tmp_path/'rule.json';rule.save(path);assert np.array_equal(CSPRule.load(path).score(x),rule.score(x))
    filters=rule.mapper.filters;rule.score([np.zeros(5)])
    assert filters==rule.mapper.filters

    def calibrate(head,views):
        _,values,labels=views['test'];assert np.array_equal(head.score(values),rule.score(x))
        assert np.array_equal(labels,y)
        return replace(head,threshold=.1),{'passed':True}
    fixed,control=calibrate_csp_rule(rule,{'test':([{'role':'threshold'}]*len(x),x,y)},calibrate)
    assert fixed.threshold==.1 and control['passed'] and np.array_equal(fixed.score(x),rule.score(x))


def test_csp_rejects_missing_class_and_corrupt_covariance():
    x=np.tile([.5,1/np.sqrt(2),.5],(4,1));names=('a','b','c')
    with pytest.raises(ValueError):fit_csp_map(x,[0]*4,np.ones(4),feature_names=names,channels=2,filter_count=2)
    x[0,0]*=2
    with pytest.raises(ValueError,match='norm'):
        fit_csp_map(x,[0,0,1,1],np.ones(4),feature_names=names,channels=2,filter_count=2)
