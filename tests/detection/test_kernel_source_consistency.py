"""Nonlinear known-answer fitting, fold moments and portable calibration."""

from dataclasses import replace

import numpy as np

from palimpsest.detection.algorithms.kernel_readout import KernelRule
from palimpsest.detection.algorithms.kernel_source_consistency import (
    fit_kernel_source_consistency, calibrate_kernel_rule)


def test_nonlinear_source_rule_and_unseen_value_cannot_change_mapper(tmp_path):
    native = np.array([-1.,1.,-1.,1.,0.,0.,-np.sqrt(2),np.sqrt(2)])[:,None]
    x = np.repeat(native,3,axis=0); y = np.repeat([0]*4+[1]*4,3)
    weights = np.ones(len(x));sources = np.repeat(np.arange(8),3)
    rule, diagnostic = fit_kernel_source_consistency(x,y,weights,sources,
        feature_names=('x',),strength=1,frequency_count=128,seed=11)
    assert np.array_equal(rule.score(x)>0,y.astype(bool))
    assert diagnostic['sources']==8 and diagnostic['maximum_absolute_gradient']<=1e-5
    assert np.allclose(rule.mapper.center,np.sum(x*weights[:,None],axis=0)/weights.sum())
    center = rule.mapper.center;rule.score([[1000.]])
    assert rule.mapper.center==center
    assert np.array_equal(rule.score(x),[rule.score([row])[0] for row in x])
    path = tmp_path/'rule.json';rule.save(path)
    assert np.array_equal(KernelRule.load(path).score(x),rule.score(x))

    def calibrate(head, views):
        records, mapped, labels = views['toy']
        assert np.array_equal(head.score(mapped),rule.score(x)) and np.array_equal(labels,y)
        assert all(r['role']=='threshold' for r in records)
        return replace(head,threshold=.123),{'passed':True}

    fixed, result = calibrate_kernel_rule(rule,{'toy':([{'role':'threshold'}]*len(x),x,y)},calibrate)
    assert fixed.threshold==.123 and result['passed']
    assert np.array_equal(fixed.score(x),rule.score(x))
