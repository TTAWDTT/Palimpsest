"""Known margin solution, source grouping and corrupted certificate controls."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from palimpsest.detection.algorithms.source_hinge import fit_source_hinge, check_lp_certificate


def test_known_symmetric_source_margin():
    x = np.array([[-2.],[-1.],[1.],[2.]])
    y = np.array([0,0,1,1]); sources = np.array([0,0,1,1])
    rule, diagnostic = fit_source_hinge(x,y,np.ones(4),sources,feature_names=('x',))
    # Equivalent unstandardized separator is w=1,bias0,hard hinge zero.
    np.testing.assert_allclose(np.array(rule.weights)/rule.scale,[1],rtol=0,atol=1e-7)
    assert abs(rule.bias) < 1e-7
    assert diagnostic['certificate']['passed']
    assert diagnostic['weighted_source_max_hinge'] < 1e-7
    assert diagnostic['certificate']['primal_objective'] == pytest.approx(.001*np.sqrt(2.5))


def test_max_source_risk_counts_each_source_once():
    # Conflicting feature observations but consistent source labels. Replicating
    # a view with its divided weight must not change finite maximum-source risk.
    x = np.array([[-2.],[-.2],[.2],[2.]])
    y = np.array([0,0,1,1]); sources = np.array([0,0,1,1])
    a, da = fit_source_hinge(x,y,np.ones(4)/4,sources,feature_names=('x',))
    b, db = fit_source_hinge(np.repeat(x,2,axis=0),np.repeat(y,2),np.ones(8)/8,
        np.repeat(sources,2),feature_names=('x',))
    np.testing.assert_allclose(a.weights,b.weights,rtol=0,atol=1e-7)
    assert da['certificate']['primal_objective'] == pytest.approx(db['certificate']['primal_objective'])


def test_infeasible_or_false_dual_certificate_refused():
    # min x subject x>=1: optimum primal x1, marginal-1.
    c,a,b = np.array([1.]),csr_matrix([[-1.]]),np.array([-1.])
    assert check_lp_certificate(c,a,b,np.array([1.]),np.array([-1.]),free_variables=0)['passed']
    for x,dual in [(np.array([0.]),np.array([-1.])),(np.array([1.]),np.array([0.]))]:
        with pytest.raises(ValueError):
            check_lp_certificate(c,a,b,x,dual,free_variables=0)


def test_conflicting_group_labels_refused():
    with pytest.raises(ValueError,match='Conflicting'):
        fit_source_hinge(np.array([[-1.],[1.]]),np.array([0,1]),np.ones(2),np.zeros(2,dtype=int),feature_names=('x',))
