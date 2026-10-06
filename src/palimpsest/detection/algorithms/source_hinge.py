"""Finite-view maximum hinge readout with an explicit LP certificate.

This is conventional robust linear classification, not a new SVM principle.
The finite convex hull of observed features does not cover unknown propagation.
"""

from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack, eye

from .paired_stability import StableRule


class MarginFitRefused(ValueError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__('Source hinge LP refused: '+str(diagnostic))


def check_lp_certificate(c, a, b, x, dual, *, free_variables, tolerance=1e-6):
    """Recompute primal/dual feasibility and gap; finite precision, not exact proof."""
    c, b, x, dual = (np.asarray(v, float) for v in (c, b, x, dual))
    if (x.shape != c.shape or dual.shape != b.shape or a.shape != (len(b), len(c))
            or not all(np.isfinite(v).all() for v in (c,b,x,dual))):
        raise ValueError('Invalid LP certificate dimensions or values')
    reduced = c-a.T@dual
    residuals = {'primal_inequality':max(0.,float(np.max(a@x-b))),
        'primal_nonnegative':max(0.,float(-np.min(x[free_variables:]))),
        'dual_inequality_sign':max(0.,float(np.max(dual))),
        'dual_free_stationarity':float(np.max(np.abs(reduced[:free_variables]), initial=0)),
        'dual_nonnegative_stationarity':max(0.,float(-np.min(reduced[free_variables:]))),
        'duality_gap':abs(float(c@x-b@dual))}
    if max(residuals.values()) > tolerance:
        raise ValueError('LP certificate residual exceeds tolerance: '+str(residuals))
    return {**residuals, 'primal_objective':float(c@x), 'dual_objective':float(b@dual),
            'passed':True, 'tolerance':tolerance}


def fit_source_hinge(values, labels, sample_weights, sources, *, feature_names,
                     maximum_source=True, penalty=.001, scale_floor=.001,
                     time_limit=120, manifest_sha=''):
    x, y = np.asarray(values, float), np.asarray(labels)
    weights, sources = np.asarray(sample_weights, float), np.asarray(sources)
    if (x.ndim != 2 or not len(x) or x.shape[1] != len(feature_names)
            or y.shape != (len(x),) or set(y.tolist()) != {0,1}
            or weights.shape != y.shape or sources.shape != y.shape or sources.dtype.kind not in 'iu'
            or set(sources.tolist()) != set(range(int(sources.max())+1))
            or not all(np.isfinite(v).all() for v in (x,weights)) or min(weights) <= 0
            or not np.isfinite([penalty,scale_floor,time_limit]).all()
            or min(penalty,scale_floor,time_limit) <= 0):
        raise ValueError('Invalid source hinge inputs')
    for source in range(int(sources.max())+1):
        if len(set(y[sources==source].tolist())) != 1:
            raise ValueError('Conflicting labels within hinge source')
    start = perf_counter()
    weights = weights/weights.sum()
    center = np.sum(x*weights[:,None], axis=0)
    scale = np.maximum(np.sqrt(np.sum((x-center)**2*weights[:,None],axis=0)), scale_floor)
    z, signed = (x-center)/scale, 2*y-1
    groups = sources if maximum_source else np.arange(len(x))
    group_weights = np.bincount(groups, weights=weights)
    n, d, g = len(x), x.shape[1], len(group_weights)
    # theta=(w,bias,source_slacks,abs_w); w/bias are free, other variables>=0.
    slack = csr_matrix((-np.ones(n),(np.arange(n),groups)), shape=(n,g))
    inequalities = hstack((csr_matrix(-signed[:,None]*z), csr_matrix(-signed[:,None]),
                           slack, csr_matrix((n,d))),format='csr')
    auxiliary = csr_matrix((d,1+g))
    a = vstack((inequalities, hstack((eye(d),auxiliary,-eye(d))),
                hstack((-eye(d),auxiliary,-eye(d)))),format='csr')
    b = np.r_[-np.ones(n), np.zeros(2*d)]
    c = np.r_[np.zeros(d+1), group_weights, np.full(d,penalty)]
    solution = linprog(c,A_ub=a,b_ub=b,bounds=[(None,None)]*(d+1)+[(0,None)]*(g+d),
        method='highs-ipm',options={'time_limit':time_limit,
            'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8,
            'ipm_optimality_tolerance':1e-9})
    base = {'solver':'SciPy HiGHS-IPM','scipy':scipy.__version__, 'status':int(solution.status),
        'message':str(solution.message),'fit_records':n,'sources':int(sources.max())+1,
        'maximum_source':maximum_source,'penalty_l1':penalty,'elapsed_s':perf_counter()-start,
        'lp_variables':len(c),'lp_inequalities':len(b),'time_limit_s':time_limit}
    if not solution.success:
        raise MarginFitRefused(base)
    try:
        certificate = check_lp_certificate(c,a,b,solution.x,solution.ineqlin.marginals,free_variables=d+1)
    except ValueError as error:
        raise MarginFitRefused({**base,'certificate_error':str(error)}) from error
    if certificate['primal_objective'] > 1+1e-6:
        raise MarginFitRefused({**base,'certificate_error':'Worse than feasible constant hinge1'})
    theta = solution.x
    losses = np.maximum(1-signed*(z@theta[:d]+theta[d]),0)
    worst = np.zeros(int(sources.max())+1)
    np.maximum.at(worst,sources,losses)
    rule = StableRule(tuple(feature_names),tuple(center),tuple(scale),tuple(theta[:d]),
        float(theta[d]),0.,penalty,fit_manifest_sha256=manifest_sha)
    return rule,{**base,'certificate':certificate,'optimizer_iterations':int(solution.nit),
        'maximum_hinge_loss':float(max(worst)), 'weighted_source_max_hinge':float(np.bincount(sources,weights=weights)@worst),
        'weighted_mean_hinge':float(weights@losses),'l1_norm':float(np.abs(theta[:d]).sum()),
        'nonzero_weights_at_1e-10':int(np.sum(np.abs(theta[:d])>1e-10)),
        'artifact_note':'StableRule legacy ridge field stores L1 penalty;fit objective is explicitly this LP',
        'scope':'Finite-precision LP optimality,not real propagation invariance'}
