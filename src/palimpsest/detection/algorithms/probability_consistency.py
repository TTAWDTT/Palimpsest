"""Bernoulli Jensen-Shannon consistency with existing source classification risk.

Known consistency principle, adapted to matched frozen-feature views. The
nonconvex optimizer returns a conventional linear rule, not an invariance proof.
"""

from dataclasses import replace

import numpy as np
from scipy.optimize import minimize

from palimpsest.detection.algorithms.readouts.source_view_risk import fit_source_risk, source_objective


def _log_mean(values, weights, groups):
    masses=np.bincount(groups,weights=weights)
    terms=values+np.log(weights/masses[groups])
    maxima=np.full(len(masses),-np.inf);np.maximum.at(maxima,groups,terms)
    sums=np.bincount(groups,weights=np.exp(terms-maxima[groups]))
    return maxima+np.log(sums),masses


def bernoulli_js(scores,weights,groups):
    scores,weights,groups=np.asarray(scores,float),np.asarray(weights,float),np.asarray(groups)
    if (scores.ndim!=1 or not len(scores) or weights.shape!=scores.shape or groups.shape!=scores.shape
            or groups.dtype.kind not in 'iu' or set(groups.tolist())!=set(range(int(groups.max())+1))
            or not np.isfinite(scores).all() or not np.isfinite(weights).all() or np.min(weights)<=0):
        raise ValueError('Invalid probability consistency arrays')
    weights=weights/weights.sum()
    log_p=-np.logaddexp(0.,-scores);log_q=-np.logaddexp(0.,scores)
    lp,mass=_log_mean(log_p,weights,groups);lq,_=_log_mean(log_q,weights,groups)
    p,q=np.exp(log_p),np.exp(log_q)
    local_entropy=-p*log_p-q*log_q
    mean_entropy=-np.exp(lp)*lp-np.exp(lq)*lq
    value=float(mass@mean_entropy-weights@local_entropy)
    if not -1e-12<=value<=np.log(2)+1e-12:raise ValueError('Bernoulli JS outside numerical bound')
    # h'(mean)-h'(individual); log_q-log_p=-score, no probability clipping.
    coefficients=weights*p*q*(lq[groups]-lp[groups]+scores)
    return value,coefficients


def probability_objective(theta,z,signed,weights,sources,js_groups,*,ridge,temperature,strength):
    value,gradient,losses=source_objective(theta,z,signed,weights,sources,ridge=ridge,temperature=temperature)
    divergence,coefficients=bernoulli_js(z@theta[:-1]+theta[-1],weights,js_groups)
    gradient+=strength*np.r_[z.T@coefficients,coefficients.sum()]
    return value+strength*divergence,gradient,losses,divergence


def fit_probability_consistency(values,labels,weights,sources,*,strength,consistency_groups=None,**kwargs):
    if not np.isfinite(strength) or strength<0:raise ValueError('Invalid probability consistency strength')
    base,baseline=fit_source_risk(values,labels,weights,sources,**kwargs)
    x,y,w,s=np.asarray(values,float),np.asarray(labels),np.asarray(weights,float),np.asarray(sources)
    w=w/w.sum();g=s if consistency_groups is None else np.asarray(consistency_groups)
    z=(x-base.center)/base.scale
    initial=np.r_[base.weights,base.bias]
    _,_=bernoulli_js(z@initial[:-1]+initial[-1],w,g)
    ridge=kwargs.get('ridge',.01);temperature=kwargs.get('temperature',.1)
    tolerance=kwargs.get('gradient_tolerance',1e-5);limit=kwargs.get('maximum_iterations',500)
    def objective(theta):
        value,gradient,_,_=probability_objective(theta,z,2*y-1,w,s,g,
            ridge=ridge,temperature=temperature,strength=strength)
        return value,gradient
    if strength==0:
        divergence=bernoulli_js(z@initial[:-1]+initial[-1],w,g)[0]
        return base,{**baseline,'probability_js':divergence,'consistency_strength':0.,'starts':[]}
    solutions=[];starts=[]
    for name,point in (('source',initial),('zero',np.zeros_like(initial))):
        result=minimize(objective,point,jac=True,method='L-BFGS-B',options={
            'maxiter':limit,'gtol':tolerance/10,'ftol':1e-14,'maxls':40,'maxcor':20})
        value,gradient=objective(result.x);residual=float(np.max(np.abs(gradient)))
        if (not result.success or not np.isfinite(result.x).all() or residual>tolerance
                or value>objective(point)[0]+1e-10):
            raise ValueError(f'Probability optimizer failed ({name}): {result.message};gradient={residual}')
        starts.append({'name':name,'objective':value,'initial_objective':objective(point)[0],
                       'maximum_absolute_gradient':residual,'iterations':int(result.nit)})
        solutions.append((value,name,result.x))
    _,name,theta=min(solutions,key=lambda v:(v[0],v[1]))
    value,gradient,losses,divergence=probability_objective(theta,z,2*y-1,w,s,g,
        ridge=ridge,temperature=temperature,strength=strength)
    rule=replace(base,weights=tuple(theta[:-1]),bias=float(theta[-1]),strength=strength)
    return rule,{'objective':value,'probability_js':divergence,'consistency_strength':strength,
        'chosen_start':name,'starts':starts,'maximum_absolute_gradient':float(np.max(np.abs(gradient))),
        'maximum_source_risk':float(max(losses)),'fit_records':len(x),'sources':int(s.max())+1,
        'scope':'Two stationary starts,not global nonconvex optimum certification'}
