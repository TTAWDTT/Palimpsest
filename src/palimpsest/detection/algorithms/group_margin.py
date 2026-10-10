"""Convex logistic readouts balancing processing groups and source pairs.

Entropy-weighted group risk is a smooth approximation, not exact max risk or
a guarantee for unseen physical channels. The exported score is a StableRule.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from palimpsest.detection.algorithms.readouts.stable_rule import StableRule


def margin_objective(parameters, z, labels_signed, weights, groups, pair_gram, *, ridge, strength, temperature):
    """Value/analytic gradient; groups contiguous integers and weights sum to 1."""
    w, bias = parameters[:-1], parameters[-1]
    scores = z@w+bias
    loss = np.logaddexp(0, -labels_signed*scores)
    q = np.bincount(groups, weights=weights)
    group_loss = np.bincount(groups, weights=weights*loss)/q
    if temperature > 0:
        logits = np.log(q)+group_loss/temperature
        log_normalizer = logsumexp(logits)
        eta = np.exp(logits-log_normalizer)
        risk = temperature*log_normalizer
    else:
        eta, risk = q, float(np.sum(weights*loss))
    coefficients = weights*eta[groups]/q[groups]*(-labels_signed*expit(-labels_signed*scores))
    paired_vector = pair_gram@w
    value = risk+ridge*np.sum(w*w)+strength*(w@paired_vector)
    gradient = np.r_[z.T@coefficients+2*ridge*w+2*strength*paired_vector, coefficients.sum()]
    return float(value), gradient, group_loss, eta


def fit_group_margin(values, labels, sample_weights, groups, pair_deltas, pair_weights, *, feature_names,
                     ridge=.01, strength=0., temperature=0., scale_floor=.001,
                     maximum_iterations=500, gradient_tolerance=1e-5, manifest_sha=''):
    x, y, sw = np.asarray(values, float), np.asarray(labels), np.asarray(sample_weights, float)
    group = np.asarray(groups)
    d, pw = np.asarray(pair_deltas, float), np.asarray(pair_weights, float)
    if (x.ndim != 2 or not len(x) or x.shape[1] != len(feature_names) or y.shape != (len(x),)
            or set(y.tolist()) != {0, 1} or sw.shape != y.shape or group.shape != y.shape
            or group.dtype.kind not in 'iu' or set(group.tolist()) != set(range(int(group.max())+1))
            or d.ndim != 2 or not len(d) or d.shape[1] != x.shape[1] or pw.shape != (len(d),)):
        raise ValueError('Invalid group margin dimensions/labels/groups')
    if (not all(np.isfinite(v).all() for v in (x, sw, d, pw)) or min(sw) <= 0 or min(pw) <= 0
            or not np.isfinite([ridge, strength, temperature, scale_floor, gradient_tolerance]).all()
            or ridge <= 0 or min(strength, temperature) < 0 or min(scale_floor, gradient_tolerance) <= 0
            or maximum_iterations < 1):
        raise ValueError('Invalid group margin values')
    sw = sw/sw.sum(); pw = pw/pw.sum()
    center = np.sum(x*sw[:, None], axis=0)
    scale = np.maximum(np.sqrt(np.sum((x-center)**2*sw[:, None], axis=0)), scale_floor)
    z, delta = (x-center)/scale, d/scale
    pair_gram = delta.T@(pw[:, None]*delta)
    signed = 2*y-1

    def objective(theta):
        value, gradient, _, _ = margin_objective(theta, z, signed, sw, group, pair_gram,
                                                ridge=ridge, strength=strength, temperature=temperature)
        return value, gradient

    initial = np.zeros(x.shape[1]+1)
    solution = minimize(objective, initial, jac=True, method='L-BFGS-B',
                        options={'maxiter': maximum_iterations, 'gtol': gradient_tolerance/10,
                                 'ftol': 1e-14, 'maxls': 40, 'maxcor': 20})
    value, gradient, losses, eta = margin_objective(solution.x, z, signed, sw, group, pair_gram,
                                                  ridge=ridge, strength=strength, temperature=temperature)
    residual = float(np.max(np.abs(gradient)))
    if not solution.success or residual > gradient_tolerance or value > objective(initial)[0]+1e-10:
        raise ValueError(f'Group optimization failed: {solution.message}; gradient={residual}')
    rule = StableRule(tuple(feature_names), tuple(center), tuple(scale), tuple(solution.x[:-1]),
                      float(solution.x[-1]), strength, ridge, fit_manifest_sha256=manifest_sha)
    return rule, {'objective': value, 'initial_objective': objective(initial)[0], 'optimizer_iterations': int(solution.nit),
                  'maximum_absolute_gradient': residual, 'optimizer_message': str(solution.message),
                  'group_losses': losses.tolist(), 'group_risk_weights': eta.tolist(),
                  'maximum_group_loss': float(max(losses)), 'minimum_group_loss': float(min(losses)),
                  'paired_score_drift': float(solution.x[:-1]@pair_gram@solution.x[:-1]),
                  'temperature': temperature, 'ridge': ridge, 'strength': strength,
                  'groups': len(losses), 'fit_records': len(x), 'paired_records': len(d)}
