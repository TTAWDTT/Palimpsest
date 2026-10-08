"""Class-balanced sigmoid rate proxies on explicit training processing groups.

A finite penalty and a stationary non-convex solution do not certify hard
classification constraints. Exact held rates remain the acceptance criterion.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit

from .directed_margin import processing_pairs
from .paired_stability import StableRule
from .source_view_risk import fit_source_risk, source_objective

METHOD = 'L2 source logistic plus class-balanced sigmoid BA/drop penalties'


@dataclass(frozen=True)
class RatePanel:
    keys: tuple
    averaging: csr_matrix
    before: tuple[int, ...]
    after: tuple[int, ...]


def training_rate_panel(records, labels, weights, sources, *, variants):
    y, w = np.asarray(labels), np.asarray(weights, float)
    # Reuse strict complete fit-role/source/class validation, not inferred order.
    processing_pairs(records, y, w, sources, variants=variants)
    groups = defaultdict(list)
    for i, row in enumerate(records):
        key = tuple(row[k] for k in ('domain', 'scene', 'condition', 'variant'))
        groups[key].append(i)
        if key[1] != 'all':
            groups[(key[0], 'all', *key[2:])].append(i)
    keys = tuple(sorted(groups)); lookup = {key: i for i, key in enumerate(keys)}
    data, row_ids, columns = [], [], []
    for group, key in enumerate(keys):
        for label in (0, 1):
            ids = [i for i in groups[key] if y[i] == label]
            if not ids:
                raise ValueError('Training rate class absent')
            data.extend((w[ids]/(2*w[ids].sum())).tolist())
            row_ids.extend([group]*len(ids)); columns.extend(ids)
    averaging = csr_matrix((data, (row_ids, columns)), shape=(len(keys), len(records)))
    edges = []
    def add(first, second):
        a = {(records[i]['domain'], records[i]['src']): int(y[i]) for i in groups[first]}
        b = {(records[i]['domain'], records[i]['src']): int(y[i]) for i in groups[second]}
        if a != b:
            raise ValueError('Training rate pair coverage differs')
        edges.append((lookup[first], lookup[second]))
    for key in keys:
        if key[2] != 'original':
            add((*key[:2], 'original', key[3]), key)
    for domain, scene, condition in sorted({key[:3] for key in keys}):
        for first, second in combinations(variants, 2):
            add((domain, scene, condition, first), (domain, scene, condition, second))
    return RatePanel(keys, averaging, tuple(a for a, _ in edges), tuple(b for _, b in edges))


def rate_penalty(parameters, z, signed, panel, *, smoothing=.25):
    theta, z, signed = map(lambda a: np.asarray(a, float), (parameters, z, signed))
    if (z.ndim != 2 or theta.shape != (z.shape[1]+1,) or signed.shape != (len(z),)
            or panel.averaging.shape[1] != len(z) or not panel.before
            or not set(signed.tolist()) <= {-1, 1}
            or not all(np.isfinite(a).all() for a in (theta, z, signed))
            or not np.isfinite(smoothing) or smoothing <= 0):
        raise ValueError('Invalid rate proxy inputs')
    with np.errstate(over='ignore', invalid='ignore'):
        margins = signed*(z@theta[:-1]+theta[-1])/smoothing
    if not np.isfinite(margins).all():
        raise ValueError('Nonfinite rate margins')
    error = expit(-margins)
    rates = np.asarray(panel.averaging@error)
    before, after = np.array(panel.before), np.array(panel.after)
    absolute = np.maximum(rates-.2, 0)
    degradation = np.maximum(rates[after]-rates[before]-.02, 0)
    value = float(np.mean(absolute**2)+np.mean(degradation**2))
    coefficients = 2*absolute/len(absolute)
    np.add.at(coefficients, after, 2*degradation/len(degradation))
    np.add.at(coefficients, before, -2*degradation/len(degradation))
    row_coefficients = np.asarray(panel.averaging.T@coefficients)*(-signed/smoothing)*error*(1-error)
    gradient = np.r_[z.T@row_coefficients, row_coefficients.sum()]
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError('Nonfinite rate derivative')
    return value, gradient, rates


def fit_rate_source_risk(values, labels, weights, sources, panel, *, strength, optimizer_ftol=1e-14, **kwargs):
    if (not np.isfinite(strength) or strength < 0 or not np.isfinite(optimizer_ftol)
            or optimizer_ftol < 0):
        raise ValueError('Invalid rate strength')
    y = np.asarray(labels)
    if y.ndim != 1 or not set(y.tolist()) <= {0, 1}:
        raise ValueError('Invalid rate labels')
    y = y.astype(np.int64)
    base, baseline = fit_source_risk(values, y, weights, sources, **kwargs)
    x, w = np.asarray(values, float), np.asarray(weights, float); w = w/w.sum()
    z = (x-base.center)/base.scale; signed = 2*y-1
    initial = np.r_[base.weights, base.bias]
    ridge, temperature = kwargs.get('ridge', .01), kwargs.get('temperature', .1)
    tolerance, limit = kwargs.get('gradient_tolerance', 1e-5), kwargs.get('maximum_iterations', 2000)
    def objective(theta):
        value, gradient, losses = source_objective(theta, z, signed, w, sources,
            ridge=ridge, temperature=temperature)
        penalty, derivative, _ = rate_penalty(theta, z, signed, panel)
        return value+strength*penalty, gradient+strength*derivative, losses
    theta, diagnostic = initial, baseline
    if strength:
        solution = minimize(lambda a: objective(a)[:2], initial, jac=True, method='L-BFGS-B', options={
            'maxiter': limit, 'gtol': tolerance/10, 'ftol': optimizer_ftol, 'maxls': 40, 'maxcor': 20})
        value, gradient, _ = objective(solution.x); residual = float(np.max(np.abs(gradient)))
        if (not solution.success or not np.isfinite(value) or not np.isfinite(residual)
                or residual > tolerance or value > objective(initial)[0]+1e-10):
            raise ValueError(f'Rate optimizer refused: {solution.message};gradient={residual}')
        theta = solution.x
        diagnostic = {'objective': value, 'initial_objective': objective(initial)[0],
            'maximum_absolute_gradient': residual, 'optimizer_iterations': int(solution.nit),
            'optimizer_message': str(solution.message), 'fit_records': len(x), 'sources': len(set(sources))}
    penalty, _, rates = rate_penalty(theta, z, signed, panel)
    margins = z@theta[:-1]+theta[-1]
    hard_errors = ((margins > 0) != (y == 1)).astype(float)
    hard_rates = np.asarray(panel.averaging@hard_errors)
    first, second = np.array(panel.before), np.array(panel.after)
    # Zero path returns the baseline object exactly, including its strength field.
    rule = base if strength == 0 else StableRule(base.feature_names, base.center, base.scale,
        tuple(theta[:-1]), float(theta[-1]), strength, ridge, fit_manifest_sha256=base.fit_manifest_sha256)
    return rule, {**diagnostic, 'consistency_strength': strength, 'readout_method': METHOD,
        'optimizer_ftol': optimizer_ftol,
        'rate_smoothing': .25, 'rate_penalty': penalty, 'rate_groups': len(panel.keys),
        'rate_comparisons': len(panel.before), 'minimum_fit_soft_ba': float(1-rates.max()),
        'maximum_fit_soft_drop': float((rates[second]-rates[first]).max()),
        'minimum_fit_hard_ba_at_zero': float(1-hard_rates.max()),
        'maximum_fit_hard_drop_at_zero': float((hard_rates[second]-hard_rates[first]).max()),
        'scope': 'Single-start stationary sigmoid penalty;not hard constraints/global optimum/physical invariance'}
