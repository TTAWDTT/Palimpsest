"""Class-signed processing-order penalty, a conventional pairwise hinge loss.

Zero violation on known edges prevents correct-to-wrong flips at a common
threshold. Small average loss alone does not bound flips near that threshold.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

from .source_view_risk import fit_source_risk, source_objective
from .paired_stability import StableRule


@dataclass(frozen=True)
class DirectedPairs:
    before: tuple[int, ...]
    after: tuple[int, ...]
    signed: tuple[int, ...]
    weights: tuple[float, ...]
    cross_source_edges: int = 0


def processing_pairs(records, labels, weights, groups, *, variants, allow_virtual_sources=False):
    n = len(records); y, w, g = map(np.asarray, (labels, weights, groups))
    if (not n or any(v.shape != (n,) for v in (y, w, g)) or g.dtype.kind not in 'iu'
            or not set(y.tolist()) <= {0, 1} or not np.isfinite(w).all() or np.min(w) <= 0
            or set(g.tolist()) != set(range(int(g.max())+1)) or not variants
            or len(set(variants)) != len(variants)):
        raise ValueError('Invalid pair graph arrays')
    by_group = {}; identities = set()
    for index, row in enumerate(records):
        identity = tuple(row[k] for k in ('domain', 'src', 'condition', 'variant'))
        if row['role'] != 'fit' or identity in identities or row['variant'] not in variants:
            raise ValueError('Pair graph requires unique training-only views')
        identities.add(identity); by_group.setdefault(int(g[index]), []).append(index)
    before, after, signs, edge_weights = [], [], [], []
    w = w/w.sum(); cross_source = 0
    for group, indices in sorted(by_group.items()):
        source_keys = {(records[i]['domain'], records[i]['src']) for i in indices}
        strata = {(records[i]['domain'], records[i]['scene'], int(y[i])) for i in indices}
        if len(strata) != 1 or (not allow_virtual_sources and len(source_keys) != 1):
            raise ValueError('Conflicting source/class/scene pair group')
        lookup = {(records[i]['condition'], records[i]['variant']): i for i in indices}
        conditions = {records[i]['condition'] for i in indices}
        if (len(conditions) != 3 or 'original' not in conditions or len(lookup) != len(indices)
                or set(lookup) != {(c, v) for c in conditions for v in variants}):
            raise ValueError('Incomplete processing graph panel')
        edges = [(lookup['original', v], lookup[c, v])
            for c in sorted(conditions-{'original'}) for v in variants]
        edges += [(lookup[c, first], lookup[c, second])
            for c in sorted(conditions) for first, second in combinations(variants, 2)]
        mass = float(w[indices].sum())/len(edges)
        for first, second in edges:
            before.append(first); after.append(second); signs.append(2*int(y[first])-1); edge_weights.append(mass)
            cross_source += (records[first]['domain'], records[first]['src']) != (
                records[second]['domain'], records[second]['src'])
    return DirectedPairs(tuple(before), tuple(after), tuple(signs), tuple(edge_weights), cross_source)


def directed_penalty(parameters, z, pairs):
    theta, z = np.asarray(parameters, float), np.asarray(z, float)
    first, second = np.asarray(pairs.before, int), np.asarray(pairs.after, int)
    signed, w = np.asarray(pairs.signed, float), np.asarray(pairs.weights, float)
    if (z.ndim != 2 or theta.shape != (z.shape[1]+1,) or not len(first)
            or any(a.shape != first.shape for a in (second, signed, w))
            or np.min(first) < 0 or np.min(second) < 0 or np.max(first) >= len(z) or np.max(second) >= len(z)
            or not set(signed.tolist()) <= {-1, 1} or not np.isfinite(w).all() or np.min(w) <= 0
            or not np.isfinite(theta).all() or not np.isfinite(z).all()):
        raise ValueError('Invalid directed loss inputs')
    difference = signed[:, None]*(z[first]-z[second])
    positive = np.maximum(difference@theta[:-1], 0)
    value = float(np.sum(w*positive**2))
    gradient = np.r_[2*difference.T@(w*positive), 0.]
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise ValueError('Nonfinite directed loss')
    return value, gradient


def directed_objective(parameters, z, signed, weights, sources, pairs, *, ridge, temperature, strength):
    value, gradient, losses = source_objective(parameters, z, signed, weights, sources,
        ridge=ridge, temperature=temperature)
    penalty, derivative = directed_penalty(parameters, z, pairs)
    return value+strength*penalty, gradient+strength*derivative, losses


def fit_directed_source_risk(values, labels, weights, sources, pairs, *, strength, **kwargs):
    if not np.isfinite(strength) or strength < 0:
        raise ValueError('Invalid directed strength')
    y = np.asarray(labels)
    if not set(y.tolist()) <= {0, 1}:
        raise ValueError('Invalid directed labels')
    y = y.astype(np.int64)
    base, baseline = fit_source_risk(values, y, weights, sources, **kwargs)
    x, w = map(np.asarray, (values, weights)); w = w/w.sum()
    z = (x-base.center)/base.scale
    initial = np.r_[base.weights, base.bias]
    penalty, _ = directed_penalty(initial, z, pairs)
    if strength == 0:
        return base, {**baseline, 'ordering_strength': 0, 'directed_score_penalty': penalty,
            'directed_edges': len(pairs.before), 'readout_method': 'L2 source logistic plus positive signed processing degradation'}
    ridge, temperature = kwargs.get('ridge', .01), kwargs.get('temperature', .1)
    tolerance, limit = kwargs.get('gradient_tolerance', 1e-5), kwargs.get('maximum_iterations', 500)
    def objective(theta):
        value, gradient, _ = directed_objective(theta, z, 2*y-1, w, sources, pairs,
            ridge=ridge, temperature=temperature, strength=strength)
        return value, gradient
    solution = minimize(objective, initial, jac=True, method='L-BFGS-B', options={
        'maxiter': limit, 'gtol': tolerance/10, 'ftol': 1e-14, 'maxls': 40, 'maxcor': 20})
    value, gradient, losses = directed_objective(solution.x, z, 2*y-1, w, sources, pairs,
        ridge=ridge, temperature=temperature, strength=strength)
    residual = float(np.max(np.abs(gradient)))
    if not solution.success or not np.isfinite(residual) or residual > tolerance or value > objective(initial)[0]+1e-10:
        raise ValueError(f'Directed optimizer failed: {solution.message};gradient={residual}')
    rule = StableRule(base.feature_names, base.center, base.scale, tuple(solution.x[:-1]),
        float(solution.x[-1]), strength, ridge, fit_manifest_sha256=base.fit_manifest_sha256)
    penalty, _ = directed_penalty(solution.x, z, pairs)
    return rule, {'objective': value, 'initial_objective': objective(initial)[0],
        'baseline_source_objective': baseline['objective'], 'fit_records': len(x), 'sources': len(losses),
        'temperature': temperature, 'ordering_strength': strength, 'maximum_absolute_gradient': residual,
        'optimizer_iterations': int(solution.nit), 'optimizer_message': str(solution.message),
        'directed_score_penalty': penalty, 'directed_edges': len(pairs.before),
        'maximum_source_risk': float(max(losses)), 'minimum_source_risk': float(min(losses)),
        'readout_method': 'L2 source logistic plus positive signed processing degradation'}
