"""Keep source-view classification risk while penalizing score variance.

Known convex consistency regularization, adapted to matched views. Penalizing
average fit variance cannot promise small accuracy decline on unseen channels.
"""

import numpy as np
from scipy.optimize import minimize

from palimpsest.detection.algorithms.readouts.stable_rule import StableRule
from palimpsest.detection.algorithms.readouts.source_view_risk import (
    fit_source_risk,
    source_objective,
)


def source_variance_gram(z, weights, groups):
    z, w, g = np.asarray(z, float), np.asarray(weights, float), np.asarray(groups)
    if (
        z.ndim != 2
        or not len(z)
        or w.shape != (len(z),)
        or g.shape != (len(z),)
        or g.dtype.kind not in "iu"
        or not np.isfinite(z).all()
        or not np.isfinite(w).all()
        or np.min(w) <= 0
        or set(g.tolist()) != set(range(int(g.max()) + 1))
    ):
        raise ValueError("Invalid consistency views")
    w = w / w.sum()
    mass = np.bincount(g, weights=w)
    means = np.zeros((len(mass), z.shape[1]))
    np.add.at(means, g, z * w[:, None])
    means /= mass[:, None]
    delta = z - means[g]
    return delta.T @ (w[:, None] * delta)


def consistent_objective(
    parameters, z, signed, weights, sources, gram, *, ridge, temperature, strength
):
    value, gradient, losses = source_objective(
        parameters, z, signed, weights, sources, ridge=ridge, temperature=temperature
    )
    direction = gram @ parameters[:-1]
    gradient[:-1] += 2 * strength * direction
    return value + strength * float(parameters[:-1] @ direction), gradient, losses


def fit_consistent_source_risk(
    values, labels, weights, sources, *, strength, consistency_groups=None, **kwargs
):
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("Invalid consistency strength")
    # Existing fitter performs the dimension,finite,label,group and optimizer
    # gates. Reuse its moments and exact zero-strength reference unchanged.
    base, baseline = fit_source_risk(values, labels, weights, sources, **kwargs)
    x = np.asarray(values, float)
    y = np.asarray(labels)
    w = np.asarray(weights, float)
    w = w / w.sum()
    s = np.asarray(sources)
    g = s if consistency_groups is None else np.asarray(consistency_groups)
    z = (x - base.center) / base.scale
    gram = source_variance_gram(z, w, g)
    if strength == 0:
        vector = np.asarray(base.weights)
        return base, {
            **baseline,
            "consistency_strength": 0,
            "source_score_variance": float(vector @ gram @ vector),
        }
    ridge = kwargs.get("ridge", 0.01)
    temperature = kwargs.get("temperature", 0.1)
    tolerance = kwargs.get("gradient_tolerance", 1e-5)
    limit = kwargs.get("maximum_iterations", 500)

    def objective(theta):
        value, gradient, _ = consistent_objective(
            theta,
            z,
            2 * y - 1,
            w,
            s,
            gram,
            ridge=ridge,
            temperature=temperature,
            strength=strength,
        )
        return value, gradient

    initial = np.r_[base.weights, base.bias]
    solution = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": limit,
            "gtol": tolerance / 10,
            "ftol": 1e-14,
            "maxls": 40,
            "maxcor": 20,
        },
    )
    value, gradient, losses = consistent_objective(
        solution.x,
        z,
        2 * y - 1,
        w,
        s,
        gram,
        ridge=ridge,
        temperature=temperature,
        strength=strength,
    )
    residual = float(np.max(np.abs(gradient)))
    if (
        not solution.success
        or residual > tolerance
        or value > objective(initial)[0] + 1e-10
    ):
        raise ValueError(
            f"Consistency optimizer failed: {solution.message};gradient={residual}"
        )
    rule = StableRule(
        base.feature_names,
        base.center,
        base.scale,
        tuple(solution.x[:-1]),
        float(solution.x[-1]),
        strength,
        ridge,
        fit_manifest_sha256=base.fit_manifest_sha256,
    )
    return rule, {
        "objective": value,
        "initial_objective": objective(initial)[0],
        "baseline_source_objective": baseline["objective"],
        "fit_records": len(x),
        "sources": len(losses),
        "temperature": temperature,
        "consistency_strength": strength,
        "maximum_absolute_gradient": residual,
        "optimizer_iterations": int(solution.nit),
        "optimizer_message": str(solution.message),
        "maximum_source_risk": float(max(losses)),
        "minimum_source_risk": float(min(losses)),
        "source_score_variance": float(solution.x[:-1] @ gram @ solution.x[:-1]),
    }
