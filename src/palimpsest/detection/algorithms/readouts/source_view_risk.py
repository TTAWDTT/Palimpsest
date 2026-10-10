"""Convex entropy-smoothed same-source view loss for frozen linear features.

Finite signed training views only; this is a MaxUp-inspired objective, not a
physical invariance theorem or a new maximum-augmentation loss principle.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from palimpsest.detection.algorithms.readouts.stable_rule import StableRule


def source_objective(parameters, z, signed, weights, sources, *, ridge, temperature):
    w, bias = parameters[:-1], parameters[-1]
    margins = signed * (z @ w + bias)
    losses = np.logaddexp(0, -margins)
    source_weights = np.bincount(sources, weights=weights)
    if temperature == 0:
        source_losses = np.bincount(sources, weights=weights * losses) / source_weights
        coefficients = weights * (-signed * expit(-margins))
    else:
        logits = losses / temperature + np.log(weights / source_weights[sources])
        maxima = np.full(len(source_weights), -np.inf)
        np.maximum.at(maxima, sources, logits)
        relative = np.exp(logits - maxima[sources])
        sums = np.bincount(sources, weights=relative)
        source_losses = temperature * (maxima + np.log(sums))
        coefficients = (
            source_weights[sources]
            * relative
            / sums[sources]
            * (-signed * expit(-margins))
        )
    value = source_weights @ source_losses + ridge * (w @ w)
    gradient = np.r_[z.T @ coefficients + 2 * ridge * w, coefficients.sum()]
    return float(value), gradient, source_losses


def fit_source_risk(
    values,
    labels,
    sample_weights,
    sources,
    *,
    feature_names,
    temperature=0.1,
    ridge=0.01,
    scale_floor=0.001,
    maximum_iterations=500,
    gradient_tolerance=1e-5,
    manifest_sha="",
):
    x, y = np.asarray(values, float), np.asarray(labels)
    weights, groups = np.asarray(sample_weights, float), np.asarray(sources)
    if (
        x.ndim != 2
        or not len(x)
        or x.shape[1] != len(feature_names)
        or y.shape != (len(x),)
        or set(y.tolist()) != {0, 1}
        or weights.shape != y.shape
        or groups.shape != y.shape
        or groups.dtype.kind not in "iu"
        or set(groups.tolist()) != set(range(int(groups.max()) + 1))
    ):
        raise ValueError("Invalid source-risk dimensions/labels/groups")
    if (
        not all(np.isfinite(v).all() for v in (x, weights))
        or min(weights) <= 0
        or not np.isfinite([temperature, ridge, scale_floor, gradient_tolerance]).all()
        or temperature < 0
        or min(ridge, scale_floor, gradient_tolerance) <= 0
        or maximum_iterations < 1
    ):
        raise ValueError("Invalid source-risk values")
    for group in range(int(groups.max()) + 1):
        if len(set(y[groups == group].tolist())) != 1:
            raise ValueError("Conflicting labels within source")
    weights = weights / weights.sum()
    center = np.sum(x * weights[:, None], axis=0)
    scale = np.maximum(
        np.sqrt(np.sum((x - center) ** 2 * weights[:, None], axis=0)), scale_floor
    )
    z = (x - center) / scale
    signed = 2 * y - 1

    def objective(theta):
        value, gradient, _ = source_objective(
            theta, z, signed, weights, groups, ridge=ridge, temperature=temperature
        )
        return value, gradient

    initial = np.zeros(x.shape[1] + 1)
    solution = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": maximum_iterations,
            "gtol": gradient_tolerance / 10,
            "ftol": 1e-14,
            "maxls": 40,
            "maxcor": 20,
        },
    )
    value, gradient, losses = source_objective(
        solution.x, z, signed, weights, groups, ridge=ridge, temperature=temperature
    )
    residual = float(np.max(np.abs(gradient)))
    if (
        not solution.success
        or residual > gradient_tolerance
        or value > objective(initial)[0] + 1e-10
    ):
        raise ValueError(
            f"Source-risk optimization failed: {solution.message};gradient={residual}"
        )
    rule = StableRule(
        tuple(feature_names),
        tuple(center),
        tuple(scale),
        tuple(solution.x[:-1]),
        float(solution.x[-1]),
        0.0,
        ridge,
        fit_manifest_sha256=manifest_sha,
    )
    return rule, {
        "objective": value,
        "initial_objective": objective(initial)[0],
        "sources": len(losses),
        "fit_records": len(x),
        "temperature": temperature,
        "maximum_source_risk": float(max(losses)),
        "minimum_source_risk": float(min(losses)),
        "maximum_absolute_gradient": residual,
        "optimizer_iterations": int(solution.nit),
        "optimizer_message": str(solution.message),
    }
