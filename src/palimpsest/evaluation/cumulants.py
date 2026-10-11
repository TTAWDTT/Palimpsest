"""Explicit fourth-order moments for mechanism controls, not source scores.

Conjugation is explicit in the supplied four columns. Population probabilities
and iid sample averages have separate APIs; neither is inferred from one image.
"""

import numpy as np


def _four_columns(values) -> np.ndarray:
    data = np.asarray(values, dtype=np.complex128)
    if data.ndim != 2 or data.shape[1] != 4 or not len(data):
        raise ValueError("values must have nonempty shape (states_or_samples, 4)")
    if not np.isfinite(data).all():
        raise ValueError("values must be finite")
    return data


def _centered_terms(data: np.ndarray, weights: np.ndarray) -> tuple[complex, complex]:
    centered = data - weights @ data
    fourth = weights @ np.prod(centered, axis=1)
    pairs = sum(
        (weights @ (centered[:, i] * centered[:, j]))
        * (weights @ (centered[:, k] * centered[:, l]))
        for i, j, k, l in ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))
    )
    return complex(fourth), complex(pairs)


def population_cumulant4(values, probabilities) -> complex:
    """Cumulant of an explicitly supplied finite joint population."""
    data = _four_columns(values)
    weights = np.asarray(probabilities, dtype=float)
    if (
        weights.shape != (len(data),)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("probabilities must be nonnegative, finite and sum to one")
    fourth, pairs = _centered_terms(data, weights)
    return fourth - pairs


def unbiased_cumulant4(values) -> complex:
    """Fourth joint k-statistic, unbiased only for iid vectors with finite moments.

    Correlated windows and a ratio of these estimates are not thereby unbiased.
    """
    data = _four_columns(values)
    n = len(data)
    if n < 4:
        raise ValueError("at least four iid sample vectors are required")
    fourth, pairs = _centered_terms(data, np.full(n, 1 / n))
    return n**2 * ((n + 1) * fourth - (n - 1) * pairs) / ((n - 1) * (n - 2) * (n - 3))


def normalized_cumulant_magnitude(cumulant, marginal_cumulants, *, floor=0.0):
    """Normalize by four auto-cumulants; return None for an explicit degeneracy.

    Not a correlation coefficient: values can exceed one. A caller-specified
    floor is a numerical policy, not a statistical confidence or noise bound.
    """
    value = complex(cumulant)
    marginals = np.asarray(marginal_cumulants, dtype=np.complex128)
    if (
        marginals.shape != (4,)
        or not np.isfinite(marginals).all()
        or not np.isfinite(value)
        or not np.isfinite(floor)
        or floor < 0
    ):
        raise ValueError("supply four finite marginals and a nonnegative floor")
    sizes = np.abs(marginals)
    if (sizes <= floor).any():
        return None
    # Logs avoid overflow/underflow from multiplying four marginal magnitudes.
    denominator = np.exp(np.mean(np.log(sizes)))
    return float(abs(value) / denominator)
