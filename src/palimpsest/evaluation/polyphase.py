"""Population moments of explicit finite periodic linear operators.

These functions evaluate a supplied model, not an image's source. Covariance
is a population input; a single image does not provide it automatically.
"""

from dataclasses import dataclass

import numpy as np


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def periodic_upsampling_matrix(kernel, stride: int, latent_length: int) -> np.ndarray:
    """Insert zeros then filter: y[i] = sum h[k] u[(i-k) modulo size]."""
    stride = _positive_integer(stride, "stride")
    latent_length = _positive_integer(latent_length, "latent_length")
    h = np.asarray(kernel, dtype=np.float64)
    if h.ndim != 1 or h.size == 0 or not np.isfinite(h).all():
        raise ValueError("kernel must be a nonempty finite vector")
    size = stride * latent_length
    impulses = np.zeros((size, latent_length), dtype=np.float64)
    impulses[::stride] = np.eye(latent_length)
    return sum((weight * np.roll(impulses, k, axis=0) for k, weight in enumerate(h)),
               np.zeros_like(impulses))


@dataclass(frozen=True)
class PopulationMoments:
    mean: np.ndarray
    covariance: np.ndarray


def linear_population_moments(operator, input_mean, input_covariance) -> PopulationMoments:
    """Propagate supplied finite first and second moments through a matrix."""
    a = np.asarray(operator, dtype=np.float64)
    mean = np.asarray(input_mean, dtype=np.float64)
    covariance = np.asarray(input_covariance, dtype=np.float64)
    if a.ndim != 2 or min(a.shape, default=0) < 1:
        raise ValueError("operator must be a nonempty matrix")
    if mean.shape != (a.shape[1],) or covariance.shape != (a.shape[1], a.shape[1]):
        raise ValueError("input moments do not match operator dimensions")
    if not all(np.isfinite(x).all() for x in (a, mean, covariance)):
        raise ValueError("operator and moments must be finite")
    scale = max(1.0, float(np.max(np.abs(covariance))))
    if not np.allclose(covariance, covariance.T, rtol=0, atol=1e-12 * scale):
        raise ValueError("input covariance must be symmetric")
    if float(np.linalg.eigvalsh(covariance).min()) < -1e-12 * scale:
        raise ValueError("input covariance must be positive semidefinite")
    return PopulationMoments(a @ mean, a @ covariance @ a.T)


def phase_summary(moments: PopulationMoments, stride: int) -> dict:
    """Average periodic per-phase moments; residual compares a one-pixel shift."""
    stride = _positive_integer(stride, "stride")
    mean = np.asarray(moments.mean, dtype=np.float64)
    covariance = np.asarray(moments.covariance, dtype=np.float64)
    if mean.ndim != 1 or mean.size == 0 or mean.size % stride:
        raise ValueError("output size must be nonzero and divisible by stride")
    if covariance.shape != (mean.size, mean.size):
        raise ValueError("output covariance dimensions are inconsistent")
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("output moments must be finite")
    indices = np.arange(mean.size)
    next_indices = (indices + 1) % mean.size
    variances = np.diag(covariance)
    adjacent = covariance[indices, next_indices]
    differences = (variances + variances[next_indices] - 2 * adjacent
                   + (mean[next_indices] - mean) ** 2)
    grouped = lambda x: x.reshape(-1, stride).mean(axis=0).tolist()
    return {
        "means": grouped(mean),
        "variances": grouped(variances),
        "adjacent_covariances": grouped(adjacent),
        "difference_energies": grouped(differences),
        "mean_shift_residual": float(np.max(np.abs(mean - np.roll(mean, 1)))),
        "covariance_shift_residual": float(np.max(np.abs(
            covariance - np.roll(covariance, (1, 1), axis=(0, 1))))),
    }
