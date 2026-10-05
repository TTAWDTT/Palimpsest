"""Coarse paired-image diagnostics in the historical RR protocol.

Inputs are already registered/resized RGB arrays; this is not registration.
"""

import numpy as np


def image_metrics(original: np.ndarray, processed: np.ndarray) -> dict[str, float]:
    reference = original.mean(axis=2)
    observed = processed.mean(axis=2)
    reference_centered = reference - reference.mean()
    observed_centered = observed - observed.mean()
    denominator = float(
        np.linalg.norm(reference_centered) * np.linalg.norm(observed_centered)
    )
    correlation = (
        float(np.sum(reference_centered * observed_centered) / denominator)
        if denominator
        else 0.0
    )
    correlation = min(1.0, max(-1.0, correlation))

    def gradient_energy(gray: np.ndarray) -> float:
        return float(
            (
                np.abs(np.diff(gray, axis=0)).mean()
                + np.abs(np.diff(gray, axis=1)).mean()
            )
            / 2
        )

    original_gradient = gradient_energy(reference)
    return {
        "coarse_luminance_correlation": correlation,
        "mean_rgb_absolute_change": float(
            np.mean(np.abs(original.mean(axis=(0, 1)) - processed.mean(axis=(0, 1))))
        ),
        "coarse_gradient_energy_ratio": gradient_energy(observed)
        / max(original_gradient, 1e-8),
    }
