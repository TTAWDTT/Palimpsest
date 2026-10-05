import numpy as np
import pytest

from palimpsest.evaluation.image_pairs import image_metrics


def test_coarse_metrics_separate_brightness_change_from_structure():
    gray = np.linspace(0, 0.5, 64).reshape(8, 8).astype(np.float32)
    original = np.repeat(gray[..., None], 3, axis=2)
    measures = image_metrics(original, original + 0.25)
    assert measures["coarse_luminance_correlation"] == pytest.approx(1)
    assert measures["mean_rgb_absolute_change"] == pytest.approx(0.25)
    assert measures["coarse_gradient_energy_ratio"] == pytest.approx(1)


def test_flat_reference_has_finite_diagnostic_values():
    reference = np.ones((8, 8, 3), dtype=np.float32)
    measures = image_metrics(reference, reference * 0.5)
    assert measures["coarse_luminance_correlation"] == 0
    assert measures["mean_rgb_absolute_change"] == pytest.approx(0.5)
    assert measures["coarse_gradient_energy_ratio"] == 0
