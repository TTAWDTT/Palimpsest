"""Numerical and domain checks for the experimental fast perspective path."""

import numpy as np

from palimpsest.simulation.screen_capture.capture import (
    ScreenCaptureParameters,
    render_screen_capture,
)


def _pose(degrees: float, perspective: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    return np.asarray(
        [
            [0.77 * np.cos(theta), -0.68 * np.sin(theta), 11.23],
            [0.77 * np.sin(theta), 0.68 * np.cos(theta), 9.47],
            [perspective, -perspective / 2, 1],
        ],
        dtype=np.float64,
    )


def test_prefilter_preserves_irradiance_against_denser_quadrature_on_mild_tilt():
    display = np.random.default_rng(20260925).random((56, 56, 3), dtype=np.float32)
    setup = ScreenCaptureParameters(
        sensor_to_display=_pose(12, 0.0007),
        fill_fraction=0.85,
        optical_blur_sigma_sensor_pixels=0.8,
    )
    fast = render_screen_capture(display, (20, 20), setup, spatial_method="prefilter")
    reference = render_screen_capture(
        display,
        (20, 20),
        setup,
        samples_per_sensor_pixel=64,
        tile_size_sensor_pixels=20,
    )
    diff = np.abs(
        fast.emitter_band_irradiance[2:-2, 2:-2]
        - reference.emitter_band_irradiance[2:-2, 2:-2]
    )
    assert float(diff.mean()) < 0.001
    assert float(np.quantile(diff, 0.99)) < 0.003
    assert np.isfinite(fast.raw_mosaic).all()


def test_prefilter_rejects_known_numerical_failure_regimes():
    display = np.ones((56, 56, 3), dtype=np.float32)
    with np.testing.assert_raises_regex(ValueError, "sigma"):
        render_screen_capture(
            display,
            (20, 20),
            ScreenCaptureParameters(
                sensor_to_display=_pose(12, 0.0007), optical_blur_sigma_sensor_pixels=0
            ),
            spatial_method="prefilter",
        )
    with np.testing.assert_raises_regex(ValueError, "Jacobian"):
        render_screen_capture(
            display,
            (20, 20),
            ScreenCaptureParameters(
                sensor_to_display=_pose(35, 0.004), optical_blur_sigma_sensor_pixels=0.8
            ),
            spatial_method="prefilter",
        )
    with np.testing.assert_raises_regex(ValueError, "diffraction"):
        render_screen_capture(
            display,
            (20, 20),
            ScreenCaptureParameters(
                sensor_to_display=_pose(12, 0.0007),
                optical_blur_sigma_sensor_pixels=0.8,
                diffraction_f_number=8,
                sensor_pixel_pitch_um=1.4,
            ),
            spatial_method="prefilter",
        )
