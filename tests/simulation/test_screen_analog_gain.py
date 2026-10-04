"""Separate collected photons from readout gain in the virtual sensor."""

from dataclasses import replace

import numpy as np
import pytest

from palimpsest.simulation.screen_capture import (
    ScreenCaptureParameters,
    render_screen_capture,
)


def test_gain_and_shutter_can_match_mean_but_not_photon_noise():
    frame = np.ones((80, 80, 3), dtype=np.float32)
    base = ScreenCaptureParameters(
        sensor_to_display=np.array([[0.25, 0, 5], [0, 0.25, 5], [0, 0, 1]]),
        fill_fraction=1,
        emitter_layout="co_spatial_rgb_control",
        display_gamma=1,
        exposure_electrons_per_unit=100,
        full_well_electrons=10000,
    )
    gain = render_screen_capture(
        frame, (64, 64), replace(base, sensor_analog_gain_relative=8), seed=11
    )
    shutter = render_screen_capture(
        frame, (64, 64), replace(base, exposure_electrons_per_unit=800), seed=11
    )
    np.testing.assert_allclose(gain.irradiance, shutter.irradiance)
    np.testing.assert_allclose(gain.noiseless_mosaic, shutter.noiseless_mosaic)
    # Same mean ADC level, but gain still draws shot noise from ~100 electrons
    # whereas the longer shutter draws it from ~800 electrons.
    noise_ratio = gain.raw_mosaic.std() / shutter.raw_mosaic.std()
    assert 2.5 < noise_ratio < 3.2  # ideal sqrt(8) = 2.828...


def test_gain_reduces_effective_adc_headroom_without_changing_irradiance():
    frame = np.ones((32, 32, 3), dtype=np.float32)
    base = ScreenCaptureParameters(
        sensor_to_display=np.array([[0.25, 0, 5], [0, 0.25, 5], [0, 0, 1]]),
        fill_fraction=1,
        emitter_layout="co_spatial_rgb_control",
        display_gamma=1,
        exposure_electrons_per_unit=5000,
        full_well_electrons=10000,
    )
    low = render_screen_capture(frame, (16, 16), base)
    high = render_screen_capture(
        frame, (16, 16), replace(base, sensor_analog_gain_relative=8)
    )
    np.testing.assert_allclose(low.irradiance, high.irradiance)
    np.testing.assert_allclose(low.noiseless_mosaic.mean(), 1 / 6, atol=1e-6)
    np.testing.assert_allclose(high.noiseless_mosaic.mean(), 1, atol=1e-6)


def test_gain_is_finite_and_at_least_unity():
    for invalid in (0.5, float("nan")):
        with pytest.raises(ValueError):
            ScreenCaptureParameters(
                sensor_to_display=np.eye(3), sensor_analog_gain_relative=invalid
            )
