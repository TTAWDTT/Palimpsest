"""Relative aperture throughput couples optics, exposure, and shot noise."""

import numpy as np
import pytest

from origin_simulation.screen_capture import ScreenCaptureParameters, render_screen_capture


def _capture(f_number: float, exposure_time_s: float, *, reference: float | None):
    frame = np.ones((64, 64, 3), dtype=np.float32) * .5
    params = ScreenCaptureParameters(
        sensor_to_display=np.array([[1/1.3272, 0, 9.137],
                                    [0, 1/1.3272, 8.219], [0, 0, 1]]),
        fill_fraction=.85, emitter_layout="co_spatial_rgb_control",
        display_gamma=1, lens_focal_length_mm=30,
        aperture_f_number=f_number, screen_distance_m=1.45,
        focus_distance_m=1.45, sensor_pixel_pitch_um=4.30652,
        defocus_psf_model="wave", throughput_reference_f_number=reference,
        electron_rate_per_unit_s=100_000, exposure_time_s=exposure_time_s,
        full_well_electrons=10_000)
    return render_screen_capture(frame, (16, 16), params,
                                 spatial_method="wave_prefilter")


def test_relative_pupil_area_scales_irradiance_and_expected_electrons():
    wide = _capture(11, .01, reference=11)
    narrow = _capture(13, .01, reference=11)
    ratio = (11 / 13) ** 2
    # A flat screen suppresses scene-dependent PSF differences; the pupil
    # ratio then changes both linear irradiance and expected photoelectrons.
    np.testing.assert_allclose(narrow.irradiance.mean() / wide.irradiance.mean(),
                               ratio, rtol=2e-3)
    np.testing.assert_allclose(narrow.noiseless_mosaic.mean() /
                               wide.noiseless_mosaic.mean(), ratio, rtol=2e-3)
    compensated = _capture(13, .01 / ratio, reference=11)
    np.testing.assert_allclose(compensated.noiseless_mosaic.mean(),
                               wide.noiseless_mosaic.mean(), rtol=2e-3)
    # Shutter compensation changes integrated photons, not the optical PSF.
    np.testing.assert_allclose(compensated.irradiance, narrow.irradiance)


def test_relative_aperture_throughput_requires_valid_reference_and_lens():
    with pytest.raises(ValueError, match="relative aperture throughput"):
        ScreenCaptureParameters(sensor_to_display=np.eye(3),
                                throughput_reference_f_number=11)
    with pytest.raises(ValueError, match="relative aperture throughput"):
        ScreenCaptureParameters(
            sensor_to_display=np.eye(3), lens_focal_length_mm=30,
            aperture_f_number=13, screen_distance_m=1.45,
            focus_distance_m=1.45, sensor_pixel_pitch_um=4.30652,
            throughput_reference_f_number=-11)
