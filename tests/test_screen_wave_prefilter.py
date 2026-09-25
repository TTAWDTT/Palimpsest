from dataclasses import replace

import numpy as np
import pytest

from origin_simulation.screen_capture import ScreenCaptureParameters, render_screen_capture


def _camera(*, focus_m: float = 1.0, psf: str = "wave") -> ScreenCaptureParameters:
    return ScreenCaptureParameters(
        sensor_to_display=np.array([[.93, 0, 9.137], [0, .93, 8.219], [0, 0, 1]]),
        fill_fraction=.85, emitter_layout="vertical_rgb", display_gamma=1,
        lens_focal_length_mm=4, aperture_f_number=2,
        screen_distance_m=.5, focus_distance_m=focus_m,
        sensor_pixel_pitch_um=4, defocus_psf_model=psf)


def test_wave_prefilter_agrees_with_pre_sensor_fine_in_tested_regime():
    frame = np.random.default_rng(89).uniform(.1, .9, (56, 56, 3)).astype(np.float32)
    params = _camera()
    fine = render_screen_capture(frame, (20, 20), params).irradiance
    fast = render_screen_capture(frame, (20, 20), params,
                                 spatial_method="wave_prefilter").irradiance
    residual = np.abs(fine[2:-2, 2:-2] - fast[2:-2, 2:-2])
    assert float(residual.mean()) < .00025
    assert float(np.quantile(residual, .99)) < .0008


def test_wave_prefilter_rejects_unprobed_optical_or_geometry_settings():
    frame = np.ones((36, 36, 3), dtype=np.float32) * .5
    params = _camera()
    with pytest.raises(ValueError, match="CoC radius"):
        render_screen_capture(frame, (8, 8), _camera(focus_m=.5),
                              spatial_method="wave_prefilter")
    with pytest.raises(ValueError, match="defocus_psf_model"):
        render_screen_capture(frame, (8, 8), _camera(psf="geometric_airy"),
                              spatial_method="wave_prefilter")
    H = params.sensor_to_display.copy()
    H[0, 1] = .1
    with pytest.raises(ValueError, match="axis-aligned"):
        render_screen_capture(frame, (8, 8), replace(params, sensor_to_display=H),
                              spatial_method="wave_prefilter")
    with pytest.raises(ValueError, match="Gaussian"):
        render_screen_capture(frame, (8, 8),
                              replace(params, optical_blur_sigma_sensor_pixels=.2),
                              spatial_method="wave_prefilter")
    with pytest.raises(ValueError, match="Airy radius"):
        render_screen_capture(frame, (8, 8),
                              replace(params, rgb_effective_wavelengths_nm=(900, 800, 700)),
                              spatial_method="wave_prefilter")
    with pytest.raises(ValueError, match="does not use fine-grid"):
        render_screen_capture(frame, (8, 8), params,
                              spatial_method="wave_prefilter",
                              samples_per_sensor_pixel=32)
