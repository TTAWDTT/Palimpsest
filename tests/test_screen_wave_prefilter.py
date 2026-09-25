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


@pytest.mark.parametrize("f_number,distance_m", [(11, 1.4452), (13, 1.45)])
def test_in_focus_high_f_number_wave_prefilter_against_fine(f_number, distance_m):
    frame = np.empty((72, 72, 3), dtype=np.float32)
    frame[:] = .1 + .8 * (np.arange(72)[None, :, None] % 2)
    params = ScreenCaptureParameters(
        sensor_to_display=np.array([[1/1.3272, 0, 13.731],
                                    [0, 1/1.3272, 12.084], [0, 0, 1]]),
        fill_fraction=.85, emitter_layout="vertical_rgb", display_gamma=1,
        lens_focal_length_mm=30, aperture_f_number=f_number,
        screen_distance_m=distance_m, focus_distance_m=distance_m,
        sensor_pixel_pitch_um=4.30652, defocus_psf_model="wave")
    fine = render_screen_capture(frame, (48, 48), params).irradiance
    fast = render_screen_capture(frame, (48, 48), params,
                                 spatial_method="wave_prefilter").irradiance
    residual = np.abs(fine - fast)
    assert float(residual.mean()) < .0002
    assert float(np.quantile(residual, .99)) < .0005


def test_wave_fine_sensor_crop_does_not_change_irradiance():
    frame = np.random.default_rng(18).uniform(.1, .9, (72, 72, 3)).astype(np.float32)
    params = ScreenCaptureParameters(
        sensor_to_display=np.array([[1/1.3272, 0, 13.731],
                                    [0, 1/1.3272, 12.084], [0, 0, 1]]),
        fill_fraction=.85, emitter_layout="vertical_rgb", display_gamma=1,
        lens_focal_length_mm=30, aperture_f_number=13,
        screen_distance_m=1.45, focus_distance_m=1.45,
        sensor_pixel_pitch_um=4.30652, defocus_psf_model="wave")
    small = render_screen_capture(frame, (20, 20), params).irradiance
    large = render_screen_capture(frame, (32, 32), params).irradiance
    np.testing.assert_allclose(small, large[:20, :20], rtol=0, atol=1e-5)


@pytest.mark.parametrize("focus", ["moderate", "high_diffraction"])
def test_wave_prefilter_tiling_matches_full_raster(focus):
    frame = np.random.default_rng(218).uniform(.1, .9, (72, 72, 3)).astype(np.float32)
    if focus == "moderate":
        params = _camera()
    else:
        params = ScreenCaptureParameters(
            sensor_to_display=np.array([[1/1.3272, 0, 13.731],
                                        [0, 1/1.3272, 12.084], [0, 0, 1]]),
            fill_fraction=.85, emitter_layout="vertical_rgb", display_gamma=1,
            lens_focal_length_mm=30, aperture_f_number=13,
            screen_distance_m=1.45, focus_distance_m=1.45,
            sensor_pixel_pitch_um=4.30652, defocus_psf_model="wave")
    full = render_screen_capture(frame, (32, 32), params,
                                 spatial_method="wave_prefilter").irradiance
    tiled = render_screen_capture(frame, (32, 32), params,
                                  spatial_method="wave_prefilter",
                                  tile_size_sensor_pixels=8).irradiance
    np.testing.assert_allclose(tiled, full, rtol=0, atol=2e-6)


def test_wave_prefilter_auto_tiles_when_full_display_exceeds_memory_cap():
    big = np.random.default_rng(19).uniform(.1, .9, (300, 300, 3)).astype(np.float32)
    small = big[:72, :72].copy()
    params = ScreenCaptureParameters(
        sensor_to_display=np.array([[1/1.3272, 0, 13.731],
                                    [0, 1/1.3272, 12.084], [0, 0, 1]]),
        fill_fraction=.85, emitter_layout="vertical_rgb", display_gamma=1,
        lens_focal_length_mm=30, aperture_f_number=13,
        screen_distance_m=1.45, focus_distance_m=1.45,
        sensor_pixel_pitch_um=4.30652, defocus_psf_model="wave")
    reference = render_screen_capture(small, (32, 32), params,
                                      spatial_method="wave_prefilter").irradiance
    tiled = render_screen_capture(big, (32, 32), params,
                                  spatial_method="wave_prefilter").irradiance
    np.testing.assert_allclose(tiled, reference, rtol=0, atol=2e-6)
