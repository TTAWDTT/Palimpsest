import numpy as np
import pytest
from scipy.special import j1

from palimpsest.simulation.optical_psf import circular_pupil_defocus_psf
from palimpsest.simulation.screen_capture import (
    ScreenCaptureParameters,
    render_screen_capture,
)


ARGS = (4.0, 2.0, 0.5, 0.5, 4.0, 540.0, 12)


def test_wave_focus_limit_equals_circular_pupil_airy_intensity():
    kernel = circular_pupil_defocus_psf(*ARGS)
    radius = kernel.shape[0] // 2
    grid = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    focal, n, screen, focus, pitch_um, wavelength_nm, sampling = ARGS
    vf_mm = focal * (focus * 1000) / (focus * 1000 - focal)
    pupil_radius_mm = focal / (2 * n)
    wave_mm = wavelength_nm / 1_000_000
    q = (
        2
        * np.pi
        * pupil_radius_mm
        / (wave_mm * vf_mm)
        * (np.hypot(xx, yy) * pitch_um / 1000 / sampling)
    )
    amplitude = np.ones_like(q)
    np.divide(2 * j1(q), q, out=amplitude, where=q != 0)
    reference = amplitude**2
    reference /= reference.sum()
    np.testing.assert_allclose(kernel, reference, rtol=0, atol=2e-7)
    assert not kernel.flags.writeable


def test_wave_defocus_kernel_is_normalized_and_conserves_radial_symmetry():
    kernel = circular_pupil_defocus_psf(4, 2, 0.5, 0.75, 4, 540, 12)
    assert np.isfinite(kernel).all()
    assert kernel.min() >= 0
    assert float(kernel.sum()) == pytest.approx(1, abs=2e-7)
    np.testing.assert_allclose(kernel, kernel[::-1, :], atol=1e-7)
    np.testing.assert_allclose(kernel, kernel[:, ::-1], atol=1e-7)


def test_wave_mode_rejects_missing_lens_and_matches_tiled_render():
    transform = np.asarray([[0.93, 0, 9.13], [0, 0.93, 8.22], [0, 0, 1]])
    with pytest.raises(ValueError, match="complete thin-lens"):
        ScreenCaptureParameters(sensor_to_display=transform, defocus_psf_model="wave")
    params = ScreenCaptureParameters(
        sensor_to_display=transform,
        fill_fraction=0.75,
        emitter_layout="co_spatial_rgb_control",
        display_gamma=1,
        lens_focal_length_mm=4,
        aperture_f_number=2,
        screen_distance_m=0.5,
        focus_distance_m=0.75,
        sensor_pixel_pitch_um=4,
        defocus_psf_model="wave",
    )
    frame = (
        np.random.default_rng(22).uniform(0.2, 0.8, size=(42, 42, 3)).astype(np.float32)
    )
    whole = render_screen_capture(
        frame, (12, 12), params, tile_size_sensor_pixels=12
    ).irradiance
    tiled = render_screen_capture(
        frame, (12, 12), params, tile_size_sensor_pixels=6
    ).irradiance
    assert np.max(np.abs(whole - tiled)) < 2e-4
