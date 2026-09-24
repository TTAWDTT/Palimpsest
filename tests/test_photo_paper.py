"""Mechanistic checks for the continuous-tone photo-paper hypothesis."""

from dataclasses import replace

import numpy as np
import pytest

from origin_simulation.photo_paper import PhotoPaperParameters, simulate_photo_paper_surface
from origin_simulation.print_camera import PrintCameraParameters, photograph_print_surface


def _params(**kwargs: object) -> PhotoPaperParameters:
    return PhotoPaperParameters(digital_ppi=200, render_ppi=800,
                                exposure_spot_sigma_um=0, dye_spread_sigma_um=0,
                                grain_density_std=0, **kwargs)


def test_density_is_monotonic_and_bounded_for_grayscale_ramp() -> None:
    ramp = np.broadcast_to(np.arange(64, dtype=np.uint8)[None, :, None] * 4,
                           (32, 64, 3)).copy()
    surface = simulate_photo_paper_surface(ramp, _params())
    line = surface.paper_reflectance_rgb[64, :, 0]
    assert np.all(np.diff(line) >= -1e-5)
    assert line[0] < .2 < line[-1]
    assert surface.paper_ppi == 800
    assert np.isclose(surface.physical_size_inches[0], 64/200)


def test_grain_is_fixed_on_paper_before_independent_camera_noise() -> None:
    gray = np.full((64, 64, 3), 128, dtype=np.uint8)
    p = PhotoPaperParameters(digital_ppi=200, render_ppi=800,
                             grain_density_std=.08, random_seed=9)
    a = simulate_photo_paper_surface(gray, p)
    b = simulate_photo_paper_surface(gray, p)
    c = simulate_photo_paper_surface(gray, replace(p, random_seed=10))
    np.testing.assert_array_equal(a.paper_reflectance_rgb, b.paper_reflectance_rgb)
    assert not np.array_equal(a.paper_reflectance_rgb, c.paper_reflectance_rgb)
    cam = PrintCameraParameters(sensor_to_paper_inches=np.diag((1/200, 1/200, 1)),
                                paper_ppi=a.paper_ppi, exposure_electrons_per_unit=2000)
    captured = photograph_print_surface(a.paper_reflectance_rgb, (32,32), cam)
    assert captured.srgb.shape == (32,32,3)


def test_exposure_spot_reduces_fine_detail_before_paper_development() -> None:
    stripes = (np.arange(64) % 2 * 255).astype(np.uint8)
    image = np.broadcast_to(stripes[None, :, None], (32, 64, 3)).copy()
    sharp = simulate_photo_paper_surface(image, _params())
    blurred = simulate_photo_paper_surface(image, replace(_params(), exposure_spot_sigma_um=120))
    sharp_variation = np.std(sharp.paper_reflectance_rgb[64, :, 0])
    blurred_variation = np.std(blurred.paper_reflectance_rgb[64, :, 0])
    assert blurred_variation < sharp_variation / 5


def test_invalid_sampling_and_density_rejected() -> None:
    with pytest.raises(ValueError):
        PhotoPaperParameters(digital_ppi=800, render_ppi=800)
    with pytest.raises(ValueError):
        PhotoPaperParameters(max_density_cmy=(1, -1, 1))
