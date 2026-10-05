"""Mechanism checks for photographing an explicitly sampled printed surface."""

from dataclasses import replace

import numpy as np
import pytest

from palimpsest.simulation.print_scan.color import (
    ColorPrintScanParameters,
    simulate_color_print_surface,
    scan_color_print_surface,
)
from palimpsest.simulation.print_capture.camera import (
    PrintCameraParameters,
    photograph_print_surface,
)


def _transform(step_inches: float) -> np.ndarray:
    return np.array([[step_inches, 0, 0], [0, step_inches, 0], [0, 0, 1]], dtype=float)


def _peak_x(image: np.ndarray) -> float:
    one_line = image.mean(axis=0)
    spectrum = np.abs(
        np.fft.rfft((one_line - one_line.mean()) * np.hanning(len(one_line)))
    )
    spectrum[:2] = 0
    return np.argmax(spectrum) / len(one_line)


def test_constant_diffuse_paper_and_point_light_distance() -> None:
    paper = np.full((256, 256, 3), 0.5, dtype=np.float32)
    params = PrintCameraParameters(
        sensor_to_paper_inches=_transform(1 / 400),
        paper_ppi=400,
        samples_per_sensor_pixel=4,
    )
    flat = photograph_print_surface(paper, (64, 64), params)
    np.testing.assert_allclose(flat.irradiance_rgb, 0.5, atol=1e-6)
    assert flat.srgb.shape == (64, 64, 3)
    light = replace(
        params,
        ambient_irradiance_rgb=(0.0, 0.0, 0.0),
        point_light_position_inches=(0.08, 0.08, 1.0),
        point_light_strength_rgb=(1.0, 1.0, 1.0),
    )
    near = photograph_print_surface(paper, (64, 64), light)
    farther = photograph_print_surface(
        paper, (64, 64), replace(light, point_light_position_inches=(0.08, 0.08, 2.0))
    )
    assert near.irradiance_rgb[32, 32, 0] > 3.9 * farther.irradiance_rgb[32, 32, 0]
    assert near.irradiance_rgb[32, 32, 0] > near.irradiance_rgb[0, 0, 0]


def test_paper_lattice_frequency_tracks_projection_scale() -> None:
    ppi = 400
    x = (np.arange(800) + 0.5) / ppi
    line = 0.5 + 0.4 * np.cos(2 * np.pi * 15 * x)
    paper = np.broadcast_to(line[None, :, None], (800, 800, 3)).astype(np.float32)
    base = PrintCameraParameters(sensor_to_paper_inches=_transform(0.01), paper_ppi=ppi)
    fine = photograph_print_surface(paper, (64, 64), base)
    coarse = photograph_print_surface(
        paper, (64, 64), replace(base, sensor_to_paper_inches=_transform(0.02))
    )
    assert abs(_peak_x(fine.irradiance_rgb[:, :, 0]) - 0.15) < 0.025
    assert abs(_peak_x(coarse.irradiance_rgb[:, :, 0]) - 0.30) < 0.025


def test_print_surface_can_feed_camera_without_scanner_output() -> None:
    rgb = np.full((32, 32, 3), (160, 210, 240), dtype=np.uint8)
    print_params = ColorPrintScanParameters(
        digital_ppi=300, render_ppi=1200, scan_ppi=600, screen_lpi=100
    )
    printed = simulate_color_print_surface(rgb, print_params)
    camera = photograph_print_surface(
        printed.paper_reflectance_rgb,
        (64, 64),
        PrintCameraParameters(
            sensor_to_paper_inches=_transform(1 / 600), paper_ppi=printed.paper_ppi
        ),
    )
    scanned = scan_color_print_surface(printed, print_params)
    assert camera.srgb.shape == (64, 64, 3)
    assert np.isfinite(camera.srgb).all()
    assert not np.array_equal(camera.srgb, scanned.scanner_output_rgb[:64, :64])


def test_sensor_noise_is_seeded_and_invalid_projection_rejected() -> None:
    paper = np.full((128, 128, 3), 0.7, dtype=np.float32)
    params = PrintCameraParameters(
        sensor_to_paper_inches=_transform(1 / 400),
        paper_ppi=400,
        exposure_electrons_per_unit=5000,
        read_noise_electrons=2,
        random_seed=7,
    )
    a = photograph_print_surface(paper, (32, 32), params)
    b = photograph_print_surface(paper, (32, 32), params)
    c = photograph_print_surface(paper, (32, 32), replace(params, random_seed=8))
    np.testing.assert_array_equal(a.raw_mosaic, b.raw_mosaic)
    assert not np.array_equal(a.raw_mosaic, c.raw_mosaic)
    with pytest.raises(ValueError):
        photograph_print_surface(
            paper, (32, 32), replace(params, sensor_to_paper_inches=_transform(1 / 90))
        )
    with pytest.raises(ValueError):
        PrintCameraParameters(
            sensor_to_paper_inches=_transform(1 / 400),
            paper_ppi=400,
            point_light_strength_rgb=(1, 1, 1),
        )
