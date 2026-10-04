"""Mechanism checks for the uncalibrated color printing and scanning chain."""

from dataclasses import replace

import numpy as np
import pytest

from palimpsest.simulation.color_print_scan import (
    ColorPrintScanParameters,
    simulate_color_print_scan,
)


def _parameters() -> ColorPrintScanParameters:
    return ColorPrintScanParameters(
        digital_ppi=300, render_ppi=1200, scan_ppi=600, screen_lpi=100
    )


def test_white_black_and_primary_separations() -> None:
    params = _parameters()
    colors = np.array(
        [[[255, 255, 255], [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]],
        dtype=np.uint8,
    )
    # Use broad constant fields to avoid interpolation across their borders.
    for value, expected in zip(colors[0], [(), (3,), (1, 2), (0, 2), (0, 1)]):
        patch = np.tile(value, (32, 32, 1))
        result = simulate_color_print_scan(patch, params)
        observed = tuple(np.where(result.ink_masks_cmyk[32, 32])[0])
        assert observed == expected
    white = simulate_color_print_scan(np.full((32, 32, 3), 255, np.uint8), params)
    np.testing.assert_allclose(
        white.scanner_linear_rgb,
        np.broadcast_to(params.paper_reflectance_rgb, white.scanner_linear_rgb.shape),
        atol=1e-6,
    )


def test_rgb_channel_coupling_and_registration_are_causal() -> None:
    params = _parameters()
    cyan_input = np.full((64, 64, 3), (150, 255, 255), dtype=np.uint8)
    base = simulate_color_print_scan(cyan_input, params)
    assert (
        base.scanner_linear_rgb[:, :, 0].mean()
        < base.scanner_linear_rgb[:, :, 1].mean()
    )
    assert (
        base.scanner_linear_rgb[:, :, 0].mean()
        < base.scanner_linear_rgb[:, :, 2].mean()
    )
    shifted = simulate_color_print_scan(
        cyan_input,
        replace(
            params, registration_um=((40.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
        ),
    )
    assert not np.array_equal(
        base.ink_masks_cmyk[:, :, 0], shifted.ink_masks_cmyk[:, :, 0]
    )
    np.testing.assert_array_equal(
        base.ink_masks_cmyk[:, :, 1:], shifted.ink_masks_cmyk[:, :, 1:]
    )


def test_frequency_is_in_physical_coordinates() -> None:
    params = replace(
        _parameters(), screen_angles_degrees=(0.0, 0.0, 0.0, 0.0), black_generation=0.0
    )
    gray = np.full((256, 256, 3), 180, dtype=np.uint8)
    for scan_ppi, expected in ((600, 100 / 600), (300, 100 / 300)):
        result = simulate_color_print_scan(gray, replace(params, scan_ppi=scan_ppi))
        image = result.scanner_output_rgb[:, :, 0]
        window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2((image - image.mean()) * window)))
        center = np.array(spectrum.shape) // 2
        spectrum[center[0] - 2 : center[0] + 3, center[1] - 2 : center[1] + 3] = 0
        peak = np.array(np.unravel_index(np.argmax(spectrum), spectrum.shape))
        axis_frequency = np.abs((peak - center) / np.array(image.shape))
        assert min(abs(axis_frequency - expected)) < 2 / min(image.shape)


def test_reproducible_noise_and_input_guards() -> None:
    params = replace(_parameters(), scanner_noise_std_linear=0.01, random_seed=19)
    source = np.full((16, 16, 3), 120, dtype=np.uint8)
    first = simulate_color_print_scan(source, params).scanner_output_rgb
    second = simulate_color_print_scan(source, params).scanner_output_rgb
    np.testing.assert_array_equal(first, second)
    third = simulate_color_print_scan(
        source, replace(params, random_seed=20)
    ).scanner_output_rgb
    assert not np.array_equal(first, third)
    with pytest.raises(ValueError):
        simulate_color_print_scan(np.zeros((16, 16), dtype=np.uint8), params)
    with pytest.raises(ValueError):
        ColorPrintScanParameters(ink_density_rgb=((-1.0, 0.0, 0.0),) * 4)
    with pytest.raises(ValueError):
        ColorPrintScanParameters(render_ppi=600, screen_lpi=100)


def test_sigma_suppresses_print_screen_before_sampling() -> None:
    source = np.full((64, 64, 3), 180, dtype=np.uint8)
    params = replace(
        _parameters(), screen_angles_degrees=(0.0, 0.0, 0.0, 0.0), black_generation=0.0
    )
    sharp = simulate_color_print_scan(source, params)
    soft = simulate_color_print_scan(source, replace(params, paper_scatter_sigma_um=60))
    assert (
        soft.pre_sample_reflectance_rgb[:, :, 0].std()
        < 0.3 * sharp.pre_sample_reflectance_rgb[:, :, 0].std()
    )
