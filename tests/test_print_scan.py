"""Mechanism-level checks for the uncalibrated print-scan forward model."""

from dataclasses import replace

import numpy as np

from origin_simulation.print_scan import PrintScanParameters, simulate_print_scan


def _axis_frequency(image: np.ndarray) -> tuple[float, float]:
    height, width = image.shape
    window = np.hanning(height)[:, None] * np.hanning(width)[None, :]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2((image - image.mean()) * window)))
    center_y, center_x = height // 2, width // 2
    spectrum[center_y - 2:center_y + 3, center_x - 2:center_x + 3] = 0
    row, col = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    return abs(col - center_x) / width, abs(row - center_y) / height


def _peak_amplitude(image: np.ndarray, x_cycles_per_pixel: float) -> float:
    height, width = image.shape
    window = np.hanning(height)[:, None] * np.hanning(width)[None, :]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2((image - image.mean()) * window)))
    center_y, center_x = height // 2, width // 2
    offset = round(x_cycles_per_pixel * width)
    return float(spectrum[center_y - 1:center_y + 2, center_x + offset - 1:center_x + offset + 2].max())


def test_white_and_black_are_full_reflectance_extremes() -> None:
    params = PrintScanParameters(digital_ppi=800, render_ppi=1600, scan_ppi=800,
                                 screen_lpi=100)
    white = simulate_print_scan(np.ones((64, 64), dtype=np.float32), params)
    black = simulate_print_scan(np.zeros((64, 64), dtype=np.float32), params)
    assert not white.halftone_ink.any()
    assert black.halftone_ink.all()
    np.testing.assert_allclose(white.scanner_output, params.paper_reflectance, atol=1e-6)
    np.testing.assert_allclose(black.scanner_output, params.ink_reflectance, atol=1e-6)


def test_print_lattice_frequency_tracks_physical_lpi_and_scanner_ppi() -> None:
    gray = np.full((256, 256), .75, dtype=np.float32)
    params = PrintScanParameters(digital_ppi=800, render_ppi=2400, scan_ppi=800,
                                 screen_lpi=100, screen_angle_degrees=0)
    scan_800 = simulate_print_scan(gray, params).scanner_output
    scan_400 = simulate_print_scan(gray, replace(params, scan_ppi=400)).scanner_output
    fx800, fy800 = _axis_frequency(scan_800)
    fx400, fy400 = _axis_frequency(scan_400)
    assert min(abs(fx800 - .125), abs(fy800 - .125)) <= 1 / 256
    assert min(abs(fx400 - .25), abs(fy400 - .25)) <= 1 / 128


def test_paper_scatter_attenuates_pre_sample_lattice_and_dot_gain_darkens() -> None:
    gray = np.full((256, 256), .75, dtype=np.float32)
    params = PrintScanParameters(digital_ppi=800, render_ppi=2400, scan_ppi=800,
                                 screen_lpi=100, screen_angle_degrees=0)
    base = simulate_print_scan(gray, params)
    scattered = simulate_print_scan(gray, replace(params, paper_scatter_sigma_um=55))
    gained = simulate_print_scan(gray, replace(params, mechanical_dot_gain_um=30))
    before = _peak_amplitude(base.scanner_output, 100 / 800)
    after = _peak_amplitude(scattered.scanner_output, 100 / 800)
    assert after < .5 * before
    assert gained.scanner_output.mean() < base.scanner_output.mean()


def test_invalid_resolution_and_reflectance_are_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        PrintScanParameters(render_ppi=300, screen_lpi=100)
    with pytest.raises(ValueError):
        PrintScanParameters(ink_reflectance=.95, paper_reflectance=.9)
