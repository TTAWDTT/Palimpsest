"""Restricted monochrome print-to-flatbed-scan forward model.

This is a structural research prototype. It has either an ordered square-spot
RIP for gray input or a direct binary printer raster for calibrated-code
experiments. Deposition, Gaussian paper/scanner PSFs and scalar tone curves
are *hypotheses*.
No real printer/scanner has yet supplied all parameters. In particular, a
strongest FFT peak measured from a scan must not be treated as proof of the
RIP's fundamental line-screen frequency.

Physical coordinates are inches. The input is a known digital grayscale
image with an explicit intended digital PPI; no implicit resize or JPEG step
is included. The scanner pixel aperture is approximated by an equal-second-
moment Gaussian prior to sampling, not a measured scanner PSF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates


UM_PER_INCH = 25_400.0


@dataclass(frozen=True)
class PrintScanParameters:
    raster_mode: str = "halftone"
    digital_ppi: float = 800.0
    render_ppi: float = 2400.0
    scan_ppi: float = 800.0
    screen_lpi: float = 141.421356
    screen_angle_degrees: float = 45.0
    screen_phase_u: float = 0.0
    screen_phase_v: float = 0.0
    print_gamma: float = 1.0
    mechanical_dot_gain_um: float = 0.0
    deposition_blur_um: float = 0.0
    paper_scatter_sigma_um: float = 0.0
    scanner_optical_sigma_um: float = 0.0
    scanner_rotation_degrees: float = 0.0
    paper_reflectance: float = 0.95
    ink_reflectance: float = 0.05
    scanner_gamma: float = 1.0
    scanner_noise_std_linear: float = 0.0
    random_seed: int = 0
    max_render_pixels: int = 8_000_000

    def __post_init__(self) -> None:
        if self.raster_mode not in ("halftone", "binary_direct"):
            raise ValueError("raster_mode must be halftone or binary_direct")
        positive = (self.digital_ppi, self.render_ppi, self.scan_ppi,
                    self.screen_lpi, self.print_gamma, self.scanner_gamma)
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError("physical resolutions, screen frequency and gamma must be finite and positive")
        if ((self.raster_mode == "halftone" and self.render_ppi < 8 * self.screen_lpi) or
                self.render_ppi < self.scan_ppi or self.render_ppi < self.digital_ppi):
            raise ValueError("render_ppi undersamples the digital input, halftone lattice or scanner")
        finite = (self.screen_angle_degrees, self.screen_phase_u, self.screen_phase_v,
                  self.mechanical_dot_gain_um, self.deposition_blur_um,
                  self.paper_scatter_sigma_um, self.scanner_optical_sigma_um,
                  self.scanner_rotation_degrees, self.paper_reflectance,
                  self.ink_reflectance, self.scanner_noise_std_linear)
        if not np.isfinite(finite).all():
            raise ValueError("print/scan parameters must be finite")
        if min(self.deposition_blur_um, self.paper_scatter_sigma_um,
               self.scanner_optical_sigma_um, self.scanner_noise_std_linear) < 0:
            raise ValueError("PSF widths and noise must be nonnegative")
        if not 0 <= self.ink_reflectance < self.paper_reflectance <= 1:
            raise ValueError("ink reflectance must be lower than paper reflectance")
        if self.max_render_pixels <= 0:
            raise ValueError("max_render_pixels must be positive")
        if (self.mechanical_dot_gain_um != 0 and
                abs(self.mechanical_dot_gain_um) * self.render_ppi / UM_PER_INCH < 1):
            raise ValueError("mechanical dot gain is below one render cell; increase render_ppi")


@dataclass(frozen=True)
class PrintScanResult:
    halftone_ink: np.ndarray
    paper_reflectance: np.ndarray
    pre_sample_reflectance: np.ndarray
    scanner_linear: np.ndarray
    scanner_output: np.ndarray


def _as_gray(digital: np.ndarray) -> np.ndarray:
    data = np.asarray(digital)
    if data.ndim != 2 or min(data.shape) <= 0:
        raise ValueError("input must be a nonempty 2D grayscale image")
    if data.dtype == np.uint8:
        return data.astype(np.float32) / 255.0
    data = data.astype(np.float32)
    if not np.isfinite(data).all() or data.min() < 0 or data.max() > 1:
        raise ValueError("float input must be finite and within [0,1]")
    return data


def _gaussian_physical(image: np.ndarray, width_um: float, render_ppi: float) -> np.ndarray:
    sigma = width_um * render_ppi / UM_PER_INCH
    if sigma <= 0:
        return image
    return gaussian_filter(image, sigma=sigma, mode="reflect").astype(np.float32)


def simulate_print_scan(digital: np.ndarray, parameters: PrintScanParameters) -> PrintScanResult:
    """Render one grayscale digital patch through a physical-coordinate chain."""
    gray = _as_gray(digital)
    if parameters.raster_mode == "binary_direct" and not np.all((gray == 0) | (gray == 1)):
        raise ValueError("binary_direct mode requires a strict 0/1 digital raster")
    height, width = gray.shape
    inches_w, inches_h = width / parameters.digital_ppi, height / parameters.digital_ppi
    fine_w = round(inches_w * parameters.render_ppi)
    fine_h = round(inches_h * parameters.render_ppi)
    if fine_w * fine_h > parameters.max_render_pixels:
        raise ValueError(f"render grid {fine_w}x{fine_h} exceeds configured limit")
    if fine_w < 2 or fine_h < 2:
        raise ValueError("physical source patch is too small for render grid")

    x_inch = (np.arange(fine_w, dtype=np.float32) + 0.5) / parameters.render_ppi
    y_inch = (np.arange(fine_h, dtype=np.float32) + 0.5) / parameters.render_ppi
    xx, yy = np.meshgrid(x_inch, y_inch)
    source = map_coordinates(
        gray, [yy * parameters.digital_ppi - .5, xx * parameters.digital_ppi - .5],
        order=0 if parameters.raster_mode == "binary_direct" else 1,
        mode="nearest",
    )
    if parameters.raster_mode == "binary_direct":
        ink = source < .5
        del xx, yy, source
    else:
        target_ink_fraction = np.power(np.clip(1 - source, 0, 1), parameters.print_gamma)
        angle = np.deg2rad(parameters.screen_angle_degrees)
        uu = parameters.screen_lpi * (np.cos(angle) * xx + np.sin(angle) * yy) + parameters.screen_phase_u
        vv = parameters.screen_lpi * (-np.sin(angle) * xx + np.cos(angle) * yy) + parameters.screen_phase_v
        phase_u, phase_v = np.mod(uu, 1), np.mod(vv, 1)
        spot_distance = np.maximum(np.abs(phase_u - .5), np.abs(phase_v - .5))
        ink = (target_ink_fraction >= 1) | (spot_distance < .5 * np.sqrt(target_ink_fraction))
        del xx, yy, uu, vv, phase_u, phase_v, spot_distance, source, target_ink_fraction

    gain_px = abs(parameters.mechanical_dot_gain_um) * parameters.render_ppi / UM_PER_INCH
    if gain_px > 0:
        if parameters.mechanical_dot_gain_um > 0:
            ink = ink | (distance_transform_edt(~ink) <= gain_px)
        else:
            ink = ink & (distance_transform_edt(ink) > gain_px)
    deposited = _gaussian_physical(ink.astype(np.float32), parameters.deposition_blur_um,
                                   parameters.render_ppi)
    paper = (parameters.paper_reflectance -
             (parameters.paper_reflectance - parameters.ink_reflectance) * deposited).astype(np.float32)
    paper = _gaussian_physical(paper, parameters.paper_scatter_sigma_um, parameters.render_ppi)

    aperture_sigma_um = UM_PER_INCH / (parameters.scan_ppi * np.sqrt(12.0))
    effective_sigma_um = float(np.hypot(parameters.scanner_optical_sigma_um, aperture_sigma_um))
    before_scan = _gaussian_physical(paper, effective_sigma_um, parameters.render_ppi)

    scan_w = round(inches_w * parameters.scan_ppi)
    scan_h = round(inches_h * parameters.scan_ppi)
    scan_x = (np.arange(scan_w, dtype=np.float32) + .5) / parameters.scan_ppi
    scan_y = (np.arange(scan_h, dtype=np.float32) + .5) / parameters.scan_ppi
    sx, sy = np.meshgrid(scan_x, scan_y)
    scan_angle = np.deg2rad(parameters.scanner_rotation_degrees)
    cx, cy = inches_w / 2, inches_h / 2
    physical_x = cx + np.cos(scan_angle) * (sx - cx) - np.sin(scan_angle) * (sy - cy)
    physical_y = cy + np.sin(scan_angle) * (sx - cx) + np.cos(scan_angle) * (sy - cy)
    linear = map_coordinates(
        before_scan,
        [physical_y * parameters.render_ppi - .5, physical_x * parameters.render_ppi - .5],
        order=1, mode="constant", cval=parameters.paper_reflectance,
    ).astype(np.float32)
    if parameters.scanner_noise_std_linear:
        rng = np.random.default_rng(parameters.random_seed)
        linear = linear + rng.normal(0, parameters.scanner_noise_std_linear, linear.shape).astype(np.float32)
    output = np.power(np.clip(linear, 0, 1), 1 / parameters.scanner_gamma).astype(np.float32)
    return PrintScanResult(ink, paper, before_scan, linear, output)
