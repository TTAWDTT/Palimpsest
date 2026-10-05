"""Uncalibrated RGB -> CMYK halftone -> reflective print -> flatbed scan.

Coordinates are inches. Optical densities are *effective RGB* densities,
not measured spectra or an ICC profile. This restricted model is intended for
mechanism tests and controlled-device calibration, not faithful reproduction
of a particular printer, stock or scanner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from palimpsest.simulation.shared.units import UM_PER_INCH
from palimpsest.simulation.shared.image import (
    blur_physical,
    rgb_unit_float,
    srgb_to_linear,
)


@dataclass(frozen=True)
class ColorPrintScanParameters:
    digital_ppi: float = 300.0
    render_ppi: float = 1200.0
    scan_ppi: float = 600.0
    screen_lpi: float = 100.0
    screen_angles_degrees: tuple[float, float, float, float] = (15.0, 75.0, 0.0, 45.0)
    screen_phases: tuple[tuple[float, float], ...] = ((0.0, 0.0),) * 4
    registration_um: tuple[tuple[float, float], ...] = ((0.0, 0.0),) * 4
    black_generation: float = 1.0
    print_gamma: float = 1.0
    ink_density_rgb: tuple[tuple[float, float, float], ...] = (
        (1.6, 0.12, 0.12),
        (0.12, 1.6, 0.12),
        (0.12, 0.12, 1.6),
        (1.5, 1.5, 1.5),
    )
    paper_reflectance_rgb: tuple[float, float, float] = (0.95, 0.95, 0.95)
    deposition_blur_um: float = 0.0
    paper_scatter_sigma_um: float = 0.0
    scanner_optical_sigma_um: float = 0.0
    scanner_rotation_degrees: float = 0.0
    scanner_noise_std_linear: float = 0.0
    scanner_gamma: float = 2.2
    random_seed: int = 0
    max_render_pixels: int = 2_000_000

    def __post_init__(self) -> None:
        positive = (
            self.digital_ppi,
            self.render_ppi,
            self.scan_ppi,
            self.screen_lpi,
            self.print_gamma,
            self.scanner_gamma,
        )
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError(
                "physical resolutions, screen frequency and gamma must be positive"
            )
        if self.render_ppi < max(self.digital_ppi, self.scan_ppi, 8 * self.screen_lpi):
            raise ValueError(
                "render_ppi undersamples input, scanner or color screen lattice"
            )
        if not 0 <= self.black_generation <= 1:
            raise ValueError("black_generation must be in [0,1]")
        if self.max_render_pixels <= 0:
            raise ValueError("max_render_pixels must be positive")
        for group in (
            self.screen_angles_degrees,
            self.screen_phases,
            self.registration_um,
            self.ink_density_rgb,
        ):
            if len(group) != 4 or not np.isfinite(np.asarray(group, dtype=float)).all():
                raise ValueError(
                    "each CMYK parameter group must have four finite members"
                )
        if any(len(pair) != 2 for pair in self.screen_phases + self.registration_um):
            raise ValueError("screen phases and registration must be 2D")
        if any(len(row) != 3 for row in self.ink_density_rgb):
            raise ValueError("effective ink densities must be RGB triples")
        density = np.asarray(self.ink_density_rgb)
        paper = np.asarray(self.paper_reflectance_rgb)
        if (
            np.any(density < 0)
            or paper.shape != (3,)
            or not np.isfinite(paper).all()
            or np.any(paper <= 0)
            or np.any(paper > 1)
        ):
            raise ValueError("densities must be nonnegative; paper RGB within (0,1]")
        optics = (
            self.deposition_blur_um,
            self.paper_scatter_sigma_um,
            self.scanner_optical_sigma_um,
            self.scanner_noise_std_linear,
        )
        if not np.isfinite(optics).all() or min(optics) < 0:
            raise ValueError("blur widths and noise must be finite and nonnegative")
        if not np.isfinite(self.scanner_rotation_degrees):
            raise ValueError("scanner rotation must be finite")


@dataclass(frozen=True)
class ColorPrintSurfaceResult:
    ink_masks_cmyk: np.ndarray
    paper_reflectance_rgb: np.ndarray
    paper_ppi: float
    physical_size_inches: tuple[float, float]


@dataclass(frozen=True)
class ColorPrintScanResult:
    ink_masks_cmyk: np.ndarray
    paper_reflectance_rgb: np.ndarray
    pre_sample_reflectance_rgb: np.ndarray
    scanner_linear_rgb: np.ndarray
    scanner_output_rgb: np.ndarray


def simulate_color_print_surface(
    digital_rgb: np.ndarray, parameters: ColorPrintScanParameters
) -> ColorPrintSurfaceResult:
    """Stop at reflective printed paper, before either scanner or camera.

    The input RGB is gamma encoded. The RIP uses a simple
    linear-light black generation; a device ICC/RIP is needed for calibration.
    """
    rgb = rgb_unit_float(digital_rgb)
    height, width = rgb.shape[:2]
    inches_w, inches_h = width / parameters.digital_ppi, height / parameters.digital_ppi
    fine_w, fine_h = (
        round(inches_w * parameters.render_ppi),
        round(inches_h * parameters.render_ppi),
    )
    if fine_w * fine_h > parameters.max_render_pixels:
        raise ValueError(f"render grid {fine_w}x{fine_h} exceeds configured limit")
    if min(fine_w, fine_h) < 2:
        raise ValueError("physical source patch is too small for render grid")
    x = (np.arange(fine_w, dtype=np.float32) + 0.5) / parameters.render_ppi
    y = (np.arange(fine_h, dtype=np.float32) + 0.5) / parameters.render_ppi
    xx, yy = np.meshgrid(x, y)
    coords = [yy * parameters.digital_ppi - 0.5, xx * parameters.digital_ppi - 0.5]
    encoded = np.stack(
        [
            map_coordinates(rgb[:, :, c], coords, order=1, mode="nearest")
            for c in range(3)
        ],
        axis=2,
    )
    linear_rgb = srgb_to_linear(encoded)
    subtractive = 1 - linear_rgb
    black = np.min(subtractive, axis=2) * parameters.black_generation
    cmyk = np.concatenate((subtractive - black[:, :, None], black[:, :, None]), axis=2)
    cmyk = np.power(np.clip(cmyk, 0, 1), parameters.print_gamma)
    ink = np.empty((fine_h, fine_w, 4), dtype=np.bool_)
    for channel in range(4):
        angle = np.deg2rad(parameters.screen_angles_degrees[channel])
        shift_x, shift_y = parameters.registration_um[channel]
        phase_u, phase_v = parameters.screen_phases[channel]
        px = xx - shift_x / UM_PER_INCH
        py = yy - shift_y / UM_PER_INCH
        uu = parameters.screen_lpi * (np.cos(angle) * px + np.sin(angle) * py) + phase_u
        vv = (
            parameters.screen_lpi * (-np.sin(angle) * px + np.cos(angle) * py) + phase_v
        )
        distance = np.maximum(np.abs(np.mod(uu, 1) - 0.5), np.abs(np.mod(vv, 1) - 0.5))
        ink[:, :, channel] = (cmyk[:, :, channel] >= 1) | (
            distance < 0.5 * np.sqrt(cmyk[:, :, channel])
        )
    # Effective optical density represents a double pass through the printed
    # layer. Spatial masks overlap before paper scattering, preserving rosettes.
    deposited = blur_physical(
        ink.astype(np.float32), parameters.deposition_blur_um, parameters.render_ppi
    )
    density_rgb = deposited @ np.asarray(parameters.ink_density_rgb, dtype=np.float32)
    paper = (
        np.asarray(parameters.paper_reflectance_rgb, dtype=np.float32)
        * np.exp(-density_rgb)
    ).astype(np.float32)
    paper = blur_physical(
        paper, parameters.paper_scatter_sigma_um, parameters.render_ppi
    )
    return ColorPrintSurfaceResult(
        ink, paper, parameters.render_ppi, (inches_w, inches_h)
    )


def scan_color_print_surface(
    surface: ColorPrintSurfaceResult, parameters: ColorPrintScanParameters
) -> ColorPrintScanResult:
    """Illuminate and sample an already rendered color print with a flatbed."""
    paper = surface.paper_reflectance_rgb
    if not np.isclose(surface.paper_ppi, parameters.render_ppi):
        raise ValueError("surface PPI differs from scanner configuration")
    inches_w, inches_h = surface.physical_size_inches
    ink = surface.ink_masks_cmyk
    aperture_sigma_um = UM_PER_INCH / (parameters.scan_ppi * np.sqrt(12.0))
    scanner_sigma_um = float(
        np.hypot(parameters.scanner_optical_sigma_um, aperture_sigma_um)
    )
    before_scan = blur_physical(paper, scanner_sigma_um, parameters.render_ppi)
    scan_w, scan_h = (
        round(inches_w * parameters.scan_ppi),
        round(inches_h * parameters.scan_ppi),
    )
    scan_x = (np.arange(scan_w, dtype=np.float32) + 0.5) / parameters.scan_ppi
    scan_y = (np.arange(scan_h, dtype=np.float32) + 0.5) / parameters.scan_ppi
    sx, sy = np.meshgrid(scan_x, scan_y)
    rotation = np.deg2rad(parameters.scanner_rotation_degrees)
    cx, cy = inches_w / 2, inches_h / 2
    physical_x = cx + np.cos(rotation) * (sx - cx) - np.sin(rotation) * (sy - cy)
    physical_y = cy + np.sin(rotation) * (sx - cx) + np.cos(rotation) * (sy - cy)
    sample_coords = [
        physical_y * parameters.render_ppi - 0.5,
        physical_x * parameters.render_ppi - 0.5,
    ]
    linear = np.stack(
        [
            map_coordinates(
                before_scan[:, :, c],
                sample_coords,
                order=1,
                mode="constant",
                cval=parameters.paper_reflectance_rgb[c],
            )
            for c in range(3)
        ],
        axis=2,
    ).astype(np.float32)
    if parameters.scanner_noise_std_linear:
        rng = np.random.default_rng(parameters.random_seed)
        linear += rng.normal(
            0, parameters.scanner_noise_std_linear, linear.shape
        ).astype(np.float32)
    output = np.power(np.clip(linear, 0, 1), 1 / parameters.scanner_gamma).astype(
        np.float32
    )
    return ColorPrintScanResult(ink, paper, before_scan, linear, output)


def simulate_color_print_scan(
    digital_rgb: np.ndarray, parameters: ColorPrintScanParameters
) -> ColorPrintScanResult:
    """Convenience wrapper for print surface followed by flatbed scanner."""
    surface = simulate_color_print_surface(digital_rgb, parameters)
    return scan_color_print_surface(surface, parameters)
