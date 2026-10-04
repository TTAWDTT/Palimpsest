"""Restricted continuous-tone photo-paper surface hypothesis.

This is an effective *three-channel density* model for a laser-exposed,
developed photographic print. It is intentionally separate from the ordered
CMYK ink model. No spectral sensitization, chemistry, printer calibration, or
claim about any particular photo lab is encoded here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .color_print_scan import _rgb_float
from .print_scan import UM_PER_INCH


@dataclass(frozen=True)
class PhotoPaperParameters:
    digital_ppi: float = 300.0
    render_ppi: float = 1200.0
    exposure_spot_sigma_um: float = 12.0
    characteristic_steepness: float = 3.0
    max_density_cmy: tuple[float, float, float] = (2.0, 2.0, 2.0)
    dye_absorption_rgb: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.04, 0.04),
        (0.04, 1.0, 0.04),
        (0.04, 0.04, 1.0),
    )
    dye_spread_sigma_um: float = 8.0
    grain_correlation_um: float = 3.0
    grain_density_std: float = 0.02
    paper_reflectance_rgb: tuple[float, float, float] = (0.96, 0.96, 0.96)
    random_seed: int = 0
    max_render_pixels: int = 2_000_000

    def __post_init__(self) -> None:
        positive = (self.digital_ppi, self.render_ppi, self.characteristic_steepness)
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError(
                "sampling resolutions and characteristic steepness must be positive"
            )
        if self.render_ppi < 2 * self.digital_ppi:
            raise ValueError("render grid must oversample the digital source")
        lengths = (
            self.exposure_spot_sigma_um,
            self.dye_spread_sigma_um,
            self.grain_correlation_um,
            self.grain_density_std,
        )
        if not np.isfinite(lengths).all() or min(lengths) < 0:
            raise ValueError(
                "physical widths and grain amplitude must be finite and nonnegative"
            )
        density = np.asarray(self.max_density_cmy, dtype=float)
        absorption = np.asarray(self.dye_absorption_rgb, dtype=float)
        paper = np.asarray(self.paper_reflectance_rgb, dtype=float)
        if (
            density.shape != (3,)
            or absorption.shape != (3, 3)
            or paper.shape != (3,)
            or not np.isfinite(density).all()
            or not np.isfinite(absorption).all()
            or not np.isfinite(paper).all()
            or np.any(density <= 0)
            or np.any(absorption < 0)
            or np.any(paper <= 0)
            or np.any(paper > 1)
        ):
            raise ValueError(
                "density, absorption and paper reflectance must have valid RGB shapes"
            )
        if self.max_render_pixels <= 0:
            raise ValueError("max_render_pixels must be positive")


@dataclass(frozen=True)
class PhotoPaperSurfaceResult:
    dye_density_cmy: np.ndarray
    paper_reflectance_rgb: np.ndarray
    paper_ppi: float
    physical_size_inches: tuple[float, float]


def _blur(image: np.ndarray, sigma_um: float, ppi: float) -> np.ndarray:
    if sigma_um == 0:
        return image
    sigma = sigma_um * ppi / UM_PER_INCH
    return gaussian_filter(image, (sigma, sigma, 0), mode="reflect").astype(np.float32)


def _normalized_characteristic(signal: np.ndarray, steepness: float) -> np.ndarray:
    """Anchored S-shaped density response to a normalized exposure command."""

    def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
        return 1 / (1 + np.exp(-x))

    low = sigmoid(-steepness / 2)
    high = sigmoid(steepness / 2)
    return ((sigmoid(steepness * (signal - 0.5)) - low) / (high - low)).astype(
        np.float32
    )


def simulate_photo_paper_surface(
    digital_rgb: np.ndarray, parameters: PhotoPaperParameters
) -> PhotoPaperSurfaceResult:
    """Digital command -> finite optical exposure spot -> developed dye -> paper.

    The RGB-to-exposure mapping is an uncalibrated effective proxy: real color
    papers have spectrally coupled layers and device-specific exposure curves.
    Grain is frozen in paper coordinates, so repeated photographs share it.
    """
    rgb = _rgb_float(digital_rgb)
    height, width = rgb.shape[:2]
    size_inches = (width / parameters.digital_ppi, height / parameters.digital_ppi)
    fine_w, fine_h = (
        round(size_inches[0] * parameters.render_ppi),
        round(size_inches[1] * parameters.render_ppi),
    )
    if fine_w * fine_h > parameters.max_render_pixels or min(fine_w, fine_h) < 2:
        raise ValueError("render grid exceeds limit or is degenerate")
    xx, yy = np.meshgrid(
        (np.arange(fine_w, dtype=np.float32) + 0.5) / parameters.render_ppi,
        (np.arange(fine_h, dtype=np.float32) + 0.5) / parameters.render_ppi,
    )
    coords = [yy * parameters.digital_ppi - 0.5, xx * parameters.digital_ppi - 0.5]
    encoded = np.stack(
        [
            map_coordinates(rgb[:, :, c], coords, order=1, mode="nearest")
            for c in range(3)
        ],
        axis=2,
    )
    linear = np.where(
        encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4
    )
    # More darkening command produces more effective CMY dye density. The
    # actual laser exposure polarity and color separation remain unknown.
    exposure_command = _blur(
        (1 - linear).astype(np.float32),
        parameters.exposure_spot_sigma_um,
        parameters.render_ppi,
    )
    developed_fraction = _normalized_characteristic(
        exposure_command, parameters.characteristic_steepness
    )
    density = developed_fraction * np.asarray(
        parameters.max_density_cmy, dtype=np.float32
    )
    density = _blur(density, parameters.dye_spread_sigma_um, parameters.render_ppi)
    if parameters.grain_density_std:
        rng = np.random.default_rng(parameters.random_seed)
        grain = rng.standard_normal(density.shape, dtype=np.float32)
        grain = _blur(grain, parameters.grain_correlation_um, parameters.render_ppi)
        # Set amplitude after spatial correlation; it is a density-domain
        # perturbation strongest at intermediate developed fractions.
        grain_std = np.std(grain, axis=(0, 1), keepdims=True)
        grain = grain / np.maximum(grain_std, 1e-6)
        density = np.maximum(
            0,
            density
            + parameters.grain_density_std
            * np.sqrt(np.maximum(developed_fraction * (1 - developed_fraction), 0))
            * grain,
        )
    effective_density = density @ np.asarray(
        parameters.dye_absorption_rgb, dtype=np.float32
    )
    paper = (
        np.asarray(parameters.paper_reflectance_rgb, dtype=np.float32)
        * np.exp(-effective_density)
    ).astype(np.float32)
    return PhotoPaperSurfaceResult(
        density.astype(np.float32), paper, parameters.render_ppi, size_inches
    )
