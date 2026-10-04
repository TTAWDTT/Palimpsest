"""Restricted reflected-paper to camera-sensor forward model.

The input is a *linear RGB reflectance proxy* sampled on a known physical
paper grid. Light is an ambient term plus an optional point source above a
flat Lambertian sheet. This is not a measured spectral BRDF, camera response
or smartphone ISP. In particular, it must not be fed a scanned RGB image as
if it were physical paper reflectance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .sensor import _bayer_masks, _demosaic_bilinear, _linear_to_srgb


@dataclass(frozen=True)
class PrintCameraParameters:
    sensor_to_paper_inches: np.ndarray
    paper_ppi: float
    ambient_irradiance_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    point_light_position_inches: tuple[float, float, float] | None = None
    point_light_strength_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    optical_blur_sigma_sensor_pixels: float = 0.0
    exposure_electrons_per_unit: float | None = None
    full_well_electrons: float = 10000.0
    read_noise_electrons: float = 0.0
    white_balance_gains_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    samples_per_sensor_pixel: int = 4
    max_fine_pixels: int = 8_000_000
    random_seed: int = 0

    def __post_init__(self) -> None:
        matrix = np.asarray(self.sensor_to_paper_inches, dtype=np.float64)
        if (
            matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
            or abs(np.linalg.det(matrix)) < 1e-15
        ):
            raise ValueError(
                "sensor_to_paper_inches must be a finite invertible 3x3 homography"
            )
        object.__setattr__(self, "sensor_to_paper_inches", matrix)
        if not np.isfinite(self.paper_ppi) or self.paper_ppi <= 0:
            raise ValueError("paper_ppi must be positive")
        for value in (
            self.ambient_irradiance_rgb,
            self.point_light_strength_rgb,
            self.white_balance_gains_rgb,
        ):
            if len(value) != 3 or not np.isfinite(value).all() or min(value) < 0:
                raise ValueError(
                    "RGB illumination and white balance must be finite and nonnegative"
                )
        if min(self.white_balance_gains_rgb) <= 0:
            raise ValueError("white balance gains must be positive")
        if self.point_light_position_inches is not None and (
            len(self.point_light_position_inches) != 3
            or not np.isfinite(self.point_light_position_inches).all()
            or self.point_light_position_inches[2] <= 0
        ):
            raise ValueError(
                "point light must have finite x/y and positive height above paper"
            )
        if (
            max(self.point_light_strength_rgb) > 0
            and self.point_light_position_inches is None
        ):
            raise ValueError("nonzero point light strength requires a position")
        scalars = (
            self.optical_blur_sigma_sensor_pixels,
            self.full_well_electrons,
            self.read_noise_electrons,
        )
        if (
            not np.isfinite(scalars).all()
            or self.full_well_electrons <= 0
            or min(self.optical_blur_sigma_sensor_pixels, self.read_noise_electrons) < 0
        ):
            raise ValueError(
                "optical blur/read noise must be nonnegative; full well positive"
            )
        if self.exposure_electrons_per_unit is not None and (
            not np.isfinite(self.exposure_electrons_per_unit)
            or self.exposure_electrons_per_unit < 0
        ):
            raise ValueError("exposure must be nonnegative")
        if self.exposure_electrons_per_unit is None and self.read_noise_electrons > 0:
            raise ValueError("read noise requires sensor exposure")
        if (
            isinstance(self.samples_per_sensor_pixel, bool)
            or not isinstance(self.samples_per_sensor_pixel, int)
            or not 2 <= self.samples_per_sensor_pixel <= 16
        ):
            raise ValueError("samples_per_sensor_pixel must be integer in [2,16]")
        if self.max_fine_pixels <= 0:
            raise ValueError("max_fine_pixels must be positive")


@dataclass(frozen=True)
class PrintCameraResult:
    irradiance_rgb: np.ndarray
    noiseless_raw_mosaic: np.ndarray
    raw_mosaic: np.ndarray
    srgb: np.ndarray


def photograph_print_surface(
    paper_reflectance_rgb: np.ndarray,
    sensor_shape: tuple[int, int],
    parameters: PrintCameraParameters,
) -> PrintCameraResult:
    """Project a printed paper patch into a simple RGGB camera.

    The homography maps sensor pixel coordinates to paper *inches*. An area
    integral at subpixel samples precedes CFA sampling, so a paper lattice
    changes observed aliasing when the camera pose/scale changes.
    """
    paper = np.asarray(paper_reflectance_rgb, dtype=np.float32)
    if (
        paper.ndim != 3
        or paper.shape[2] != 3
        or min(paper.shape[:2]) <= 0
        or (not np.isfinite(paper).all() or paper.min() < 0 or paper.max() > 1)
    ):
        raise ValueError("paper reflectance must be finite HxWx3 within [0,1]")
    if len(sensor_shape) != 2 or any(
        not isinstance(v, int) or v <= 0 for v in sensor_shape
    ):
        raise ValueError("sensor_shape must contain two positive integers")
    height, width = sensor_shape
    samples = parameters.samples_per_sensor_pixel
    if height * width * samples * samples > parameters.max_fine_pixels:
        raise ValueError(
            "sensor fine grid exceeds max_fine_pixels; render a smaller patch"
        )
    fine_y, fine_x = np.indices((height * samples, width * samples))
    sx, sy = (fine_x + 0.5) / samples, (fine_y + 0.5) / samples
    transform = parameters.sensor_to_paper_inches
    denominator = transform[2, 0] * sx + transform[2, 1] * sy + transform[2, 2]
    if np.any(np.abs(denominator) < 1e-10):
        raise ValueError("homography reaches the horizon")
    px = (transform[0, 0] * sx + transform[0, 1] * sy + transform[0, 2]) / denominator
    py = (transform[1, 0] * sx + transform[1, 1] * sy + transform[1, 2]) / denominator
    paper_w, paper_h = (
        paper.shape[1] / parameters.paper_ppi,
        paper.shape[0] / parameters.paper_ppi,
    )
    if px.min() < 0 or py.min() < 0 or px.max() >= paper_w or py.max() >= paper_h:
        raise ValueError("sensor projection must remain inside the modeled paper patch")
    coords = (py * parameters.paper_ppi - 0.5, px * parameters.paper_ppi - 0.5)
    sampled = np.stack(
        [
            map_coordinates(paper[:, :, c], coords, order=1, mode="nearest")
            for c in range(3)
        ],
        axis=-1,
    )
    illumination = np.asarray(parameters.ambient_irradiance_rgb, dtype=np.float32)
    if parameters.point_light_position_inches is not None:
        lx, ly, lz = parameters.point_light_position_inches
        distance_sq = (px - lx) ** 2 + (py - ly) ** 2 + lz**2
        # Inverse-square distance multiplied by the plane's incident cosine.
        # Strength is irradiance at a 1-inch normal-incidence reference range.
        point_gain = lz / np.power(distance_sq, 1.5)
        illumination = illumination + point_gain[:, :, None] * np.asarray(
            parameters.point_light_strength_rgb, dtype=np.float32
        )
    fine_irradiance = sampled * illumination
    sigma = parameters.optical_blur_sigma_sensor_pixels * samples
    if sigma:
        fine_irradiance = gaussian_filter(
            fine_irradiance, sigma=(sigma, sigma, 0), mode="reflect"
        )
    irradiance = fine_irradiance.reshape(height, samples, width, samples, 3).mean(
        axis=(1, 3)
    )
    masks = _bayer_masks((height, width))
    mosaic = (irradiance * masks).sum(axis=-1)
    noiseless = mosaic.astype(np.float32)
    raw = noiseless.copy()
    if parameters.exposure_electrons_per_unit is not None:
        expectation = np.maximum(mosaic, 0) * parameters.exposure_electrons_per_unit
        noiseless = (
            np.clip(expectation, 0, parameters.full_well_electrons)
            / parameters.full_well_electrons
        )
        rng = np.random.default_rng(parameters.random_seed)
        electrons = rng.poisson(expectation)
        if parameters.read_noise_electrons:
            electrons = electrons + rng.normal(
                0, parameters.read_noise_electrons, electrons.shape
            )
        raw = np.clip(electrons / parameters.full_well_electrons, 0, 1).astype(
            np.float32
        )
    linear = _demosaic_bilinear(raw, masks) * np.asarray(
        parameters.white_balance_gains_rgb
    )
    return PrintCameraResult(
        irradiance.astype(np.float32),
        noiseless.astype(np.float32),
        raw.astype(np.float32),
        _linear_to_srgb(linear),
    )
