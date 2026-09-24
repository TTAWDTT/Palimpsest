"""A deliberately small, calibratable screen-to-sensor forward model.

Coordinates of ``sensor_to_display`` map continuous sensor pixel coordinates
to continuous display pixel coordinates. This first version supports only
frontoparallel scale and translation, so rectangular pixel/subpixel areas can
be integrated accurately. It assumes vertical RGB subpixels and an RGGB
sensor; neither is a universal property of real devices.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve, gaussian_filter


@dataclass(frozen=True)
class ScreenCaptureParameters:
    sensor_to_display: np.ndarray
    fill_fraction: float = 0.85
    display_gamma: float = 2.2
    optical_blur_sigma_sensor_pixels: float = 0.0
    exposure_electrons_per_unit: float | None = None
    full_well_electrons: float = 10000.0
    read_noise_electrons: float = 0.0

    def __post_init__(self) -> None:
        transform = np.asarray(self.sensor_to_display, dtype=np.float64)
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            raise ValueError("sensor_to_display must be a finite 3x3 matrix")
        normalizer = transform[2, 2] if abs(transform[2, 2]) > 1e-12 * np.linalg.norm(transform) else np.linalg.norm(transform)
        transform = transform / normalizer
        if abs(np.linalg.det(transform)) < 1e-12:
            raise ValueError("sensor_to_display must be invertible")
        if not np.allclose(
            transform[[0, 1, 2, 2], [1, 0, 0, 1]], 0, atol=1e-12, rtol=0
        ) or transform[0, 0] <= 0 or transform[1, 1] <= 0:
            raise ValueError("this prototype supports only frontoparallel scale and translation")
        scalars = (
            self.fill_fraction, self.display_gamma,
            self.optical_blur_sigma_sensor_pixels,
            self.full_well_electrons, self.read_noise_electrons,
        )
        if not np.isfinite(scalars).all():
            raise ValueError("screen and sensor parameters must be finite")
        if self.exposure_electrons_per_unit is not None and not np.isfinite(self.exposure_electrons_per_unit):
            raise ValueError("exposure must be finite")
        if not 0 < self.fill_fraction <= 1:
            raise ValueError("fill_fraction must be in (0, 1]")
        if self.display_gamma <= 0 or self.optical_blur_sigma_sensor_pixels < 0:
            raise ValueError("display gamma must be positive and blur nonnegative")
        if ((self.exposure_electrons_per_unit is not None and self.exposure_electrons_per_unit < 0)
                or self.full_well_electrons <= 0 or self.read_noise_electrons < 0):
            raise ValueError("exposure/read noise must be nonnegative and full well positive")
        if self.exposure_electrons_per_unit is None and self.read_noise_electrons > 0:
            raise ValueError("read noise requires numeric exposure; None is noiseless preview")
        transform.setflags(write=False)
        object.__setattr__(self, "sensor_to_display", transform.copy())
        self.sensor_to_display.setflags(write=False)


@dataclass(frozen=True)
class ScreenCaptureResult:
    irradiance: np.ndarray
    noiseless_mosaic: np.ndarray
    raw_mosaic: np.ndarray
    srgb: np.ndarray


def _screen_radiance(
    frame: np.ndarray,
    parameters: ScreenCaptureParameters,
    display_x: np.ndarray,
    display_y: np.ndarray,
    samples_per_sensor_pixel: int,
) -> np.ndarray:
    """Integrate rectangular RGB emitters over each fine sensor cell."""
    height, width, _ = frame.shape
    base_x = np.floor(display_x).astype(np.int64)
    base_y = np.floor(display_y).astype(np.int64)
    cell_width = parameters.sensor_to_display[0, 0] / samples_per_sensor_pixel
    cell_height = parameters.sensor_to_display[1, 1] / samples_per_sensor_pixel
    left, right = display_x - cell_width / 2, display_x + cell_width / 2
    top, bottom = display_y - cell_height / 2, display_y + cell_height / 2
    gap = (1 - parameters.fill_fraction) / 2
    radiance = np.zeros(display_x.shape + (3,), dtype=np.float32)
    for y_offset in (-1, 0, 1):
        row = base_y + y_offset
        row_weight = np.maximum(
            0, np.minimum(bottom, row + 1 - gap) - np.maximum(top, row + gap)
        ) / cell_height
        for x_offset in (-1, 0, 1):
            column = base_x + x_offset
            valid = (row >= 0) & (row < height) & (column >= 0) & (column < width)
            if not np.any(valid):
                continue
            safe_row = np.clip(row, 0, height - 1)
            safe_column = np.clip(column, 0, width - 1)
            for channel in range(3):
                emitter_left = column + (channel + gap) / 3
                emitter_right = column + (channel + 1 - gap) / 3
                column_weight = np.maximum(
                    0, np.minimum(right, emitter_right) - np.maximum(left, emitter_left)
                ) / cell_width
                radiance[..., channel] += (
                    valid * row_weight * column_weight *
                    frame[safe_row, safe_column, channel] ** parameters.display_gamma
                )
    return radiance


def _bayer_masks(shape: tuple[int, int]) -> np.ndarray:
    rows, columns = np.indices(shape)
    masks = np.zeros(shape + (3,), dtype=np.float32)
    masks[..., 0] = (rows % 2 == 0) & (columns % 2 == 0)
    masks[..., 2] = (rows % 2 == 1) & (columns % 2 == 1)
    masks[..., 1] = 1 - masks[..., 0] - masks[..., 2]
    return masks


def _demosaic_bilinear(raw: np.ndarray, masks: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
    channels = []
    for channel in range(3):
        numerator = convolve(raw * masks[..., channel], kernel, mode="reflect")
        denominator = convolve(masks[..., channel], kernel, mode="reflect")
        channels.append(numerator / np.maximum(denominator, 1e-8))
    return np.stack(channels, axis=-1)


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0, 1)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1 / 2.4) - 0.055,
    ).astype(np.float32)


def _minimum_samples_for_projection(
    transform: np.ndarray, sensor_shape: tuple[int, int], fill_fraction: float
) -> int:
    """Keep the blur/integration grid fine relative to projected emitters."""
    del sensor_shape
    horizontal = 24 * transform[0, 0] / fill_fraction
    vertical = 8 * transform[1, 1] / fill_fraction
    return max(2, int(np.ceil(max(horizontal, vertical))))


def render_screen_capture(
    frame: np.ndarray,
    sensor_shape: tuple[int, int],
    parameters: ScreenCaptureParameters,
    *,
    seed: int = 0,
    samples_per_sensor_pixel: int | None = None,
) -> ScreenCaptureResult:
    """Render emission, optics, pixel integration, Bayer/noise, and simple ISP.

    The output is an unencoded frontoparallel crop. It omits spectral
    display/camera response, rolling shutter, measured PSF, device ISP,
    perspective, lens distortion, and JPEG. The
    supplied parameters are virtual until fitted with controlled device data.
    """
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 3 or frame.shape[2] != 3 or not np.isfinite(frame).all():
        raise ValueError("frame must be a finite HxWx3 array")
    if frame.size == 0 or frame.min() < 0 or frame.max() > 1:
        raise ValueError("frame values must lie in [0, 1]")
    if len(sensor_shape) != 2 or any(int(size) != size or size <= 0 for size in sensor_shape):
        raise ValueError("sensor_shape must contain two positive integer dimensions")
    height, width = map(int, sensor_shape)
    required_samples = _minimum_samples_for_projection(
        parameters.sensor_to_display, (height, width), parameters.fill_fraction
    )
    if samples_per_sensor_pixel is None:
        samples_per_sensor_pixel = required_samples
    if isinstance(samples_per_sensor_pixel, (bool, np.bool_)) or not isinstance(samples_per_sensor_pixel, (int, np.integer)):
        raise ValueError("samples_per_sensor_pixel must be an integer")
    if not 2 <= samples_per_sensor_pixel <= 128:
        raise ValueError("samples_per_sensor_pixel must lie in [2, 128]")
    oversampling = int(samples_per_sensor_pixel)
    if oversampling < required_samples:
        raise ValueError(
            f"underresolved emitter: use at least {required_samples} samples per sensor pixel"
        )
    if height * width * oversampling**2 > 12_000_000:
        raise ValueError("render a smaller crop; this prototype is not tiled")

    fine_y, fine_x = np.indices((height * oversampling, width * oversampling))
    sensor_x = (fine_x + 0.5) / oversampling
    sensor_y = (fine_y + 0.5) / oversampling
    transform = parameters.sensor_to_display
    denominator = transform[2, 0] * sensor_x + transform[2, 1] * sensor_y + transform[2, 2]
    if np.any(np.abs(denominator) < 1e-8):
        raise ValueError("projection reaches the homography horizon")
    display_x = (transform[0, 0] * sensor_x + transform[0, 1] * sensor_y + transform[0, 2]) / denominator
    display_y = (transform[1, 0] * sensor_x + transform[1, 1] * sensor_y + transform[1, 2]) / denominator

    fine_radiance = _screen_radiance(frame, parameters, display_x, display_y, oversampling)
    sigma = parameters.optical_blur_sigma_sensor_pixels * oversampling
    if sigma > 0:
        fine_radiance = gaussian_filter(fine_radiance, (sigma, sigma, 0), mode="reflect")
    irradiance = fine_radiance.reshape(height, oversampling, width, oversampling, 3).mean(axis=(1, 3))

    masks = _bayer_masks((height, width))
    mosaiced_irradiance = (irradiance * masks).sum(axis=-1)
    noiseless_mosaic = mosaiced_irradiance.copy()
    raw_mosaic = noiseless_mosaic.copy()
    if parameters.exposure_electrons_per_unit is not None:
        expected_electrons = np.maximum(mosaiced_irradiance, 0) * parameters.exposure_electrons_per_unit
        noiseless_mosaic = np.clip(expected_electrons, 0, parameters.full_well_electrons) / parameters.full_well_electrons
        rng = np.random.default_rng(seed)
        electrons = rng.poisson(expected_electrons)
        if parameters.read_noise_electrons > 0:
            electrons = electrons + rng.normal(0, parameters.read_noise_electrons, electrons.shape)
        raw_mosaic = np.clip(electrons / parameters.full_well_electrons, 0, 1).astype(np.float32)

    linear_rgb = _demosaic_bilinear(raw_mosaic, masks)
    return ScreenCaptureResult(
        irradiance=irradiance,
        noiseless_mosaic=noiseless_mosaic,
        raw_mosaic=raw_mosaic,
        srgb=_linear_to_srgb(linear_rgb),
    )
