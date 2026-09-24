"""A deliberately small, calibratable screen-to-sensor forward model.

Coordinates of ``sensor_to_display`` map continuous sensor pixel coordinates
to continuous display pixel coordinates. Axis-aligned views integrate the
rectangular subpixel overlap; general homographies use bounded fine-grid
quadrature. Vertical RGB subpixels, a global square-wave display PWM, a
Gaussian optical PSF, and an RGGB sensor are hypotheses, not calibrated facts.
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from scipy.signal import fftconvolve


@dataclass(frozen=True)
class ScreenCaptureParameters:
    sensor_to_display: np.ndarray
    fill_fraction: float = 0.85
    display_gamma: float = 2.2
    optical_blur_sigma_sensor_pixels: float = 0.0
    exposure_electrons_per_unit: float | None = None
    electron_rate_per_unit_s: float | None = None
    full_well_electrons: float = 10000.0
    read_noise_electrons: float = 0.0
    pwm_frequency_hz: float | None = None
    pwm_duty_cycle: float = 1.0
    pwm_off_level: float = 0.0
    exposure_time_s: float | None = None
    sensor_row_interval_s: float = 0.0
    pwm_phase_cycles: float = 0.0

    def __post_init__(self) -> None:
        transform = np.asarray(self.sensor_to_display, dtype=np.float64)
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            raise ValueError("sensor_to_display must be a finite 3x3 matrix")
        normalizer = transform[2, 2] if abs(transform[2, 2]) > 1e-12 * np.linalg.norm(transform) else np.linalg.norm(transform)
        transform = transform / normalizer
        if abs(np.linalg.det(transform)) < 1e-12:
            raise ValueError("sensor_to_display must be invertible")
        scalars = (
            self.fill_fraction, self.display_gamma,
            self.optical_blur_sigma_sensor_pixels,
            self.full_well_electrons, self.read_noise_electrons,
            self.pwm_duty_cycle, self.pwm_off_level,
            self.sensor_row_interval_s, self.pwm_phase_cycles,
        )
        if not np.isfinite(scalars).all():
            raise ValueError("screen and sensor parameters must be finite")
        if self.exposure_electrons_per_unit is not None and not np.isfinite(self.exposure_electrons_per_unit):
            raise ValueError("exposure must be finite")
        if self.electron_rate_per_unit_s is not None and (
                not np.isfinite(self.electron_rate_per_unit_s) or self.electron_rate_per_unit_s < 0):
            raise ValueError("electron rate must be finite and nonnegative")
        if self.exposure_electrons_per_unit is not None and self.electron_rate_per_unit_s is not None:
            raise ValueError("choose either integrated electrons or electron rate")
        if not 0 < self.fill_fraction <= 1:
            raise ValueError("fill_fraction must be in (0, 1]")
        if self.display_gamma <= 0 or self.optical_blur_sigma_sensor_pixels < 0:
            raise ValueError("display gamma must be positive and blur nonnegative")
        if ((self.exposure_electrons_per_unit is not None and self.exposure_electrons_per_unit < 0)
                or self.full_well_electrons <= 0 or self.read_noise_electrons < 0):
            raise ValueError("exposure/read noise must be nonnegative and full well positive")
        if (self.exposure_electrons_per_unit is None and self.electron_rate_per_unit_s is None
                and self.read_noise_electrons > 0):
            raise ValueError("read noise requires numeric exposure; None is noiseless preview")
        if not 0 < self.pwm_duty_cycle <= 1 or not 0 <= self.pwm_off_level <= 1:
            raise ValueError("PWM duty must be in (0, 1] and off level in [0, 1]")
        if self.sensor_row_interval_s < 0 or not 0 <= self.pwm_phase_cycles < 1:
            raise ValueError("row interval must be nonnegative and PWM phase in [0, 1)")
        if self.exposure_time_s is not None and (
                not np.isfinite(self.exposure_time_s) or self.exposure_time_s <= 0):
            raise ValueError("exposure_time_s must be positive and finite")
        if self.pwm_frequency_hz is not None and (
                not np.isfinite(self.pwm_frequency_hz) or self.pwm_frequency_hz <= 0
                or self.exposure_time_s is None):
            raise ValueError("PWM requires positive frequency and exposure time")
        if self.electron_rate_per_unit_s is not None and self.exposure_time_s is None:
            raise ValueError("electron rate requires exposure_time_s")
        transform.setflags(write=False)
        object.__setattr__(self, "sensor_to_display", transform.copy())
        self.sensor_to_display.setflags(write=False)


@dataclass(frozen=True)
class ScreenCaptureResult:
    irradiance: np.ndarray
    noiseless_mosaic: np.ndarray
    raw_mosaic: np.ndarray
    srgb: np.ndarray
    row_exposure_gain: np.ndarray


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
                    frame[safe_row, safe_column, channel]
                )
    return radiance


def _screen_radiance_projective(
    frame: np.ndarray,
    parameters: ScreenCaptureParameters,
    display_x: np.ndarray,
    display_y: np.ndarray,
) -> np.ndarray:
    """Point-quadrature of the projected rectangular RGB emitter lattice.

    Each point is one fine-grid cell center. The later pixel integration takes
    their mean. This is slower and less exact than the axis-aligned overlap
    integral, and should be checked for convergence at the chosen crop/pose.
    """
    height, width, _ = frame.shape
    x = np.floor(display_x).astype(np.int64)
    y = np.floor(display_y).astype(np.int64)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    safe_x = np.clip(x, 0, width - 1)
    safe_y = np.clip(y, 0, height - 1)
    phase_x = display_x - x
    phase_y = display_y - y
    gap = (1 - parameters.fill_fraction) / 2
    active_y = (phase_y >= gap) & (phase_y < 1 - gap) & valid
    radiance = np.zeros(display_x.shape + (3,), dtype=np.float32)
    for channel in range(3):
        active_x = (phase_x >= (channel + gap) / 3) & (phase_x < (channel + 1 - gap) / 3)
        radiance[..., channel] = active_y * active_x * frame[safe_y, safe_x, channel]
    return radiance


def _is_axis_aligned_positive(transform: np.ndarray) -> bool:
    return bool(
        np.allclose(transform[[0, 1, 2, 2], [1, 0, 0, 1]], 0, atol=1e-12, rtol=0)
        and transform[0, 0] > 0 and transform[1, 1] > 0
    )


def _homography_jacobian(transform: np.ndarray, x: float, y: float) -> np.ndarray:
    denominator = transform[2, 0] * x + transform[2, 1] * y + transform[2, 2]
    if abs(denominator) < 1e-10:
        raise ValueError("projection reaches the homography horizon")
    numerator_x = transform[0, 0] * x + transform[0, 1] * y + transform[0, 2]
    numerator_y = transform[1, 0] * x + transform[1, 1] * y + transform[1, 2]
    return np.array([
        [(transform[0, 0] * denominator - numerator_x * transform[2, 0]) / denominator**2,
         (transform[0, 1] * denominator - numerator_x * transform[2, 1]) / denominator**2],
        [(transform[1, 0] * denominator - numerator_y * transform[2, 0]) / denominator**2,
         (transform[1, 1] * denominator - numerator_y * transform[2, 1]) / denominator**2],
    ])


def _pwm_row_gain(height: int, parameters: ScreenCaptureParameters) -> np.ndarray:
    if parameters.pwm_frequency_hz is None:
        return np.ones(height, dtype=np.float32)
    frequency = parameters.pwm_frequency_hz
    duty = parameters.pwm_duty_cycle
    duration = parameters.exposure_time_s
    phase = parameters.pwm_phase_cycles
    row_start_cycles = phase + np.arange(height, dtype=np.float64) * parameters.sensor_row_interval_s * frequency
    row_end_cycles = row_start_cycles + duration * frequency

    def bright_cycles(t: np.ndarray) -> np.ndarray:
        whole = np.floor(t)
        return whole * duty + np.minimum(t - whole, duty)

    bright_fraction = (bright_cycles(row_end_cycles) - bright_cycles(row_start_cycles)) / (duration * frequency)
    mean_level = parameters.pwm_off_level + (1 - parameters.pwm_off_level) * duty
    gain = (parameters.pwm_off_level + (1 - parameters.pwm_off_level) * bright_fraction) / mean_level
    return gain.astype(np.float32)


def _optical_blur(fine_radiance: np.ndarray, sigma: float) -> np.ndarray:
    """Reflect-boundary Gaussian blur, using FFT for large fine grids."""
    if sigma <= 0:
        return fine_radiance
    cells = fine_radiance.shape[0] * fine_radiance.shape[1]
    if sigma < 6 or cells < 200_000 or cells > 8_000_000:
        return gaussian_filter(fine_radiance, (sigma, sigma, 0), mode="reflect")
    radius = int(4 * sigma + .5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel_1d = np.exp(-.5 * (x / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()
    kernel = np.outer(kernel_1d, kernel_1d).astype(np.float32)
    padded = np.pad(fine_radiance, ((radius, radius), (radius, radius), (0, 0)), mode="symmetric")
    blurred = fftconvolve(padded, kernel[:, :, None], mode="same", axes=(0, 1))
    return blurred[radius:-radius, radius:-radius].astype(np.float32)


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
    if _is_axis_aligned_positive(transform):
        horizontal = 24 * transform[0, 0] / fill_fraction
        vertical = 8 * transform[1, 1] / fill_fraction
    else:
        height, width = sensor_shape
        points = ((0, 0), (width, 0), (0, height), (width, height), (width / 2, height / 2))
        denominators = np.array([transform[2, 0] * x + transform[2, 1] * y + transform[2, 2]
                                 for x, y in points[:4]])
        if np.min(denominators) <= 1e-10 and np.max(denominators) >= -1e-10:
            raise ValueError("projection reaches the homography horizon")
        jacobians = np.stack([_homography_jacobian(transform, x, y) for x, y in points])
        horizontal = 24 * np.max(np.abs(jacobians[:, 0, :])) / fill_fraction
        vertical = 8 * np.max(np.abs(jacobians[:, 1, :])) / fill_fraction
    return max(2, int(np.ceil(max(horizontal, vertical))))


def _spatial_tile(
    emitted_frame: np.ndarray,
    sensor_shape: tuple[int, int],
    parameters: ScreenCaptureParameters,
    oversampling: int,
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    """Render one sensor rectangle with enough halo for the Gaussian PSF."""
    height, width = sensor_shape
    y0, y1, x0, x1 = bounds
    halo = int(np.ceil(4 * parameters.optical_blur_sigma_sensor_pixels)) + (
        1 if parameters.optical_blur_sigma_sensor_pixels > 0 else 0
    )
    extended_y0, extended_y1 = max(0, y0 - halo), min(height, y1 + halo)
    extended_x0, extended_x1 = max(0, x0 - halo), min(width, x1 + halo)
    extended_height = extended_y1 - extended_y0
    extended_width = extended_x1 - extended_x0

    fine_y, fine_x = np.indices((extended_height * oversampling, extended_width * oversampling))
    sensor_x = extended_x0 + (fine_x + 0.5) / oversampling
    sensor_y = extended_y0 + (fine_y + 0.5) / oversampling
    transform = parameters.sensor_to_display
    denominator = transform[2, 0] * sensor_x + transform[2, 1] * sensor_y + transform[2, 2]
    if np.any(np.abs(denominator) < 1e-8):
        raise ValueError("projection reaches the homography horizon")
    display_x = (transform[0, 0] * sensor_x + transform[0, 1] * sensor_y + transform[0, 2]) / denominator
    display_y = (transform[1, 0] * sensor_x + transform[1, 1] * sensor_y + transform[1, 2]) / denominator

    if _is_axis_aligned_positive(transform):
        fine_radiance = _screen_radiance(emitted_frame, parameters, display_x, display_y, oversampling)
    else:
        fine_radiance = _screen_radiance_projective(emitted_frame, parameters, display_x, display_y)
    sigma = parameters.optical_blur_sigma_sensor_pixels * oversampling
    if sigma > 0:
        fine_radiance = _optical_blur(fine_radiance, sigma)
    averaged = fine_radiance.reshape(extended_height, oversampling, extended_width, oversampling, 3).mean(axis=(1, 3))
    return averaged[y0 - extended_y0:y1 - extended_y0, x0 - extended_x0:x1 - extended_x0]


def render_screen_capture(
    frame: np.ndarray,
    sensor_shape: tuple[int, int],
    parameters: ScreenCaptureParameters,
    *,
    seed: int = 0,
    samples_per_sensor_pixel: int | None = None,
    tile_size_sensor_pixels: int | None = None,
) -> ScreenCaptureResult:
    """Render display, projective optics, sensor integration, RAW and simple ISP.

    The output is an unencoded crop. Optional temporal PWM/rolling exposure
    uses one square-wave global screen luminance signal and row start times.
    Spectral response, measured PSF, lens distortion, real ISP and JPEG remain
    outside this model. Parameters are virtual until fitted to device data.
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
    halo = int(np.ceil(4 * parameters.optical_blur_sigma_sensor_pixels)) + (
        1 if parameters.optical_blur_sigma_sensor_pixels > 0 else 0
    )
    safe_side = int(np.floor(np.sqrt(12_000_000) / oversampling)) - 2 * halo
    if safe_side < 1:
        raise ValueError("oversampling and optical blur exceed tile memory limit")
    if tile_size_sensor_pixels is not None and (
            isinstance(tile_size_sensor_pixels, (bool, np.bool_))
            or not isinstance(tile_size_sensor_pixels, (int, np.integer))
            or tile_size_sensor_pixels < 1):
        raise ValueError("tile_size_sensor_pixels must be a positive integer")
    tile_size = min(safe_side, int(tile_size_sensor_pixels) if tile_size_sensor_pixels is not None else safe_side)
    irradiance = np.empty((height, width, 3), dtype=np.float32)
    emitted_frame = np.power(frame, parameters.display_gamma)
    for y0 in range(0, height, tile_size):
        y1 = min(height, y0 + tile_size)
        for x0 in range(0, width, tile_size):
            x1 = min(width, x0 + tile_size)
            irradiance[y0:y1, x0:x1] = _spatial_tile(
                emitted_frame, (height, width), parameters, oversampling, (y0, y1, x0, x1)
            )
    row_exposure_gain = _pwm_row_gain(height, parameters)
    irradiance *= row_exposure_gain[:, None, None]

    masks = _bayer_masks((height, width))
    mosaiced_irradiance = (irradiance * masks).sum(axis=-1)
    noiseless_mosaic = mosaiced_irradiance.copy()
    raw_mosaic = noiseless_mosaic.copy()
    electrons_per_unit = parameters.exposure_electrons_per_unit
    if parameters.electron_rate_per_unit_s is not None:
        electrons_per_unit = parameters.electron_rate_per_unit_s * parameters.exposure_time_s
    if electrons_per_unit is not None:
        expected_electrons = np.maximum(mosaiced_irradiance, 0) * electrons_per_unit
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
        row_exposure_gain=row_exposure_gain,
    )
