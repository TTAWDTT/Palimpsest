"""A deliberately small, calibratable screen-to-sensor forward model.

Coordinates of ``sensor_to_display`` map continuous sensor pixel coordinates
to continuous display pixel coordinates. Axis-aligned views integrate the
rectangular subpixel overlap; general homographies use bounded fine-grid
quadrature. Vertical RGB subpixels, a global square-wave display PWM, a
Gaussian optical PSF, and an RGGB sensor are hypotheses, not calibrated facts.
"""

from dataclasses import dataclass
from math import ceil, floor, sqrt, pi

import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from scipy.signal import fftconvolve
from scipy.special import j1, ndtr
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class ScreenCaptureParameters:
    sensor_to_display: np.ndarray
    fill_fraction: float = 0.85
    display_gamma: float = 2.2
    optical_blur_sigma_sensor_pixels: float = 0.0
    diffraction_f_number: float | None = None
    sensor_pixel_pitch_um: float | None = None
    rgb_effective_wavelengths_nm: tuple[float, float, float] = (610.0, 540.0, 460.0)
    sensor_spectral_mix_rgb: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
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
    isp_luma_sharpen_amount: float = 0.0
    isp_luma_sharpen_sigma_pixels: float = 1.0

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
            self.isp_luma_sharpen_amount, self.isp_luma_sharpen_sigma_pixels,
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
        if self.diffraction_f_number is not None:
            if (not np.isfinite(self.diffraction_f_number) or self.diffraction_f_number <= 0 or
                    self.sensor_pixel_pitch_um is None or
                    not np.isfinite(self.sensor_pixel_pitch_um) or self.sensor_pixel_pitch_um <= 0):
                raise ValueError("diffraction requires positive f-number and sensor pixel pitch")
        if (len(self.rgb_effective_wavelengths_nm) != 3 or
                not np.isfinite(self.rgb_effective_wavelengths_nm).all() or
                min(self.rgb_effective_wavelengths_nm) <= 0):
            raise ValueError("RGB effective wavelengths must be three finite positive values")
        spectral_mix = np.asarray(self.sensor_spectral_mix_rgb, dtype=np.float64)
        if (spectral_mix.shape != (3, 3) or not np.isfinite(spectral_mix).all()
                or np.any(spectral_mix < 0) or np.any(spectral_mix.sum(axis=1) <= 0)):
            raise ValueError("sensor spectral mix must be a nonnegative finite 3x3 matrix with positive rows")
        object.__setattr__(self, "sensor_spectral_mix_rgb",
                           tuple(tuple(float(value) for value in row) for row in spectral_mix))
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
        if self.isp_luma_sharpen_amount < 0 or self.isp_luma_sharpen_sigma_pixels <= 0:
            raise ValueError("ISP luma sharpening amount must be nonnegative and sigma positive")
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
    emitter_band_irradiance: np.ndarray
    irradiance: np.ndarray
    noiseless_mosaic: np.ndarray
    raw_mosaic: np.ndarray
    srgb_before_sharpen: np.ndarray
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


def _normal_cdf_antiderivative(z: np.ndarray) -> np.ndarray:
    """F'(z)=Phi(z), with Phi the standard-normal CDF."""
    return z * ndtr(z) + np.exp(-.5 * z * z) / sqrt(2 * pi)


def _emitter_pixel_weight(left: float, right: float,
                          emitter_left: np.ndarray, emitter_right: np.ndarray,
                          sigma: float) -> np.ndarray:
    """Mean Gaussian-blurred rectangle irradiance over one sensor footprint.

    All coordinates and sigma are in display-pixel units. At sigma=0 this is
    the exact rectangle overlap; positive sigma integrates the Gaussian CDF
    over the sensor pixel, rather than point-sampling the optical image.
    """
    width = right - left
    if sigma == 0:
        return np.maximum(0, np.minimum(right, emitter_right) -
                          np.maximum(left, emitter_left)) / width
    u = _normal_cdf_antiderivative
    weight = sigma / width * (
        u((emitter_right - left) / sigma) - u((emitter_right - right) / sigma)
        - u((emitter_left - left) / sigma) + u((emitter_left - right) / sigma)
    )
    return np.clip(weight, 0, 1)


def _axis_emitter_matrix(sensor_length: int, display_length: int,
                         scale: float, offset: float, fill_fraction: float,
                         sigma_sensor: float, channel: int | None) -> csr_matrix:
    """Sparse exact area+Gaussian transfer along one axis.

    channel=None integrates the common vertical display-pixel fill. Other
    values select an RGB stripe in a horizontal display pixel.
    """
    gap = (1 - fill_fraction) / 2
    sigma = sigma_sensor * scale
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for sensor_index in range(sensor_length):
        left = scale * sensor_index + offset
        right = left + scale
        support = 6 * sigma
        first = max(0, floor(left - support) - 1)
        last = min(display_length, ceil(right + support) + 1)
        if first >= last:
            continue
        indices = np.arange(first, last, dtype=np.int32)
        if channel is None:
            emitter_left = indices + gap
            emitter_right = indices + 1 - gap
        else:
            emitter_left = indices + (channel + gap) / 3
            emitter_right = indices + (channel + 1 - gap) / 3
        weights = _emitter_pixel_weight(left, right, emitter_left, emitter_right, sigma)
        keep = weights > 1e-9
        rows.extend([sensor_index] * int(keep.sum()))
        columns.extend(indices[keep].tolist())
        data.extend(weights[keep].tolist())
    return csr_matrix((np.asarray(data, dtype=np.float32),
                       (np.asarray(rows), np.asarray(columns))),
                      shape=(sensor_length, display_length), dtype=np.float32)


def _spatial_axis_analytic(emitted_frame: np.ndarray,
                           sensor_shape: tuple[int, int],
                           parameters: ScreenCaptureParameters) -> np.ndarray:
    """Exact rectangle/Gaussian/pixel integral for front-facing display.

    Outside the finite source frame radiance is zero. Optical support extends
    beyond the sensor crop, so no artificial reflection at crop boundaries is
    imposed. The optional Airy diffraction kernel is not covered here.
    """
    transform = parameters.sensor_to_display
    if not _is_axis_aligned_positive(transform) or parameters.diffraction_f_number is not None:
        raise ValueError("analytic spatial method requires positive axis alignment and no diffraction")
    sensor_h, sensor_w = sensor_shape
    display_h, display_w, _ = emitted_frame.shape
    sigma = parameters.optical_blur_sigma_sensor_pixels
    vertical = _axis_emitter_matrix(sensor_h, display_h, transform[1, 1],
                                    transform[1, 2], parameters.fill_fraction, sigma, None)
    result = np.empty((sensor_h, sensor_w, 3), dtype=np.float32)
    for channel in range(3):
        horizontal = _axis_emitter_matrix(sensor_w, display_w, transform[0, 0],
                                          transform[0, 2], parameters.fill_fraction,
                                          sigma, channel)
        intermediate = vertical @ emitted_frame[..., channel]
        result[..., channel] = (horizontal @ intermediate.T).T
    return result


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
    """Fraction of peak-on display radiance collected during each row exposure.

    The display input and electron-rate parameters refer to the emitter's
    *on-state* radiance. Lower duty therefore collects fewer photons over a
    complete PWM cycle; no compensation for perceived brightness is assumed.
    """
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
    gain = parameters.pwm_off_level + (1 - parameters.pwm_off_level) * bright_fraction
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


def _airy_radius_sensor_pixels(parameters: ScreenCaptureParameters) -> float:
    """Largest first-zero radius for the assumed RGB monochromatic bands."""
    if parameters.diffraction_f_number is None:
        return 0.0
    return (1.22 * max(parameters.rgb_effective_wavelengths_nm) * 1e-3 *
            parameters.diffraction_f_number / parameters.sensor_pixel_pitch_um)


def _airy_kernel(radius_fine: int, first_zero_fine: float) -> np.ndarray:
    """Normalized circular-aperture intensity PSF, truncated at 4 first zeros."""
    grid = np.arange(-radius_fine, radius_fine + 1, dtype=np.float64)
    xx, yy = np.meshgrid(grid, grid)
    # The first zero of J1 occurs at 3.8317 = pi * 1.21967.
    argument = np.pi * 1.22 * np.hypot(xx, yy) / first_zero_fine
    amplitude = np.ones_like(argument)
    np.divide(2 * j1(argument), argument, out=amplitude, where=argument != 0)
    kernel = amplitude * amplitude
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def _diffraction_blur(fine_radiance: np.ndarray, parameters: ScreenCaptureParameters,
                      oversampling: int) -> np.ndarray:
    if parameters.diffraction_f_number is None:
        return fine_radiance
    blurred = np.empty_like(fine_radiance)
    for channel, wavelength_nm in enumerate(parameters.rgb_effective_wavelengths_nm):
        first_zero_fine = (1.22 * wavelength_nm * 1e-3 * parameters.diffraction_f_number /
                           parameters.sensor_pixel_pitch_um * oversampling)
        radius = int(np.ceil(4 * first_zero_fine))
        if radius > 512:
            raise ValueError("diffraction PSF support exceeds 512 fine cells; reduce oversampling")
        kernel = _airy_kernel(radius, first_zero_fine)
        padded = np.pad(fine_radiance[..., channel], radius, mode="symmetric")
        convolution = fftconvolve(padded, kernel, mode="same")
        blurred[..., channel] = np.maximum(convolution[radius:-radius, radius:-radius], 0)
    return blurred


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


def _isp_luma_unsharp(srgb: np.ndarray, amount: float, sigma_pixels: float) -> np.ndarray:
    """Optional post-tone luma edge enhancement; not a calibrated camera ISP."""
    if amount == 0:
        return srgb
    luma = (srgb[..., 0] * .2126 + srgb[..., 1] * .7152 + srgb[..., 2] * .0722)
    lowpass = gaussian_filter(luma, sigma_pixels, mode="reflect")
    delta = amount * (luma - lowpass)
    return np.clip(srgb + delta[..., None], 0, 1).astype(np.float32)


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
    halo = (int(np.ceil(4 * parameters.optical_blur_sigma_sensor_pixels)) +
            int(np.ceil(4 * _airy_radius_sensor_pixels(parameters))) +
            (1 if parameters.optical_blur_sigma_sensor_pixels > 0 else 0))
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
    fine_radiance = _diffraction_blur(fine_radiance, parameters, oversampling)
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
    spatial_method: str = "fine",
) -> ScreenCaptureResult:
    """Render display, projective optics, sensor integration, RAW and simple ISP.

    The output is an unencoded crop. Optional temporal PWM/rolling exposure
    uses one square-wave global screen luminance signal and row start times.
    The optional fixed spectral mix is an effective display-primary to sensor
    channel response, not measured spectral sensitivity. Measured PSF, lens
    distortion, real ISP and JPEG remain outside this model. Optional luma
    sharpening is an ISP hypothesis. Parameters are virtual until fitted.
    """
    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim != 3 or frame.shape[2] != 3 or not np.isfinite(frame).all():
        raise ValueError("frame must be a finite HxWx3 array")
    if frame.size == 0 or frame.min() < 0 or frame.max() > 1:
        raise ValueError("frame values must lie in [0, 1]")
    if len(sensor_shape) != 2 or any(int(size) != size or size <= 0 for size in sensor_shape):
        raise ValueError("sensor_shape must contain two positive integer dimensions")
    height, width = map(int, sensor_shape)
    if spatial_method not in ("fine", "analytic"):
        raise ValueError("spatial_method must be fine or analytic")
    emitted_frame = np.power(frame, parameters.display_gamma)
    if spatial_method == "analytic":
        if samples_per_sensor_pixel is not None or tile_size_sensor_pixels is not None:
            raise ValueError("analytic spatial method does not use fine-grid samples or tiles")
        irradiance = _spatial_axis_analytic(emitted_frame, (height, width), parameters)
    else:
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
        halo = (int(np.ceil(4 * parameters.optical_blur_sigma_sensor_pixels)) +
                int(np.ceil(4 * _airy_radius_sensor_pixels(parameters))) +
                (1 if parameters.optical_blur_sigma_sensor_pixels > 0 else 0))
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
        for y0 in range(0, height, tile_size):
            y1 = min(height, y0 + tile_size)
            for x0 in range(0, width, tile_size):
                x1 = min(width, x0 + tile_size)
                irradiance[y0:y1, x0:x1] = _spatial_tile(
                    emitted_frame, (height, width), parameters, oversampling, (y0, y1, x0, x1)
                )
    row_exposure_gain = _pwm_row_gain(height, parameters)
    irradiance *= row_exposure_gain[:, None, None]

    emitter_band_irradiance = irradiance
    spectral_mix = np.asarray(parameters.sensor_spectral_mix_rgb, dtype=np.float32)
    if not np.array_equal(spectral_mix, np.eye(3, dtype=np.float32)):
        irradiance = np.einsum("hwd,cd->hwc", emitter_band_irradiance,
                               spectral_mix, optimize=True).astype(np.float32)

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
    srgb_before_sharpen = _linear_to_srgb(linear_rgb)
    return ScreenCaptureResult(
        emitter_band_irradiance=emitter_band_irradiance,
        irradiance=irradiance,
        noiseless_mosaic=noiseless_mosaic,
        raw_mosaic=raw_mosaic,
        srgb_before_sharpen=srgb_before_sharpen,
        srgb=_isp_luma_unsharp(srgb_before_sharpen,
                              parameters.isp_luma_sharpen_amount,
                              parameters.isp_luma_sharpen_sigma_pixels),
        row_exposure_gain=row_exposure_gain,
    )
