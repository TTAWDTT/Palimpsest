"""Spatial quadrature and accelerated approximations with validity guards."""

from math import ceil, floor, pi, sqrt

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.sparse import csr_matrix
from scipy.special import ndtr

from palimpsest.simulation._screen.emission import (
    _display_emitter_raster,
    _screen_radiance,
    _screen_radiance_projective,
)
from palimpsest.simulation._screen.optics import (
    _airy_radius_sensor_pixels,
    _defocus_disk_blur,
    _diffraction_blur,
    _effective_diffraction_f_number,
    _optical_blur,
    _optical_halo_sensor_pixels,
    _wave_defocus_blur,
    thin_lens_coc_radius_sensor_pixels,
)
from palimpsest.simulation._screen.parameters import ScreenCaptureParameters
from palimpsest.simulation.optical_psf import (
    circular_pupil_defocus_psf_display,
)


def _is_axis_aligned_positive(transform: np.ndarray) -> bool:
    return bool(
        np.allclose(transform[[0, 1, 2, 2], [1, 0, 0, 1]], 0, atol=1e-12, rtol=0)
        and transform[0, 0] > 0
        and transform[1, 1] > 0
    )


def _normal_cdf_antiderivative(z: np.ndarray) -> np.ndarray:
    """F'(z)=Phi(z), with Phi the standard-normal CDF."""
    return z * ndtr(z) + np.exp(-0.5 * z * z) / sqrt(2 * pi)


def _emitter_pixel_weight(
    left: float,
    right: float,
    emitter_left: np.ndarray,
    emitter_right: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Mean Gaussian-blurred rectangle irradiance over one sensor footprint.

    All coordinates and sigma are in display-pixel units. At sigma=0 this is
    the exact rectangle overlap; positive sigma integrates the Gaussian CDF
    over the sensor pixel, rather than point-sampling the optical image.
    """
    width = right - left
    if sigma == 0:
        return (
            np.maximum(
                0, np.minimum(right, emitter_right) - np.maximum(left, emitter_left)
            )
            / width
        )
    u = _normal_cdf_antiderivative
    weight = (
        sigma
        / width
        * (
            u((emitter_right - left) / sigma)
            - u((emitter_right - right) / sigma)
            - u((emitter_left - left) / sigma)
            + u((emitter_left - right) / sigma)
        )
    )
    return np.clip(weight, 0, 1)


def _axis_emitter_matrix(
    sensor_length: int,
    display_length: int,
    scale: float,
    offset: float,
    fill_fraction: float,
    sigma_sensor: float,
    channel: int | None,
) -> csr_matrix:
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
    return csr_matrix(
        (np.asarray(data, dtype=np.float32), (np.asarray(rows), np.asarray(columns))),
        shape=(sensor_length, display_length),
        dtype=np.float32,
    )


def _spatial_axis_analytic(
    emitted_frame: np.ndarray,
    sensor_shape: tuple[int, int],
    parameters: ScreenCaptureParameters,
) -> np.ndarray:
    """Exact rectangle/Gaussian/pixel integral for front-facing display.

    Outside the finite source frame radiance is zero. Optical support extends
    beyond the sensor crop, so no artificial reflection at crop boundaries is
    imposed. The optional Airy diffraction kernel is not covered here.
    """
    transform = parameters.sensor_to_display
    if (
        not _is_axis_aligned_positive(transform)
        or _effective_diffraction_f_number(parameters) is not None
    ):
        raise ValueError(
            "analytic spatial method requires positive axis alignment and no diffraction"
        )
    sensor_h, sensor_w = sensor_shape
    display_h, display_w, _ = emitted_frame.shape
    sigma = parameters.optical_blur_sigma_sensor_pixels
    vertical = _axis_emitter_matrix(
        sensor_h,
        display_h,
        transform[1, 1],
        transform[1, 2],
        parameters.fill_fraction,
        sigma,
        None,
    )
    result = np.empty((sensor_h, sensor_w, 3), dtype=np.float32)
    for channel in range(3):
        horizontal = _axis_emitter_matrix(
            sensor_w,
            display_w,
            transform[0, 0],
            transform[0, 2],
            parameters.fill_fraction,
            sigma,
            channel if parameters.emitter_layout == "vertical_rgb" else None,
        )
        intermediate = vertical @ emitted_frame[..., channel]
        result[..., channel] = (horizontal @ intermediate.T).T
        if parameters.emitter_layout == "co_spatial_rgb_control":
            result[..., channel] /= 3
    return result


def _homography_jacobian(transform: np.ndarray, x: float, y: float) -> np.ndarray:
    denominator = transform[2, 0] * x + transform[2, 1] * y + transform[2, 2]
    if abs(denominator) < 1e-10:
        raise ValueError("projection reaches the homography horizon")
    numerator_x = transform[0, 0] * x + transform[0, 1] * y + transform[0, 2]
    numerator_y = transform[1, 0] * x + transform[1, 1] * y + transform[1, 2]
    return np.array(
        [
            [
                (transform[0, 0] * denominator - numerator_x * transform[2, 0])
                / denominator**2,
                (transform[0, 1] * denominator - numerator_x * transform[2, 1])
                / denominator**2,
            ],
            [
                (transform[1, 0] * denominator - numerator_y * transform[2, 0])
                / denominator**2,
                (transform[1, 1] * denominator - numerator_y * transform[2, 1])
                / denominator**2,
            ],
        ]
    )


def _integrate_display_raster(
    raster: np.ndarray,
    sensor_shape: tuple[int, int],
    H: np.ndarray,
    display_samples: int,
    sensor_samples: int,
) -> np.ndarray:
    """Project prefiltered display radiance into sensor pixel areas."""
    height, width = sensor_shape
    S = display_samples
    base_y, base_x = np.indices((height, width), dtype=np.float32)
    result = np.zeros((height, width, 3), dtype=np.float32)
    for sy in range(sensor_samples):
        for sx in range(sensor_samples):
            x = base_x + (sx + 0.5) / sensor_samples
            y = base_y + (sy + 0.5) / sensor_samples
            denominator = H[2, 0] * x + H[2, 1] * y + H[2, 2]
            u = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / denominator
            v = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / denominator
            result += cv2.remap(
                raster,
                (u * S - 0.5).astype(np.float32),
                (v * S - 0.5).astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
    return result / (sensor_samples * sensor_samples)


def _spatial_prefilter(
    emitted_frame: np.ndarray,
    sensor_shape: tuple[int, int],
    parameters: ScreenCaptureParameters,
) -> np.ndarray:
    """Experimental bounded perspective approximation for Gaussian optics.

    Raster cells store *area coverage* of rectangular display emitters. A
    center-Jacobian Gaussian is applied in display coordinates, then each
    sensor pixel is integrated by 4x4 quadrature. Reject geometries outside
    the probed numerical regime instead of silently returning an inaccurate
    result. Physical LCD/PSF calibration remains a separate requirement.
    """
    if _effective_diffraction_f_number(parameters) is not None:
        raise ValueError("prefilter method does not support Airy diffraction")
    sigma_sensor = parameters.optical_blur_sigma_sensor_pixels
    if sigma_sensor < 0.55:
        raise ValueError(
            "prefilter method requires Gaussian sigma >= 0.55 sensor pixels"
        )
    height, width = sensor_shape
    H = parameters.sensor_to_display
    positions = ((0, 0), (width, 0), (0, height), (width, height))
    denominators = [H[2, 0] * x + H[2, 1] * y + H[2, 2] for x, y in positions]
    if min(denominators) * max(denominators) <= 0 or min(map(abs, denominators)) < 1e-8:
        raise ValueError("prefilter projection reaches the homography horizon")
    center = _homography_jacobian(H, width / 2, height / 2)
    variation = max(
        np.linalg.norm(_homography_jacobian(H, x, y) - center) for x, y in positions
    ) / np.linalg.norm(center)
    covariance = center @ center.T
    axis_correlation = abs(covariance[0, 1]) / sqrt(covariance[0, 0] * covariance[1, 1])
    if variation > 0.05 or axis_correlation > 0.05:
        raise ValueError(
            "prefilter geometry exceeds probed Jacobian range; use fine method"
        )

    display_samples, sensor_samples = 8, 4
    display_h, display_w, _ = emitted_frame.shape
    if display_h * display_w * display_samples**2 > 8_000_000:
        raise ValueError(
            "prefilter display raster exceeds memory limit; use fine method"
        )
    raster = _display_emitter_raster(emitted_frame, parameters, display_samples)
    S = display_samples
    sigma_x = sigma_sensor * sqrt(covariance[0, 0]) * S
    sigma_y = sigma_sensor * sqrt(covariance[1, 1]) * S
    raster = gaussian_filter(raster, (sigma_y, sigma_x, 0), mode="constant", cval=0)
    return _integrate_display_raster(
        raster, sensor_shape, H, display_samples, sensor_samples
    )


def _spatial_wave_prefilter(
    emitted_frame: np.ndarray,
    sensor_shape: tuple[int, int],
    parameters: ScreenCaptureParameters,
    tile_size_sensor_pixels: int | None = None,
) -> np.ndarray:
    """Bounded fast frontoparallel wave-PSF path, before sensor sampling.

    This display-grid quadrature has been stress-tested for moderate defocus
    or diffraction-dominant exact focus, each at a limited lattice-scale/fill
    range. It is a numerical acceleration of the same ideal optical
    hypothesis, not a calibrated PSF.
    """
    H = parameters.sensor_to_display
    if parameters.defocus_psf_model != "wave":
        raise ValueError("wave_prefilter requires defocus_psf_model='wave'")
    if not _is_axis_aligned_positive(H):
        raise ValueError("wave_prefilter requires positive axis-aligned projection")
    if parameters.optical_blur_sigma_sensor_pixels != 0:
        raise ValueError(
            "wave_prefilter has not been tested with extra Gaussian optical blur"
        )
    coc_radius = thin_lens_coc_radius_sensor_pixels(parameters)
    airy_max = _airy_radius_sensor_pixels(parameters)
    airy_min = airy_max * (
        min(parameters.rgb_effective_wavelengths_nm)
        / max(parameters.rgb_effective_wavelengths_nm)
    )
    moderate_defocus = (
        coc_radius >= 0.65
        and 0.25 <= airy_min
        and airy_max <= 0.5
        and 0.7 <= H[0, 0] <= 1.1
        and 0.7 <= H[1, 1] <= 1.1
        and 0.7 <= parameters.fill_fraction <= 0.95
    )
    diffraction_dominant_focus = (
        coc_radius <= 1e-9
        and 1.35 <= airy_min
        and airy_max <= 2.4
        and 0.7 <= H[0, 0] <= 0.8
        and 0.7 <= H[1, 1] <= 0.8
        and 0.8 <= parameters.fill_fraction <= 0.9
    )
    if not (moderate_defocus or diffraction_dominant_focus):
        raise ValueError(
            "wave_prefilter CoC radius, Airy radius or projection/fill outside tested numerical range"
        )
    display_samples, sensor_samples = 12, 6
    display_h, display_w, _ = emitted_frame.shape
    kernels = tuple(
        circular_pupil_defocus_psf_display(
            parameters.lens_focal_length_mm,
            parameters.aperture_f_number,
            parameters.screen_distance_m,
            parameters.focus_distance_m,
            parameters.sensor_pixel_pitch_um,
            wavelength,
            H[0, 0],
            H[1, 1],
            display_samples,
        )
        for wavelength in parameters.rgb_effective_wavelengths_nm
    )
    if (
        tile_size_sensor_pixels is None
        and display_h * display_w * display_samples**2 <= 12_000_000
    ):
        raster = _display_emitter_raster(emitted_frame, parameters, display_samples)
        for channel, kernel in enumerate(kernels):
            raster[..., channel] = np.maximum(
                fftconvolve(raster[..., channel], kernel, mode="same"), 0
            )
        return _integrate_display_raster(
            raster, sensor_shape, H, display_samples, sensor_samples
        )

    side = tile_size_sensor_pixels if tile_size_sensor_pixels is not None else 128
    if (
        isinstance(side, (bool, np.bool_))
        or not isinstance(side, (int, np.integer))
        or side < 1
    ):
        raise ValueError("wave_prefilter tile size must be a positive integer")
    halo_x = max(kernel.shape[1] // 2 for kernel in kernels) / display_samples
    halo_y = max(kernel.shape[0] // 2 for kernel in kernels) / display_samples
    result = np.empty(sensor_shape + (3,), dtype=np.float32)
    for y0 in range(0, sensor_shape[0], side):
        y1 = min(y0 + side, sensor_shape[0])
        for x0 in range(0, sensor_shape[1], side):
            x1 = min(x0 + side, sensor_shape[1])
            display_x0 = max(0, int(floor(H[0, 0] * x0 + H[0, 2] - halo_x - 2)))
            display_x1 = min(display_w, int(ceil(H[0, 0] * x1 + H[0, 2] + halo_x + 2)))
            display_y0 = max(0, int(floor(H[1, 1] * y0 + H[1, 2] - halo_y - 2)))
            display_y1 = min(display_h, int(ceil(H[1, 1] * y1 + H[1, 2] + halo_y + 2)))
            if display_x1 <= display_x0 or display_y1 <= display_y0:
                result[y0:y1, x0:x1] = 0
                continue
            crop_h, crop_w = display_y1 - display_y0, display_x1 - display_x0
            if crop_h * crop_w * display_samples**2 > 12_000_000:
                raise ValueError(
                    "wave_prefilter tile exceeds memory limit; reduce tile_size_sensor_pixels"
                )
            raster = _display_emitter_raster(
                emitted_frame[display_y0:display_y1, display_x0:display_x1],
                parameters,
                display_samples,
            )
            for channel, kernel in enumerate(kernels):
                raster[..., channel] = np.maximum(
                    fftconvolve(raster[..., channel], kernel, mode="same"), 0
                )
            local_H = H.copy()
            local_H[0, 2] += H[0, 0] * x0 - display_x0
            local_H[1, 2] += H[1, 1] * y0 - display_y0
            result[y0:y1, x0:x1] = _integrate_display_raster(
                raster, (y1 - y0, x1 - x0), local_H, display_samples, sensor_samples
            )
    return result


def _minimum_samples_for_projection(
    transform: np.ndarray, sensor_shape: tuple[int, int], fill_fraction: float
) -> int:
    """Keep the blur/integration grid fine relative to projected emitters."""
    if _is_axis_aligned_positive(transform):
        horizontal = 24 * transform[0, 0] / fill_fraction
        vertical = 8 * transform[1, 1] / fill_fraction
    else:
        height, width = sensor_shape
        points = (
            (0, 0),
            (width, 0),
            (0, height),
            (width, height),
            (width / 2, height / 2),
        )
        denominators = np.array(
            [
                transform[2, 0] * x + transform[2, 1] * y + transform[2, 2]
                for x, y in points[:4]
            ]
        )
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
    halo = _optical_halo_sensor_pixels(parameters, oversampling)
    # Illumination outside the requested sensor crop still contributes through
    # the optical PSF. Extend every tile beyond image bounds before filtering;
    # the display frame itself supplies the physical zero-radiance boundary.
    extended_y0, extended_y1 = y0 - halo, y1 + halo
    extended_x0, extended_x1 = x0 - halo, x1 + halo
    extended_height = extended_y1 - extended_y0
    extended_width = extended_x1 - extended_x0

    fine_y, fine_x = np.indices(
        (extended_height * oversampling, extended_width * oversampling)
    )
    sensor_x = extended_x0 + (fine_x + 0.5) / oversampling
    sensor_y = extended_y0 + (fine_y + 0.5) / oversampling
    transform = parameters.sensor_to_display
    denominator = (
        transform[2, 0] * sensor_x + transform[2, 1] * sensor_y + transform[2, 2]
    )
    if np.any(np.abs(denominator) < 1e-8):
        raise ValueError("projection reaches the homography horizon")
    display_x = (
        transform[0, 0] * sensor_x + transform[0, 1] * sensor_y + transform[0, 2]
    ) / denominator
    display_y = (
        transform[1, 0] * sensor_x + transform[1, 1] * sensor_y + transform[1, 2]
    ) / denominator

    if _is_axis_aligned_positive(transform):
        fine_radiance = _screen_radiance(
            emitted_frame, parameters, display_x, display_y, oversampling
        )
    else:
        fine_radiance = _screen_radiance_projective(
            emitted_frame, parameters, display_x, display_y
        )
    if parameters.defocus_psf_model == "wave":
        fine_radiance = _wave_defocus_blur(fine_radiance, parameters, oversampling)
    else:
        fine_radiance = _diffraction_blur(fine_radiance, parameters, oversampling)
        fine_radiance = _defocus_disk_blur(
            fine_radiance, thin_lens_coc_radius_sensor_pixels(parameters) * oversampling
        )
    sigma = parameters.optical_blur_sigma_sensor_pixels * oversampling
    if sigma > 0:
        fine_radiance = _optical_blur(fine_radiance, sigma)
    averaged = fine_radiance.reshape(
        extended_height, oversampling, extended_width, oversampling, 3
    ).mean(axis=(1, 3))
    return averaged[
        y0 - extended_y0 : y1 - extended_y0, x0 - extended_x0 : x1 - extended_x0
    ]


def integrate_screen_radiance(
    emitted_frame: np.ndarray,
    sensor_shape: tuple[int, int],
    parameters: ScreenCaptureParameters,
    *,
    spatial_method: str,
    samples_per_sensor_pixel: int | None,
    tile_size_sensor_pixels: int | None,
) -> np.ndarray:
    """Select a guarded spatial integrator; return pre-exposure RGB irradiance."""
    height, width = sensor_shape
    if spatial_method not in ("fine", "analytic", "prefilter", "wave_prefilter"):
        raise ValueError(
            "spatial_method must be fine, analytic, prefilter or wave_prefilter"
        )
    if spatial_method == "analytic":
        if samples_per_sensor_pixel is not None or tile_size_sensor_pixels is not None:
            raise ValueError(
                "analytic spatial method does not use fine-grid samples or tiles"
            )
        irradiance = _spatial_axis_analytic(emitted_frame, (height, width), parameters)
    elif spatial_method == "prefilter":
        if samples_per_sensor_pixel is not None or tile_size_sensor_pixels is not None:
            raise ValueError(
                "prefilter spatial method does not use fine-grid samples or tiles"
            )
        irradiance = _spatial_prefilter(emitted_frame, (height, width), parameters)
    elif spatial_method == "wave_prefilter":
        if samples_per_sensor_pixel is not None:
            raise ValueError(
                "wave_prefilter spatial method does not use fine-grid samples"
            )
        irradiance = _spatial_wave_prefilter(
            emitted_frame, (height, width), parameters, tile_size_sensor_pixels
        )
    else:
        required_samples = _minimum_samples_for_projection(
            parameters.sensor_to_display, (height, width), parameters.fill_fraction
        )
        if samples_per_sensor_pixel is None:
            samples_per_sensor_pixel = required_samples
        if isinstance(samples_per_sensor_pixel, (bool, np.bool_)) or not isinstance(
            samples_per_sensor_pixel, (int, np.integer)
        ):
            raise ValueError("samples_per_sensor_pixel must be an integer")
        if not 2 <= samples_per_sensor_pixel <= 128:
            raise ValueError("samples_per_sensor_pixel must lie in [2, 128]")
        oversampling = int(samples_per_sensor_pixel)
        if oversampling < required_samples:
            raise ValueError(
                f"underresolved emitter: use at least {required_samples} samples per sensor pixel"
            )
        halo = _optical_halo_sensor_pixels(parameters, oversampling)
        safe_side = int(np.floor(np.sqrt(12_000_000) / oversampling)) - 2 * halo
        if safe_side < 1:
            raise ValueError("oversampling and optical blur exceed tile memory limit")
        if tile_size_sensor_pixels is not None and (
            isinstance(tile_size_sensor_pixels, (bool, np.bool_))
            or not isinstance(tile_size_sensor_pixels, (int, np.integer))
            or tile_size_sensor_pixels < 1
        ):
            raise ValueError("tile_size_sensor_pixels must be a positive integer")
        tile_size = min(
            safe_side,
            int(tile_size_sensor_pixels)
            if tile_size_sensor_pixels is not None
            else safe_side,
        )
        irradiance = np.empty((height, width, 3), dtype=np.float32)
        for y0 in range(0, height, tile_size):
            y1 = min(height, y0 + tile_size)
            for x0 in range(0, width, tile_size):
                x1 = min(width, x0 + tile_size)
                irradiance[y0:y1, x0:x1] = _spatial_tile(
                    emitted_frame,
                    (height, width),
                    parameters,
                    oversampling,
                    (y0, y1, x0, x1),
                )
    return irradiance
