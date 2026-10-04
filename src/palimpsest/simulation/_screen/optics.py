"""Restricted thin-lens and pupil models in sensor-pixel units."""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
from scipy.special import j1

from palimpsest.simulation._screen.parameters import ScreenCaptureParameters
from palimpsest.simulation.optical_psf import (
    circular_pupil_defocus_psf,
    circular_pupil_support_fine,
)


def _optical_blur(fine_radiance: np.ndarray, sigma: float) -> np.ndarray:
    """Reflect-boundary Gaussian blur, using FFT for large fine grids."""
    if sigma <= 0:
        return fine_radiance
    cells = fine_radiance.shape[0] * fine_radiance.shape[1]
    if sigma < 6 or cells < 200_000 or cells > 8_000_000:
        return gaussian_filter(fine_radiance, (sigma, sigma, 0), mode="reflect")
    radius = int(4 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d /= kernel_1d.sum()
    kernel = np.outer(kernel_1d, kernel_1d).astype(np.float32)
    padded = np.pad(
        fine_radiance, ((radius, radius), (radius, radius), (0, 0)), mode="symmetric"
    )
    blurred = fftconvolve(padded, kernel[:, :, None], mode="same", axes=(0, 1))
    return blurred[radius:-radius, radius:-radius].astype(np.float32)


def _airy_radius_sensor_pixels(parameters: ScreenCaptureParameters) -> float:
    """Largest first-zero radius for the assumed RGB monochromatic bands."""
    f_number = _effective_diffraction_f_number(parameters)
    if f_number is None:
        return 0.0
    return (
        1.22
        * max(parameters.rgb_effective_wavelengths_nm)
        * 1e-3
        * f_number
        / parameters.sensor_pixel_pitch_um
    )


def _effective_diffraction_f_number(
    parameters: ScreenCaptureParameters,
) -> float | None:
    """A thin-lens aperture also gives an Airy limit; legacy diffraction stays available."""
    return (
        parameters.aperture_f_number
        if parameters.aperture_f_number is not None
        else parameters.diffraction_f_number
    )


def thin_lens_coc_radius_sensor_pixels(parameters: ScreenCaptureParameters) -> float:
    """Paraxial circle-of-confusion radius for one planar screen distance.

    This is the geometric disk radius, not the complete optical PSF. The
    sensor is placed at the image distance for ``focus_distance_m``.
    """
    if parameters.lens_focal_length_mm is None:
        return 0.0
    focal_mm = parameters.lens_focal_length_mm
    focused_object_mm = parameters.focus_distance_m * 1000
    screen_object_mm = parameters.screen_distance_m * 1000
    focused_image_mm = focal_mm * focused_object_mm / (focused_object_mm - focal_mm)
    screen_image_mm = focal_mm * screen_object_mm / (screen_object_mm - focal_mm)
    aperture_diameter_mm = focal_mm / parameters.aperture_f_number
    coc_diameter_mm = (
        aperture_diameter_mm * abs(focused_image_mm - screen_image_mm) / screen_image_mm
    )
    return float(coc_diameter_mm * 1000 / parameters.sensor_pixel_pitch_um / 2)


def thin_lens_frontoparallel_sensor_to_display(
    *,
    lens_focal_length_mm: float,
    screen_distance_m: float,
    focus_distance_m: float,
    sensor_pixel_pitch_um: float,
    display_pixel_pitch_mm: float,
    display_origin_xy: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Consistent frontoparallel display map for a paraxial thin lens.

    The chief ray intersects the *actual* sensor plane at ``v_focus``. Thus a
    focus change slightly changes apparent display scale (focus breathing)
    as well as the circle of confusion. ``display_origin_xy`` encodes the
    chosen crop/optical-axis offset, not a measured camera pose.
    """
    origin = np.asarray(display_origin_xy, dtype=np.float64)
    values = np.asarray(
        (
            lens_focal_length_mm,
            screen_distance_m,
            focus_distance_m,
            sensor_pixel_pitch_um,
            display_pixel_pitch_mm,
        ),
        dtype=np.float64,
    )
    if (
        origin.shape != (2,)
        or not np.isfinite(origin).all()
        or not np.isfinite(values).all()
        or np.min(values) <= 0
    ):
        raise ValueError(
            "thin-lens frontoparallel geometry requires finite positive scales"
        )
    focal_mm = lens_focal_length_mm
    screen_mm = screen_distance_m * 1000
    focus_mm = focus_distance_m * 1000
    if min(screen_mm, focus_mm) <= focal_mm:
        raise ValueError("screen and focus distances must exceed lens focal length")
    sensor_plane_mm = focal_mm * focus_mm / (focus_mm - focal_mm)
    scale = (
        sensor_pixel_pitch_um
        * 1e-3
        * screen_mm
        / (sensor_plane_mm * display_pixel_pitch_mm)
    )
    return np.asarray(
        [[scale, 0, origin[0]], [0, scale, origin[1]], [0, 0, 1]], dtype=np.float64
    )


def _defocus_disk_blur(fine_radiance: np.ndarray, radius_fine: float) -> np.ndarray:
    """Apply a normalized geometric blur disk before pixel-area integration."""
    if radius_fine <= 1e-12:
        return fine_radiance
    radius = int(np.ceil(radius_fine + 0.75))
    if radius > 256:
        raise ValueError(
            "defocus disk support exceeds 256 fine cells; reduce oversampling"
        )
    grid = np.arange(-radius, radius + 1, dtype=np.float32)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    kernel = np.zeros_like(xx)
    # Fractional coverage of each fine cell keeps tiny disks continuous.
    for dy in (-0.375, -0.125, 0.125, 0.375):
        for dx in (-0.375, -0.125, 0.125, 0.375):
            kernel += (xx + dx) ** 2 + (yy + dy) ** 2 <= radius_fine**2
    if kernel.sum() <= 0:
        kernel[radius, radius] = 1
    kernel /= kernel.sum()
    blurred = np.empty_like(fine_radiance)
    for channel in range(3):
        padded = np.pad(fine_radiance[..., channel], radius, mode="symmetric")
        convolution = fftconvolve(padded, kernel, mode="same")
        blurred[..., channel] = np.maximum(
            convolution[radius:-radius, radius:-radius], 0
        )
    return blurred


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


def _diffraction_blur(
    fine_radiance: np.ndarray, parameters: ScreenCaptureParameters, oversampling: int
) -> np.ndarray:
    f_number = _effective_diffraction_f_number(parameters)
    if f_number is None:
        return fine_radiance
    blurred = np.empty_like(fine_radiance)
    for channel, wavelength_nm in enumerate(parameters.rgb_effective_wavelengths_nm):
        first_zero_fine = (
            1.22
            * wavelength_nm
            * 1e-3
            * f_number
            / parameters.sensor_pixel_pitch_um
            * oversampling
        )
        radius = int(np.ceil(4 * first_zero_fine))
        if radius > 512:
            raise ValueError(
                "diffraction PSF support exceeds 512 fine cells; reduce oversampling"
            )
        kernel = _airy_kernel(radius, first_zero_fine)
        padded = np.pad(fine_radiance[..., channel], radius, mode="symmetric")
        convolution = fftconvolve(padded, kernel, mode="same")
        blurred[..., channel] = np.maximum(
            convolution[radius:-radius, radius:-radius], 0
        )
    return blurred


def _wave_defocus_blur(
    fine_radiance: np.ndarray, parameters: ScreenCaptureParameters, oversampling: int
) -> np.ndarray:
    """Apply wavelength-specific through-focus pupil intensity before sampling."""
    blurred = np.empty_like(fine_radiance)
    for channel, wavelength_nm in enumerate(parameters.rgb_effective_wavelengths_nm):
        kernel = circular_pupil_defocus_psf(
            parameters.lens_focal_length_mm,
            parameters.aperture_f_number,
            parameters.screen_distance_m,
            parameters.focus_distance_m,
            parameters.sensor_pixel_pitch_um,
            wavelength_nm,
            oversampling,
        )
        radius = kernel.shape[0] // 2
        padded = np.pad(fine_radiance[..., channel], radius, mode="symmetric")
        convolution = fftconvolve(padded, kernel, mode="same")
        blurred[..., channel] = np.maximum(
            convolution[radius:-radius, radius:-radius], 0
        )
    return blurred


def _optical_halo_sensor_pixels(
    parameters: ScreenCaptureParameters, oversampling: int
) -> int:
    gaussian = int(np.ceil(4 * parameters.optical_blur_sigma_sensor_pixels)) + (
        1 if parameters.optical_blur_sigma_sensor_pixels > 0 else 0
    )
    if parameters.defocus_psf_model == "wave":
        support_fine = max(
            circular_pupil_support_fine(
                parameters.lens_focal_length_mm,
                parameters.aperture_f_number,
                parameters.screen_distance_m,
                parameters.focus_distance_m,
                parameters.sensor_pixel_pitch_um,
                wavelength,
                oversampling,
            )
            for wavelength in parameters.rgb_effective_wavelengths_nm
        )
        return gaussian + int(np.ceil(support_fine / oversampling)) + 1
    disk_radius = thin_lens_coc_radius_sensor_pixels(parameters)
    return (
        gaussian
        + int(np.ceil(4 * _airy_radius_sensor_pixels(parameters)))
        + (int(np.ceil(disk_radius)) + 1 if disk_radius > 0 else 0)
    )
