"""Ideal paraxial, incoherent, circular-pupil through-focus PSF.

This is a reference optical branch for a planar screen at one distance. The
phase is integrated over the exit pupil before taking squared field magnitude;
convolving an in-focus Airy intensity with a geometric disk is a distinct
approximation. Neither branch represents measured lens aberrations.
"""

from functools import lru_cache

import numpy as np
from scipy.special import j0, roots_legendre


def _image_distance_mm(focal_mm: float, object_mm: float) -> float:
    return focal_mm * object_mm / (object_mm - focal_mm)


def circular_pupil_support_fine(
    focal_mm: float, aperture_f_number: float, screen_distance_m: float,
    focus_distance_m: float, sensor_pixel_pitch_um: float,
    wavelength_nm: float, oversampling: int,
) -> int:
    """Finite support: geometric CoC plus six first-zero Airy radii."""
    vs = _image_distance_mm(focal_mm, screen_distance_m * 1000)
    vf = _image_distance_mm(focal_mm, focus_distance_m * 1000)
    aperture_radius_mm = focal_mm / (2 * aperture_f_number)
    pitch_mm = sensor_pixel_pitch_um / 1000
    wavelength_mm = wavelength_nm / 1_000_000
    geometric_radius_fine = (aperture_radius_mm * abs(vf - vs) / vs /
                             pitch_mm * oversampling)
    airy_radius_fine = (1.21967 * wavelength_mm * vf / aperture_radius_mm /
                        (2 * pitch_mm) * oversampling)
    return int(np.ceil(geometric_radius_fine + 6 * airy_radius_fine))


def _pupil_intensity_on_sensor_steps(
    focal_mm: float, aperture_f_number: float, screen_distance_m: float,
    focus_distance_m: float, sensor_pixel_pitch_um: float,
    wavelength_nm: float, step_x_sensor_pixels: float,
    step_y_sensor_pixels: float,
) -> np.ndarray:
    """Normalized intensity PSF on a rectangular grid in sensor-pixel units."""
    values = (focal_mm, aperture_f_number, screen_distance_m, focus_distance_m,
              sensor_pixel_pitch_um, wavelength_nm,
              step_x_sensor_pixels, step_y_sensor_pixels)
    if (not np.isfinite(values).all() or min(values) <= 0 or
            screen_distance_m * 1000 <= focal_mm or
            focus_distance_m * 1000 <= focal_mm):
        raise ValueError("circular-pupil PSF requires positive physical distances, aperture and sampling")
    vs = _image_distance_mm(focal_mm, screen_distance_m * 1000)
    vf = _image_distance_mm(focal_mm, focus_distance_m * 1000)
    aperture_radius_mm = focal_mm / (2 * aperture_f_number)
    wavelength_mm = wavelength_nm / 1_000_000
    pitch_mm = sensor_pixel_pitch_um / 1000
    geometric_radius_sensor = aperture_radius_mm * abs(vf - vs) / vs / pitch_mm
    airy_radius_sensor = 1.21967 * wavelength_mm * vf / (2 * aperture_radius_mm * pitch_mm)
    support_sensor = geometric_radius_sensor + 6 * airy_radius_sensor
    half_x = int(np.ceil(support_sensor / step_x_sensor_pixels))
    half_y = int(np.ceil(support_sensor / step_y_sensor_pixels))
    if max(half_x, half_y) > 512:
        raise ValueError("wave-optical PSF support exceeds 512 fine cells; reduce oversampling")
    alpha = np.pi * aperture_radius_mm ** 2 / wavelength_mm * (1 / vf - 1 / vs)
    node_count = max(96, int(np.ceil(abs(alpha) / np.pi * 24)))
    if node_count > 2048:
        raise ValueError("wave-optical PSF phase requires more than 2048 quadrature nodes")
    gauss_nodes, gauss_weights = roots_legendre(node_count)
    rho = (gauss_nodes + 1) / 2
    weights = gauss_weights * rho * np.exp(1j * alpha * rho * rho)
    beta = 2 * np.pi * aperture_radius_mm / (wavelength_mm * vf)
    x = np.arange(-half_x, half_x + 1, dtype=np.float64) * step_x_sensor_pixels
    y = np.arange(-half_y, half_y + 1, dtype=np.float64) * step_y_sensor_pixels
    yy, xx = np.meshgrid(y, x, indexing="ij")
    radii_mm = np.hypot(xx, yy).ravel() * pitch_mm
    intensity = np.empty(radii_mm.size, dtype=np.float64)
    for begin in range(0, len(radii_mm), 2048):
        block = radii_mm[begin:begin + 2048]
        amplitude = j0(beta * block[:, None] * rho[None, :]) @ weights
        intensity[begin:begin + len(block)] = np.abs(amplitude) ** 2
    kernel = intensity.reshape(xx.shape)
    kernel /= kernel.sum()
    kernel = kernel.astype(np.float32)
    kernel.setflags(write=False)
    return kernel


@lru_cache(maxsize=32)
def circular_pupil_defocus_psf(
    focal_mm: float, aperture_f_number: float, screen_distance_m: float,
    focus_distance_m: float, sensor_pixel_pitch_um: float,
    wavelength_nm: float, oversampling: int,
) -> np.ndarray:
    """Intensity PSF at sensor fine-grid centers; immutable/cached.

    In a scalar Fresnel model the normalized pupil amplitude at image radius
    ``r`` is 2∫₀¹ ρ exp(i αρ²) J₀(βrρ) dρ, with
    α = πa²(1/v_focus − 1/v_screen)/λ and β = 2πa/(λv_focus).
    """
    if not isinstance(oversampling, int) or oversampling < 1:
        raise ValueError("oversampling must be a positive integer")
    return _pupil_intensity_on_sensor_steps(
        focal_mm, aperture_f_number, screen_distance_m,
        focus_distance_m, sensor_pixel_pitch_um, wavelength_nm,
        1 / oversampling, 1 / oversampling)


@lru_cache(maxsize=32)
def circular_pupil_defocus_psf_display(
    focal_mm: float, aperture_f_number: float, screen_distance_m: float,
    focus_distance_m: float, sensor_pixel_pitch_um: float,
    wavelength_nm: float, display_scale_x: float, display_scale_y: float,
    samples_per_display_pixel: int,
) -> np.ndarray:
    """Same sensor PSF sampled on an axis-aligned display-subpixel grid."""
    if not isinstance(samples_per_display_pixel, int) or samples_per_display_pixel < 1:
        raise ValueError("display sampling must be a positive integer")
    if (not np.isfinite((display_scale_x, display_scale_y)).all() or
            min(display_scale_x, display_scale_y) <= 0):
        raise ValueError("display projection scales must be positive finite")
    return _pupil_intensity_on_sensor_steps(
        focal_mm, aperture_f_number, screen_distance_m,
        focus_distance_m, sensor_pixel_pitch_um, wavelength_nm,
        1 / (display_scale_x * samples_per_display_pixel),
        1 / (display_scale_y * samples_per_display_pixel))
