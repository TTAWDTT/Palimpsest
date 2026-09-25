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


@lru_cache(maxsize=32)
def circular_pupil_defocus_psf(
    focal_mm: float, aperture_f_number: float, screen_distance_m: float,
    focus_distance_m: float, sensor_pixel_pitch_um: float,
    wavelength_nm: float, oversampling: int,
) -> np.ndarray:
    """Return a normalized sampled intensity PSF for one wavelength.

    In a scalar Fresnel model, the normalized pupil amplitude at image radius
    ``r`` is 2∫₀¹ ρ exp(i αρ²) J₀(βrρ) dρ, with
    α = πa²(1/v_focus − 1/v_screen)/λ and β = 2πa/(λv_focus).
    The sign of α has no effect on this rotationally symmetric intensity.
    The return array is immutable so a cached kernel cannot be modified.
    """
    values = (focal_mm, aperture_f_number, screen_distance_m, focus_distance_m,
              sensor_pixel_pitch_um, wavelength_nm)
    if (not np.isfinite(values).all() or min(values) <= 0 or
            screen_distance_m * 1000 <= focal_mm or
            focus_distance_m * 1000 <= focal_mm or
            not isinstance(oversampling, int) or oversampling < 1):
        raise ValueError("circular-pupil PSF requires positive physical distances, aperture and sampling")
    support = circular_pupil_support_fine(*values, oversampling)
    if support > 512:
        raise ValueError("wave-optical PSF support exceeds 512 fine cells; reduce oversampling")
    vs = _image_distance_mm(focal_mm, screen_distance_m * 1000)
    vf = _image_distance_mm(focal_mm, focus_distance_m * 1000)
    aperture_radius_mm = focal_mm / (2 * aperture_f_number)
    wavelength_mm = wavelength_nm / 1_000_000
    pitch_mm = sensor_pixel_pitch_um / 1000
    alpha = np.pi * aperture_radius_mm ** 2 / wavelength_mm * (1 / vf - 1 / vs)
    node_count = max(96, int(np.ceil(abs(alpha) / np.pi * 24)))
    if node_count > 2048:
        raise ValueError("wave-optical PSF phase requires more than 2048 quadrature nodes")
    gauss_nodes, gauss_weights = roots_legendre(node_count)
    rho = (gauss_nodes + 1) / 2
    weights = gauss_weights * rho * np.exp(1j * alpha * rho * rho)
    beta = 2 * np.pi * aperture_radius_mm / (wavelength_mm * vf)
    grid = np.arange(-support, support + 1, dtype=np.float64)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    radii_mm = np.hypot(xx, yy).ravel() * pitch_mm / oversampling
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
