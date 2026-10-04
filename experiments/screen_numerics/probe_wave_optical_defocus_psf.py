"""Compare a scalar circular-pupil PSF to the current Airy*disk approximation.

Paraxial incoherent imaging, spatially invariant frontoparallel plane, no lens
aberrations or measured camera parameters.  This is a numerical reference,
not a device calibration or a model-selection fit to photographs.
"""

from palimpsest.paths import WORK_DIR

import json

import numpy as np
from scipy.signal import fftconvolve
from scipy.special import j0, j1, roots_legendre


OUT = WORK_DIR / "wave_optical_defocus_psf_probe.json"
PITCH_MM = 0.004
FOCAL_MM = 4.0
SCREEN_MM = 500.0
APERTURE_N = 2.0
WAVELENGTH_MM = 0.00054
OVERSAMPLING = 12


def image_distance_mm(object_distance_mm: float) -> float:
    return FOCAL_MM * object_distance_mm / (object_distance_mm - FOCAL_MM)


def normalized_wave_psf(
    focus_mm: float, support_fine: int, quadrature_nodes: int
) -> tuple[np.ndarray, float, float]:
    vs = image_distance_mm(SCREEN_MM)
    vf = image_distance_mm(focus_mm)
    aperture_radius = FOCAL_MM / (2 * APERTURE_N)
    alpha = np.pi * aperture_radius**2 / WAVELENGTH_MM * (1 / vf - 1 / vs)
    nodes, weights = roots_legendre(quadrature_nodes)
    rho = (nodes + 1) / 2
    weights = weights / 2
    radial_weights = 2 * rho * weights * np.exp(1j * alpha * rho * rho)
    x = np.arange(-support_fine, support_fine + 1)
    yy, xx = np.meshgrid(x, x, indexing="ij")
    r_mm = np.hypot(xx, yy).ravel() * PITCH_MM / OVERSAMPLING
    beta = 2 * np.pi * aperture_radius / (WAVELENGTH_MM * vf)
    intensities = np.empty(r_mm.size, dtype=np.float64)
    for begin in range(0, r_mm.size, 2048):
        radii = r_mm[begin : begin + 2048]
        amplitude = j0(beta * radii[:, None] * rho[None, :]) @ radial_weights
        intensities[begin : begin + len(radii)] = np.abs(amplitude) ** 2
    kernel = intensities.reshape(xx.shape)
    return kernel / kernel.sum(), float(alpha), float(vf)


def approximate_psf(focus_mm: float, support_fine: int) -> np.ndarray:
    vs = image_distance_mm(SCREEN_MM)
    vf = image_distance_mm(focus_mm)
    aperture_radius = FOCAL_MM / (2 * APERTURE_N)
    radius_geom_mm = aperture_radius * abs(vf - vs) / vs
    radius_geom_fine = radius_geom_mm / PITCH_MM * OVERSAMPLING
    x = np.arange(-support_fine, support_fine + 1)
    yy, xx = np.meshgrid(x, x, indexing="ij")
    airy_arg = (
        np.pi
        * np.hypot(xx, yy)
        / (1.22 * WAVELENGTH_MM * APERTURE_N / PITCH_MM * OVERSAMPLING)
    )
    airy_arg *= 1.22
    airy_amplitude = np.ones_like(airy_arg)
    np.divide(2 * j1(airy_arg), airy_arg, out=airy_amplitude, where=airy_arg != 0)
    airy = airy_amplitude**2
    airy /= airy.sum()
    if radius_geom_fine <= 1e-10:
        return airy
    disk = np.zeros_like(airy)
    for dy in (-0.375, -0.125, 0.125, 0.375):
        for dx in (-0.375, -0.125, 0.125, 0.375):
            disk += (xx + dx) ** 2 + (yy + dy) ** 2 <= radius_geom_fine**2
    disk /= disk.sum()
    combined = fftconvolve(airy, disk, mode="same")
    combined = np.maximum(combined, 0)
    return combined / combined.sum()


def transfer(kernel: np.ndarray, sensor_frequency: float) -> float:
    x = np.arange(kernel.shape[1]) - kernel.shape[1] // 2
    return float(
        np.sum(
            kernel * np.cos(2 * np.pi * sensor_frequency * x[None, :] / OVERSAMPLING)
        )
    )


def main() -> None:
    rows = []
    for focus_mm in (SCREEN_MM, 750.0, 1000.0, 2000.0):
        vs = image_distance_mm(SCREEN_MM)
        vf = image_distance_mm(focus_mm)
        aperture_radius = FOCAL_MM / (2 * APERTURE_N)
        geom_fine = aperture_radius * abs(vf - vs) / vs / PITCH_MM * OVERSAMPLING
        airy_fine = 1.22 * WAVELENGTH_MM * APERTURE_N / PITCH_MM * OVERSAMPLING
        support = int(np.ceil(geom_fine + 6 * airy_fine))
        alpha_abs = abs(np.pi * aperture_radius**2 / WAVELENGTH_MM * (1 / vf - 1 / vs))
        nodes = max(96, int(np.ceil(alpha_abs / np.pi * 24)))
        wave, alpha, _ = normalized_wave_psf(focus_mm, support, nodes)
        approximation = approximate_psf(focus_mm, support)
        midpoint = support
        xx, yy = np.meshgrid(
            np.arange(-support, support + 1),
            np.arange(-support, support + 1),
            indexing="xy",
        )
        center = xx * xx + yy * yy <= (OVERSAMPLING / 2) ** 2
        rows.append(
            {
                "focus_distance_mm": focus_mm,
                "defocus_pupil_phase_edge_radians": alpha,
                "geometric_coc_radius_sensor_pixels": geom_fine / OVERSAMPLING,
                "quadrature_nodes": nodes,
                "kernel_support_sensor_pixels": support / OVERSAMPLING,
                "wave_center_energy_half_sensor_pixel": float(wave[center].sum()),
                "approx_center_energy_half_sensor_pixel": float(
                    approximation[center].sum()
                ),
                "kernel_total_variation_distance": float(
                    np.abs(wave - approximation).sum() / 2
                ),
                "wave_transfer_0p15": transfer(wave, 0.15),
                "approx_transfer_0p15": transfer(approximation, 0.15),
                "wave_transfer_0p93": transfer(wave, 0.93),
                "approx_transfer_0p93": transfer(approximation, 0.93),
                "center_wave_intensity": float(wave[midpoint, midpoint]),
                "center_approx_intensity": float(approximation[midpoint, midpoint]),
            }
        )
    # Check that the key moderate-defocus discrepancy is not quadrature or
    # finite-window noise before interpreting it as a model difference.
    reference_focus = 750.0
    vs = image_distance_mm(SCREEN_MM)
    vf = image_distance_mm(reference_focus)
    aperture_radius = FOCAL_MM / (2 * APERTURE_N)
    geom_fine = aperture_radius * abs(vf - vs) / vs / PITCH_MM * OVERSAMPLING
    airy_fine = 1.22 * WAVELENGTH_MM * APERTURE_N / PITCH_MM * OVERSAMPLING
    convergence = []
    for extra_airy, nodes in ((6, 96), (6, 160), (9, 96), (9, 160)):
        support = int(np.ceil(geom_fine + extra_airy * airy_fine))
        kernel, _, _ = normalized_wave_psf(reference_focus, support, nodes)
        convergence.append(
            {
                "focus_distance_mm": reference_focus,
                "airy_radii_beyond_coc": extra_airy,
                "quadrature_nodes": nodes,
                "wave_transfer_0p93": transfer(kernel, 0.93),
                "wave_transfer_0p15": transfer(kernel, 0.15),
            }
        )
    report = {
        "assumption": "paraxial incoherent circular pupil; wave amplitude is radial Fresnel pupil integral",
        "parameters": {
            "focal_mm": FOCAL_MM,
            "aperture_f_number": APERTURE_N,
            "screen_distance_mm": SCREEN_MM,
            "sensor_pitch_mm": PITCH_MM,
            "wavelength_mm": WAVELENGTH_MM,
            "oversampling": OVERSAMPLING,
        },
        "rows": rows,
        "convergence": convergence,
        "qualification": "wave PSF is an idealized numerical reference, not a measured device PSF; finite support normalizes truncated tails",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
