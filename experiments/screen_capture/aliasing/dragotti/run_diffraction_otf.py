"""Conditional ideal diffraction cutoff for a published recapture setting.

The camera's actual per-image aperture/focus and display raster are unknown.
This only checks whether our finite PSF kernel respects a known ideal-OTF
prediction at a conditional lattice projection scale.
"""

from palimpsest.paths import WORK_DIR

import json

import numpy as np

from palimpsest.simulation.shared.optical_psf import circular_pupil_defocus_psf


OUT = WORK_DIR / "dragotti_ideal_diffraction_otf_probe.json"
FOCAL_MM = 30.0
PITCH_MM = 0.00430652
LATTICE_SENSOR_FREQUENCY = 1 / 1.3272


def analytic_mtf(
    f_number: float, wavelength_nm: float, distance_m: float
) -> tuple[float, float]:
    u = distance_m * 1000
    v = FOCAL_MM * u / (u - FOCAL_MM)
    aperture_radius = FOCAL_MM / (2 * f_number)
    wavelength_mm = wavelength_nm / 1_000_000
    cutoff = 2 * aperture_radius * PITCH_MM / (wavelength_mm * v)
    normalized_frequency = LATTICE_SENSOR_FREQUENCY / cutoff
    if normalized_frequency >= 1:
        return float(cutoff), 0.0
    mtf = (2 / np.pi) * (
        np.arccos(normalized_frequency)
        - normalized_frequency * np.sqrt(1 - normalized_frequency**2)
    )
    return float(cutoff), float(mtf)


def sampled_transfer(kernel: np.ndarray, oversampling: int) -> float:
    x = np.arange(kernel.shape[1]) - kernel.shape[1] // 2
    return float(
        np.sum(
            kernel
            * np.cos(2 * np.pi * LATTICE_SENSOR_FREQUENCY * x[None, :] / oversampling)
        )
    )


def main() -> None:
    rows = []
    for f_number, distance_m in ((11, 1.4452), (13, 1.45)):
        for color, wavelength_nm in (("red", 610), ("green", 540), ("blue", 460)):
            cutoff, ideal = analytic_mtf(f_number, wavelength_nm, distance_m)
            for oversampling in (16, 24):
                kernel = circular_pupil_defocus_psf(
                    FOCAL_MM,
                    f_number,
                    distance_m,
                    distance_m,
                    PITCH_MM * 1000,
                    wavelength_nm,
                    oversampling,
                )
                rows.append(
                    {
                        "f_number": f_number,
                        "distance_m": distance_m,
                        "color": color,
                        "wavelength_nm": wavelength_nm,
                        "lattice_frequency_cycles_per_sensor_pixel": LATTICE_SENSOR_FREQUENCY,
                        "ideal_cutoff_cycles_per_sensor_pixel": cutoff,
                        "analytic_ideal_mtf": ideal,
                        "oversampling": oversampling,
                        "finite_kernel_transfer": sampled_transfer(
                            kernel, oversampling
                        ),
                        "kernel_halfwidth_sensor_pixels": kernel.shape[0]
                        // 2
                        / oversampling,
                    }
                )
    report = {
        "scope": "EOS600D f/11 text example vs f/13 thesis table; conditional projected LCD pitch 1.3272 sensor pixels per LCD pixel",
        "rows": rows,
        "qualification": "not measured optical MTF or actual per-image aperture; effective RGB wavelengths and display pitch are hypotheses",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
