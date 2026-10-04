"""Numerical check of fast wave prefilter for in-focus high-f-number cases.

The two apertures are paper-level Dragotti EOS 600D possibilities; no actual
published PNG has been assigned an aperture by this script.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.signal import fftconvolve

from origin_simulation.optical_psf import circular_pupil_defocus_psf_display
from origin_simulation.screen_capture import (
    ScreenCaptureParameters,
    _display_emitter_raster,
    _integrate_display_raster,
    render_screen_capture,
)


OUT = Path("work/focused_diffraction_prefilter_probe.json")


def fast(
    frame: np.ndarray,
    params: ScreenCaptureParameters,
    shape: tuple[int, int],
    S: int = 12,
    Q: int = 6,
) -> np.ndarray:
    raster = _display_emitter_raster(np.power(frame, params.display_gamma), params, S)
    H = params.sensor_to_display
    for color, wavelength in enumerate(params.rgb_effective_wavelengths_nm):
        kernel = circular_pupil_defocus_psf_display(
            params.lens_focal_length_mm,
            params.aperture_f_number,
            params.screen_distance_m,
            params.focus_distance_m,
            params.sensor_pixel_pitch_um,
            wavelength,
            H[0, 0],
            H[1, 1],
            S,
        )
        raster[..., color] = np.maximum(
            fftconvolve(raster[..., color], kernel, mode="same"), 0
        )
    return _integrate_display_raster(raster, shape, H, S, Q)


def main() -> None:
    rng = np.random.default_rng(42)
    stripe = np.empty((56, 56, 3), dtype=np.float32)
    stripe[:] = (np.arange(56)[None, :, None] % 2) * 0.8 + 0.1
    frames = {
        "random": rng.uniform(0.1, 0.9, (56, 56, 3)).astype(np.float32),
        "flat": np.ones((56, 56, 3), dtype=np.float32) * 0.5,
        "vertical_stripe": stripe,
    }
    rows = []
    dense_checks = []
    phase_checks = []
    quadrature_checks = []
    for f_number, distance_m in ((11.0, 1.4452), (13.0, 1.45)):
        params = ScreenCaptureParameters(
            sensor_to_display=np.asarray(
                [[1 / 1.3272, 0, 9.137], [0, 1 / 1.3272, 8.219], [0, 0, 1]]
            ),
            fill_fraction=0.85,
            emitter_layout="vertical_rgb",
            display_gamma=1,
            lens_focal_length_mm=30,
            aperture_f_number=f_number,
            screen_distance_m=distance_m,
            focus_distance_m=distance_m,
            sensor_pixel_pitch_um=4.30652,
            defocus_psf_model="wave",
        )
        for name, frame in frames.items():
            start = perf_counter()
            reference = render_screen_capture(frame, (20, 20), params).irradiance
            middle = perf_counter()
            candidate = fast(frame, params, (20, 20))
            end = perf_counter()
            residual = np.abs(reference[2:-2, 2:-2] - candidate[2:-2, 2:-2])
            rows.append(
                {
                    "f_number": f_number,
                    "distance_m": distance_m,
                    "pattern": name,
                    "mae": float(residual.mean()),
                    "p99": float(np.quantile(residual, 0.99)),
                    "fine_seconds": middle - start,
                    "candidate_seconds": end - middle,
                }
            )
            if name == "random" and f_number == 13:
                denser = render_screen_capture(
                    frame, (20, 20), params, samples_per_sensor_pixel=32
                ).irradiance
                for method, values in (
                    ("fine_default", reference),
                    ("fast_S12_Q6", candidate),
                ):
                    delta = np.abs(denser[2:-2, 2:-2] - values[2:-2, 2:-2])
                    dense_checks.append(
                        {
                            "method": method,
                            "mae_vs_fine32": float(delta.mean()),
                            "p99_vs_fine32": float(np.quantile(delta, 0.99)),
                        }
                    )
        for phase_x, phase_y in (
            (9.137, 8.219),
            (9.417, 8.617),
            (9.731, 8.084),
            (9.051, 8.889),
        ):
            phase_params = ScreenCaptureParameters(
                sensor_to_display=np.asarray(
                    [[1 / 1.3272, 0, phase_x], [0, 1 / 1.3272, phase_y], [0, 0, 1]]
                ),
                fill_fraction=0.85,
                emitter_layout="vertical_rgb",
                display_gamma=1,
                lens_focal_length_mm=30,
                aperture_f_number=f_number,
                screen_distance_m=distance_m,
                focus_distance_m=distance_m,
                sensor_pixel_pitch_um=4.30652,
                defocus_psf_model="wave",
            )
            for name in ("random", "vertical_stripe"):
                reference = render_screen_capture(
                    frames[name], (20, 20), phase_params
                ).irradiance
                candidate = render_screen_capture(
                    frames[name],
                    (20, 20),
                    phase_params,
                    spatial_method="wave_prefilter",
                ).irradiance
                residual = np.abs(reference[2:-2, 2:-2] - candidate[2:-2, 2:-2])
                phase_checks.append(
                    {
                        "f_number": f_number,
                        "phase_x": phase_x,
                        "phase_y": phase_y,
                        "pattern": name,
                        "mae": float(residual.mean()),
                        "p99": float(np.quantile(residual, 0.99)),
                    }
                )
                if (phase_x, phase_y, name) == (9.731, 8.084, "vertical_stripe"):
                    dense = render_screen_capture(
                        frames[name],
                        (20, 20),
                        phase_params,
                        samples_per_sensor_pixel=32,
                    ).irradiance
                    for method, values in (
                        ("fine_default", reference),
                        ("fast_S12_Q6", candidate),
                    ):
                        delta = np.abs(dense[2:-2, 2:-2] - values[2:-2, 2:-2])
                        dense_checks.append(
                            {
                                "f_number": f_number,
                                "pattern": name,
                                "method": method,
                                "mae_vs_fine32": float(delta.mean()),
                                "p99_vs_fine32": float(np.quantile(delta, 0.99)),
                            }
                        )
                    for S, Q in ((12, 6), (16, 8), (20, 10)):
                        start = perf_counter()
                        candidate = fast(frames[name], phase_params, (20, 20), S, Q)
                        elapsed = perf_counter() - start
                        residual = np.abs(reference[2:-2, 2:-2] - candidate[2:-2, 2:-2])
                        quadrature_checks.append(
                            {
                                "f_number": f_number,
                                "S": S,
                                "Q": Q,
                                "mae": float(residual.mean()),
                                "p99": float(np.quantile(residual, 0.99)),
                                "seconds": elapsed,
                            }
                        )
    report = {
        "scope": "virtual 56x56 display -> 20x20 sensor; conditional EOS600D paper settings, 1/1.3272 display/sensor scale",
        "rows": rows,
        "dense_checks": dense_checks,
        "phase_checks": phase_checks,
        "quadrature_checks": quadrature_checks,
        "qualification": "paper settings conflict f/11 vs f/13; actual per-PNG aperture, displayed raster, pose, focus, RAW/ISP unavailable; this is numerical, not Dragotti photo validation",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
