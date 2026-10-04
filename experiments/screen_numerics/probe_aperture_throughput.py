"""Ideal fixed-geometry aperture/exposure intervention, not device calibration."""

import json
from pathlib import Path

import numpy as np

from origin_simulation.screen_capture import (
    ScreenCaptureParameters,
    render_screen_capture,
)


OUT = Path("work/aperture_throughput_probe.json")
FRAME = np.full((64, 64, 3), 0.5, dtype=np.float32)


def row(f_number: float, shutter_s: float) -> dict:
    params = ScreenCaptureParameters(
        sensor_to_display=np.array(
            [[1 / 1.3272, 0, 9.137], [0, 1 / 1.3272, 8.219], [0, 0, 1]]
        ),
        fill_fraction=0.85,
        emitter_layout="co_spatial_rgb_control",
        display_gamma=1,
        lens_focal_length_mm=30,
        aperture_f_number=f_number,
        throughput_reference_f_number=11,
        screen_distance_m=1.45,
        focus_distance_m=1.45,
        sensor_pixel_pitch_um=4.30652,
        defocus_psf_model="wave",
        electron_rate_per_unit_s=100_000,
        exposure_time_s=shutter_s,
        full_well_electrons=10_000,
        read_noise_electrons=3,
    )
    result = render_screen_capture(
        FRAME, (16, 16), params, spatial_method="wave_prefilter", seed=0
    )
    expected_electrons = result.noiseless_mosaic * params.full_well_electrons
    sigma_electrons = np.sqrt(expected_electrons + params.read_noise_electrons**2)
    return {
        "f_number": f_number,
        "shutter_s": shutter_s,
        "mean_linear_irradiance": float(result.irradiance.mean()),
        "mean_expected_electrons": float(expected_electrons.mean()),
        "mean_ideal_pixel_snr": float((expected_electrons / sigma_electrons).mean()),
        "mean_one_realized_raw": float(result.raw_mosaic.mean()),
    }


def main() -> None:
    reference = row(11, 0.01)
    stopped = row(13, 0.01)
    compensated = row(13, 0.01 * (13 / 11) ** 2)
    report = {
        "qualification": "ideal fixed-display/focus/ISO/transmission relative aperture model; no measured camera electrons, T-stop, vignetting, ISO, or exposure metadata",
        "pupil_area_ratio_f13_to_f11": (11 / 13) ** 2,
        "shot_snr_ratio_at_equal_shutter": 11 / 13,
        "rows": [reference, stopped, compensated],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
