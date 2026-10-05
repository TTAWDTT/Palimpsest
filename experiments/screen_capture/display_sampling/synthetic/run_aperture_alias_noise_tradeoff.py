"""Virtual aperture intervention: presampling LCD alias and photon budget."""

from palimpsest.paths import WORK_DIR

import json
from time import perf_counter

import numpy as np

from palimpsest.simulation.screen_capture import (
    ScreenCaptureParameters,
    render_screen_capture,
)


OUT = WORK_DIR / "aperture_alias_noise_tradeoff.json"
H_SCALE = 1 / 1.3272
ALIAS_F = 1 - H_SCALE
FRAME = np.full((64, 192, 3), 0.5, dtype=np.float32)


def alias_amplitude(signal: np.ndarray) -> float:
    x = np.arange(16, 112, dtype=np.float64)
    values = signal[16:112]
    model = np.column_stack(
        (
            np.ones_like(x),
            x - x.mean(),
            np.cos(2 * np.pi * ALIAS_F * x),
            np.sin(2 * np.pi * ALIAS_F * x),
        )
    )
    weights = np.linalg.lstsq(model, values, rcond=None)[0]
    return float(np.hypot(weights[2], weights[3]) / weights[0])


def main() -> None:
    rows = []
    for n in (8.0, 11.0, 13.0):
        params = ScreenCaptureParameters(
            sensor_to_display=np.array(
                [[H_SCALE, 0, 30.137], [0, H_SCALE, 12.219], [0, 0, 1]]
            ),
            fill_fraction=0.85,
            emitter_layout="vertical_rgb",
            display_gamma=1,
            lens_focal_length_mm=30,
            aperture_f_number=n,
            throughput_reference_f_number=11,
            screen_distance_m=1.45,
            focus_distance_m=1.45,
            sensor_pixel_pitch_um=4.30652,
            defocus_psf_model="wave",
            electron_rate_per_unit_s=100_000,
            exposure_time_s=0.01,
            full_well_electrons=10_000,
            read_noise_electrons=3,
        )
        start = perf_counter()
        result = render_screen_capture(FRAME, (16, 128), params)
        end = perf_counter()
        signals = result.irradiance[4:12].mean(axis=0)
        rows.append(
            {
                "f_number": n,
                "seconds_fine_render": end - start,
                "relative_pupil_area_to_f11": (11 / n) ** 2,
                "mean_expected_electrons": float(
                    (result.noiseless_mosaic * 10000).mean()
                ),
                "relative_alias_amplitude_rgb": [
                    alias_amplitude(signals[:, c]) for c in range(3)
                ],
            }
        )
    report = {
        "scope": "virtual flat LCD, uniform distance 1.45 m, 30 mm ideal circular pupil, 1/1.3272 display pixels/sensor pixel, 16x128 fine sensor grid",
        "alias_frequency_cycles_per_sensor_pixel": ALIAS_F,
        "rows": rows,
        "qualification": "model intervention only; RGB wavelengths, display lattice and photon rate not calibrated to actual camera",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
