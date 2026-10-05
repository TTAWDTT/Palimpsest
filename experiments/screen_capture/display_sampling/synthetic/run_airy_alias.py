"""Structural Airy/alias sweep at conditional LCD projection pitches.

This uses a synthetic uniform display, not Dragotti image pixels. It checks
the causal frequency and contrast response only; all optical settings are
illustrative rather than assigned to published photos.
"""

from palimpsest.paths import WORK_DIR

import json

import numpy as np

from palimpsest.simulation.screen_capture.capture import (
    ScreenCaptureParameters,
    render_screen_capture,
)


OUT = WORK_DIR / "screen_airy_alias_probe.json"


def main() -> None:
    frame = np.ones((128, 128, 3), dtype=np.float32)
    rows = []
    for projected_pitch in (1.25, 4 / 3, 1.5):
        display_per_sensor = 1 / projected_pitch
        transform = np.array(
            [[display_per_sensor, 0, 12], [0, display_per_sensor, 12], [0, 0, 1]],
            dtype=float,
        )
        alias_frequency = 1 - display_per_sensor
        for f_number in (None, 5.6, 11.0, 16.0):
            setup = ScreenCaptureParameters(
                sensor_to_display=transform,
                diffraction_f_number=f_number,
                sensor_pixel_pitch_um=4.30652 if f_number is not None else None,
            )
            image = render_screen_capture(frame, (96, 96), setup).irradiance
            line = image[12:-12, 12:-12, 1].mean(axis=0)
            coordinate = np.arange(line.size)
            amplitude = 2 * abs(
                np.mean(
                    (line - line.mean())
                    * np.exp(-2j * np.pi * alias_frequency * coordinate)
                )
            )
            spectrum = np.abs(np.fft.rfft(line - line.mean()))
            frequency = np.fft.rfftfreq(line.size)
            spectrum[0] = 0
            measured_peak = float(frequency[np.argmax(spectrum)])
            rows.append(
                {
                    "projected_screen_pitch_sensor_px": projected_pitch,
                    "expected_alias_cycles_per_sensor_px": alias_frequency,
                    "f_number": f_number,
                    "green_alias_amplitude": float(amplitude),
                    "green_line_mean": float(line.mean()),
                    "measured_dominant_frequency": measured_peak,
                }
            )
            print(
                f"pitch {projected_pitch:.3f}, f/{f_number}: {amplitude:.6f}, peak {measured_peak:.4f}",
                flush=True,
            )

    for pitch in (1.25, 4 / 3, 1.5):
        base = next(
            r["green_alias_amplitude"]
            for r in rows
            if r["projected_screen_pitch_sensor_px"] == pitch and r["f_number"] is None
        )
        for row in rows:
            if row["projected_screen_pitch_sensor_px"] == pitch:
                row["amplitude_fraction_of_no_diffraction"] = (
                    row["green_alias_amplitude"] / base
                )

    report = {
        "status": "synthetic process response only; no empirical camera fit",
        "wavelengths_nm": [610, 540, 460],
        "pixel_pitch_um": 4.30652,
        "rows": rows,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
