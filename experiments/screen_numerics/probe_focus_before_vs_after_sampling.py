"""Counterexample: pre-sensor optical blur vs post-capture Gaussian blur.

All parameters are virtual and chosen to expose an alias near 0.07 cycles per
sensor pixel, similar in scale to two released FDNet diagnostic patches.
"""

from palimpsest.paths import WORK_DIR

import json

import cv2
import numpy as np

from palimpsest.simulation.screen_capture import (
    ScreenCaptureParameters,
    render_screen_capture,
)


OUT = WORK_DIR / "screen_focus_before_vs_after_sampling.json"
N_DISPLAY = 200
N_SENSOR = 128
SCALE = 0.93
LATTICE_ALIAS = 1 - SCALE
CONTENT_FREQ_SENSOR = 0.15


def setup(sigma: float) -> ScreenCaptureParameters:
    return ScreenCaptureParameters(
        sensor_to_display=np.asarray(
            [[SCALE, 0, 12.137], [0, SCALE, 11.219], [0, 0, 1]]
        ),
        fill_fraction=0.85,
        emitter_layout="co_spatial_rgb_control",
        display_gamma=1,
        optical_blur_sigma_sensor_pixels=sigma,
    )


def amplitude(image: np.ndarray, frequency: float) -> float:
    line = image[16:-16, 16:-16, 1].mean(axis=0)
    line = line - line.mean()
    window = np.hanning(len(line))
    waves = np.exp(-2j * np.pi * frequency * np.arange(len(line)))
    return float(2 * abs(np.sum(line * window * waves)) / window.sum())


def render(frame: np.ndarray, sigma: float) -> np.ndarray:
    return render_screen_capture(
        frame, (N_SENSOR, N_SENSOR), setup(sigma), spatial_method="analytic"
    ).irradiance


def main() -> None:
    flat = np.ones((N_DISPLAY, N_DISPLAY, 3), dtype=np.float32) * 0.5
    x = np.arange(N_DISPLAY)
    content = (
        flat + 0.25 * np.sin(2 * np.pi * CONTENT_FREQ_SENSOR / SCALE * x)[None, :, None]
    )
    focus_flat, defocus_flat = render(flat, 0), render(flat, 0.55)
    focus_content, defocus_content = render(content, 0), render(content, 0.55)
    target_alias = amplitude(defocus_flat, LATTICE_ALIAS)
    controls = []
    for sigma in np.arange(0, 12.01, 0.25):
        if sigma:
            post_flat = cv2.GaussianBlur(focus_flat, (0, 0), float(sigma))
            post_content = cv2.GaussianBlur(focus_content, (0, 0), float(sigma))
        else:
            post_flat, post_content = focus_flat, focus_content
        controls.append(
            {
                "postcapture_sigma_sensor_pixels": float(sigma),
                "alias_amplitude": amplitude(post_flat, LATTICE_ALIAS),
                "content_amplitude": amplitude(post_content, CONTENT_FREQ_SENSOR),
            }
        )
    matched = min(controls, key=lambda row: abs(row["alias_amplitude"] - target_alias))
    report = {
        "hypothesis": "optical lowpass before sampling cannot generally be replaced by Gaussian blur after aliasing",
        "virtual_settings": {
            "sensor_to_display_scale": SCALE,
            "lattice_alias_cycles_per_sensor_pixel": LATTICE_ALIAS,
            "content_cycles_per_sensor_pixel": CONTENT_FREQ_SENSOR,
            "optical_sigma_sensor_pixels": 0.55,
            "method": "axis-aligned analytic rectangle+Gaussian area integral",
        },
        "focused_alias_amplitude": amplitude(focus_flat, LATTICE_ALIAS),
        "optically_defocused_alias_amplitude": target_alias,
        "focused_content_amplitude": amplitude(focus_content, CONTENT_FREQ_SENSOR),
        "optically_defocused_content_amplitude": amplitude(
            defocus_content, CONTENT_FREQ_SENSOR
        ),
        "matched_postcapture_gaussian": matched,
        "controls": controls,
        "qualification": "virtual numerical mechanism test, not calibration of FDNet phones or screen",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "controls"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
