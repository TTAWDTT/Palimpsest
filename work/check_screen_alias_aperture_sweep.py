"""Check coupled screen-grid and image-detail response of the prototype.

This is a numerical model check, not validation against a real device.
"""

import json
from pathlib import Path

import numpy as np

from origin_simulation.screen_capture import ScreenCaptureParameters, render_screen_capture


def sinusoid_amplitude(line: np.ndarray, frequency: float) -> float:
    x = np.arange(line.size, dtype=np.float64)
    design = np.stack((np.ones_like(x), np.cos(2 * np.pi * frequency * x),
                       np.sin(2 * np.pi * frequency * x)), axis=1)
    coefficients, *_ = np.linalg.lstsq(design, line, rcond=None)
    return float(np.hypot(coefficients[1], coefficients[2]))


def main() -> None:
    screen_pixels_per_sensor_pixel = 1 / 1.327151
    homography = np.array([[screen_pixels_per_sensor_pixel, 0, 30.13],
                           [0, screen_pixels_per_sensor_pixel, 30.29],
                           [0, 0, 1]])
    size = 64
    flat = np.ones((128, 128, 3), dtype=np.float32) * 0.5
    x = np.arange(128, dtype=np.float32)
    tone = 0.5 + 0.25 * np.sin(2 * np.pi * 0.2 * x)
    content = np.broadcast_to(tone[None, :, None], flat.shape).copy()
    alias_frequency = 1 - screen_pixels_per_sensor_pixel
    content_frequency = 0.2 * screen_pixels_per_sensor_pixel
    results = []
    for aperture in (None, 8.0, 11.0, 13.0):
        params = ScreenCaptureParameters(
            sensor_to_display=homography,
            fill_fraction=0.85,
            display_gamma=1.0,
            diffraction_f_number=aperture,
            sensor_pixel_pitch_um=4.30652 if aperture is not None else None,
        )
        images = {}
        for name, frame in (("flat", flat), ("content", content)):
            images[name] = render_screen_capture(frame, (size, size), params).irradiance
        # Keep away from frame edges and PSF padding. Channel response is read
        # before CFA/ISP so any change is from the optical subchain.
        region = slice(12, 52)
        row = images["flat"][region, region].mean(axis=0)
        content_row = images["content"][region, region].mean(axis=0)
        results.append({
            "f_number": aperture,
            "flat_grid_alias_amplitude_per_rgb": [
                sinusoid_amplitude(row[:, c], alias_frequency) for c in range(3)
            ],
            "flat_channel_mean_per_rgb": row.mean(axis=0).tolist(),
            "content_low_frequency_amplitude_per_rgb": [
                sinusoid_amplitude(content_row[:, c], content_frequency) for c in range(3)
            ],
        })
    record = {
        "status": "internal_numerical_check_not_real_device_validation",
        "sensor_pixels_per_screen_pixel_assumption": 1.327151,
        "screen_grid_alias_cycles_per_sensor_pixel": alias_frequency,
        "content_cycles_per_sensor_pixel": content_frequency,
        "source_radiance": "flat 0.5; sinusoidal content 0.5+0.25*sin(2*pi*0.2*display_x)",
        "sensor_shape": [size, size],
        "fit_region": [12, 52, 12, 52],
        "cases": results,
    }
    output = Path("work/screen_alias_aperture_sweep.json")
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
