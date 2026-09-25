"""Virtual physical focus intervention with coupled projection and optical PSF.

This is a structural simulation, not an estimate of any released phone/screen.
"""

import json
from pathlib import Path

import numpy as np

from origin_simulation.screen_capture import (
    ScreenCaptureParameters,
    render_screen_capture,
    thin_lens_coc_radius_sensor_pixels,
    thin_lens_frontoparallel_sensor_to_display,
)


OUT = Path("work/screen_thin_lens_focus_lattice_probe.json")
SENSOR_N = 56
DISPLAY_N = 96
FOCAL_MM = 4.0
SCREEN_M = 0.5
PITCH_UM = 4.0
APERTURE_N = 2.0
# Chosen to put the projected lattice near 0.93 display cells/sensor cell.
DISPLAY_PITCH_MM = 0.533


def camera(focus_m: float, psf_model: str = "geometric_airy") -> ScreenCaptureParameters:
    return ScreenCaptureParameters(
        sensor_to_display=thin_lens_frontoparallel_sensor_to_display(
            lens_focal_length_mm=FOCAL_MM, screen_distance_m=SCREEN_M,
            focus_distance_m=focus_m, sensor_pixel_pitch_um=PITCH_UM,
            display_pixel_pitch_mm=DISPLAY_PITCH_MM,
            display_origin_xy=(12.137, 12.219)),
        fill_fraction=.85, emitter_layout="co_spatial_rgb_control",
        display_gamma=1, lens_focal_length_mm=FOCAL_MM,
        aperture_f_number=APERTURE_N, screen_distance_m=SCREEN_M,
        focus_distance_m=focus_m, sensor_pixel_pitch_um=PITCH_UM,
        defocus_psf_model=psf_model)


def amplitude(irradiance: np.ndarray, frequency: float) -> float:
    line = irradiance[8:-8, 8:-8, 1].mean(axis=0)
    line = line - line.mean()
    taper = np.hanning(len(line))
    carrier = np.exp(-2j * np.pi * frequency * np.arange(len(line)))
    return float(2 * abs(np.sum(line * taper * carrier)) / taper.sum())


def main() -> None:
    scale = camera(SCREEN_M).sensor_to_display[0, 0]
    flat = np.ones((DISPLAY_N, DISPLAY_N, 3), dtype=np.float32) * .5
    x = np.arange(DISPLAY_N)
    content = flat + .25 * np.sin(2 * np.pi * .15 / scale * x)[None, :, None]
    models = {}
    for psf_model in ("geometric_airy", "wave"):
        data = {}
        for label, focus_m in (("focused", SCREEN_M), ("defocused", 1.0)):
            params = camera(focus_m, psf_model)
            rendered_flat = render_screen_capture(flat, (SENSOR_N, SENSOR_N), params).irradiance
            rendered_content = render_screen_capture(content, (SENSOR_N, SENSOR_N), params).irradiance
            current_scale = float(params.sensor_to_display[0, 0])
            data[label] = {
                "sensor_to_display_scale": current_scale,
                "coc_radius_sensor_pixels": thin_lens_coc_radius_sensor_pixels(params),
                "lattice_alias_cycles_per_sensor_pixel": 1 - current_scale,
                "lattice_alias_amplitude": amplitude(rendered_flat, 1 - current_scale),
                "content_increment_amplitude_at_nominal_frequency": amplitude(
                    rendered_content - rendered_flat, .15),
            }
        models[psf_model] = data
    report = {
        "qualification": "virtual planar thin lens; no phone/screen calibration and no measured optical PSF",
        "settings": {
            "focal_mm": FOCAL_MM, "aperture_f_number": APERTURE_N,
            "screen_m": SCREEN_M, "focused_distance_m": SCREEN_M,
            "defocused_distance_m": 1.0, "sensor_pitch_um": PITCH_UM,
            "display_pitch_mm": DISPLAY_PITCH_MM,
            "content_nominal_cycles_per_sensor_pixel": .15,
        },
        "models": models,
        "amplitude_ratios": {
            model: {
                "defocused_to_focused_lattice_alias": (
                    data["defocused"]["lattice_alias_amplitude"] /
                    data["focused"]["lattice_alias_amplitude"]),
                "defocused_to_focused_content_increment": (
                    data["defocused"]["content_increment_amplitude_at_nominal_frequency"] /
                    data["focused"]["content_increment_amplitude_at_nominal_frequency"]),
            }
            for model, data in models.items()
        },
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
