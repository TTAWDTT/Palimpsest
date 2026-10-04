"""Experimental fast display-domain integration of a wave-optical screen PSF.

The calibrated fine spatial renderer is the numerical reference here, not a
real camera. This candidate only handles positive axis-aligned geometry.
"""

from palimpsest.paths import WORK_DIR

import json
from dataclasses import replace
from time import perf_counter

import cv2
import numpy as np
from scipy.signal import fftconvolve
from scipy.special import j0, roots_legendre

from palimpsest.simulation.screen_capture import render_screen_capture
from experiments.screen_capture.focus_response.synthetic.run_physical_focus_display_lattice import (
    camera,
)


OUT = WORK_DIR / "wave_display_prefilter_probe.json"


def emitter_raster(frame: np.ndarray, params, samples: int) -> np.ndarray:
    h, w, _ = frame.shape
    x = np.arange(w * samples, dtype=np.int32)
    y = np.arange(h * samples, dtype=np.int32)
    px, py = x // samples, y // samples
    fx, fy = (x % samples) / samples, (y % samples) / samples
    gap = (1 - params.fill_fraction) / 2
    vy = samples * np.maximum(
        0, np.minimum(fy + 1 / samples, 1 - gap) - np.maximum(fy, gap)
    )
    result = np.empty((h * samples, w * samples, 3), dtype=np.float32)
    for channel in range(3):
        if params.emitter_layout == "vertical_rgb":
            left, right, factor = (channel + gap) / 3, (channel + 1 - gap) / 3, 1
        else:
            left, right, factor = gap, 1 - gap, 1 / 3
        vx = samples * np.maximum(
            0, np.minimum(fx + 1 / samples, right) - np.maximum(fx, left)
        )
        result[..., channel] = (
            frame[py[:, None], px[None, :], channel]
            * vy[:, None]
            * vx[None, :]
            * factor
        )
    return result


def wave_kernel_display(params, wavelength_nm: float, samples: int) -> np.ndarray:
    focal = params.lens_focal_length_mm
    aperture_radius = focal / (2 * params.aperture_f_number)
    screen_mm = params.screen_distance_m * 1000
    focus_mm = params.focus_distance_m * 1000
    vs = focal * screen_mm / (screen_mm - focal)
    vf = focal * focus_mm / (focus_mm - focal)
    wavelength_mm = wavelength_nm / 1_000_000
    pitch_mm = params.sensor_pixel_pitch_um / 1000
    scale_x = params.sensor_to_display[0, 0]
    scale_y = params.sensor_to_display[1, 1]
    radius_geom_sensor = aperture_radius * abs(vf - vs) / vs / pitch_mm
    airy_zero_sensor = 0.609835 * wavelength_mm * vf / aperture_radius / pitch_mm
    support_sensor = radius_geom_sensor + 6 * airy_zero_sensor
    hx = int(np.ceil(support_sensor * scale_x * samples))
    hy = int(np.ceil(support_sensor * scale_y * samples))
    yy, xx = np.meshgrid(np.arange(-hy, hy + 1), np.arange(-hx, hx + 1), indexing="ij")
    radius_mm = pitch_mm * np.hypot(xx / (samples * scale_x), yy / (samples * scale_y))
    alpha = np.pi * aperture_radius**2 / wavelength_mm * (1 / vf - 1 / vs)
    nodes, weights = roots_legendre(max(96, int(np.ceil(abs(alpha) / np.pi * 24))))
    rho = (nodes + 1) / 2
    weights = weights * rho * np.exp(1j * alpha * rho * rho)
    beta = 2 * np.pi * aperture_radius / (wavelength_mm * vf)
    intensity = np.empty(radius_mm.size)
    radii = radius_mm.ravel()
    for begin in range(0, len(radii), 2048):
        block = radii[begin : begin + 2048]
        amp = j0(beta * block[:, None] * rho[None, :]) @ weights
        intensity[begin : begin + len(block)] = np.abs(amp) ** 2
    kernel = intensity.reshape(radius_mm.shape)
    return (kernel / kernel.sum()).astype(np.float32)


def fast_irradiance(
    frame: np.ndarray,
    params,
    shape: tuple[int, int],
    display_samples: int = 8,
    sensor_samples: int = 4,
) -> np.ndarray:
    emitted = np.power(frame, params.display_gamma)
    raster = emitter_raster(emitted, params, display_samples)
    for channel, wavelength in enumerate(params.rgb_effective_wavelengths_nm):
        kernel = wave_kernel_display(params, wavelength, display_samples)
        raster[..., channel] = np.maximum(
            fftconvolve(raster[..., channel], kernel, mode="same"), 0
        )
    height, width = shape
    y, x = np.indices(shape, dtype=np.float32)
    result = np.zeros(shape + (3,), dtype=np.float32)
    H = params.sensor_to_display
    for sy in range(sensor_samples):
        for sx in range(sensor_samples):
            u = H[0, 0] * (x + (sx + 0.5) / sensor_samples) + H[0, 2]
            v = H[1, 1] * (y + (sy + 0.5) / sensor_samples) + H[1, 2]
            result += cv2.remap(
                raster,
                (u * display_samples - 0.5).astype(np.float32),
                (v * display_samples - 0.5).astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
    return result / sensor_samples**2


def main() -> None:
    rng = np.random.default_rng(20260925)
    frames = {
        "random": rng.uniform(0.1, 0.9, (96, 96, 3)).astype(np.float32),
        "flat": np.ones((96, 96, 3), dtype=np.float32) * 0.5,
    }
    rows = []
    for focus_m in (0.5, 1.0):
        for layout in ("vertical_rgb", "co_spatial_rgb_control"):
            params = replace(camera(focus_m, "wave"), emitter_layout=layout)
            for pattern_name, frame in frames.items():
                t0 = perf_counter()
                reference = render_screen_capture(frame, (56, 56), params).irradiance
                t1 = perf_counter()
                candidate = fast_irradiance(frame, params, (56, 56))
                t2 = perf_counter()
                residual = np.abs(reference[4:-4, 4:-4] - candidate[4:-4, 4:-4])
                rows.append(
                    {
                        "focus_m": focus_m,
                        "layout": layout,
                        "pattern": pattern_name,
                        "mae": float(residual.mean()),
                        "p99": float(np.quantile(residual, 0.99)),
                        "fine_s": t1 - t0,
                        "candidate_s": t2 - t1,
                    }
                )
    phase_checks = []
    for focus_m in (0.75, 1.0):
        for offset in ((12.137, 12.219), (12.477, 12.691)):
            params = replace(camera(focus_m, "wave"), emitter_layout="vertical_rgb")
            H = params.sensor_to_display.copy()
            H[0, 2], H[1, 2] = offset
            params = replace(params, sensor_to_display=H)
            reference = render_screen_capture(
                frames["random"], (56, 56), params
            ).irradiance
            for ds, qs in ((8, 4), (12, 6)):
                candidate = fast_irradiance(frames["random"], params, (56, 56), ds, qs)
                residual = np.abs(reference[4:-4, 4:-4] - candidate[4:-4, 4:-4])
                phase_checks.append(
                    {
                        "focus_m": focus_m,
                        "offset_xy": offset,
                        "display_samples": ds,
                        "sensor_samples": qs,
                        "mae": float(residual.mean()),
                        "p99": float(np.quantile(residual, 0.99)),
                    }
                )
    dense_checks = []
    dense_frame = frames["random"][:56, :56]
    for focus_m in (0.5, 1.0):
        params = replace(camera(focus_m, "wave"), emitter_layout="vertical_rgb")
        default = render_screen_capture(dense_frame, (20, 20), params).irradiance
        dense = render_screen_capture(
            dense_frame, (20, 20), params, samples_per_sensor_pixel=64
        ).irradiance
        for method, candidate in (
            ("fine_default", default),
            ("fast_S8_Q4", fast_irradiance(dense_frame, params, (20, 20), 8, 4)),
            ("fast_S12_Q6", fast_irradiance(dense_frame, params, (20, 20), 12, 6)),
        ):
            residual = np.abs(dense[2:-2, 2:-2] - candidate[2:-2, 2:-2])
            dense_checks.append(
                {
                    "focus_m": focus_m,
                    "method": method,
                    "mae_vs_fine64": float(residual.mean()),
                    "p99_vs_fine64": float(np.quantile(residual, 0.99)),
                }
            )
    report = {
        "scope": "virtual 96x96 display, 56x56 sensor; candidate S8/Q4, axis aligned",
        "rows": rows,
        "phase_checks": phase_checks,
        "dense_checks": dense_checks,
        "qualification": "numerical comparison to finite fine grid, not a camera validation",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
