"""Prototype a fast tilted-screen display-raster prefilter approximation.

This is a numerical candidate, not a validated physical simulation path.
The optical Gaussian is pushed into display coordinates using the homography
Jacobian at image center and approximated as axis-aligned in that space.
"""

from palimpsest.paths import WORK_DIR

import json
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from palimpsest.simulation.screen_capture import (
    ScreenCaptureParameters,
    render_screen_capture,
    _homography_jacobian,
)
from experiments.screen_capture.display_sampling.synthetic.run_projective_convergence import (
    pose,
)


OUT = WORK_DIR / "screen_display_prefilter_probe.json"


def geometry_diagnostics(transform: np.ndarray, sensor_shape: tuple[int, int]) -> dict:
    sh, sw = sensor_shape
    center = _homography_jacobian(transform, sw / 2, sh / 2)
    corners = [
        _homography_jacobian(transform, x, y)
        for x, y in ((0, 0), (sw, 0), (0, sh), (sw, sh))
    ]
    variation = max(
        float(np.linalg.norm(j - center) / np.linalg.norm(center)) for j in corners
    )
    covariance = center @ center.T
    rho = abs(float(covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])))
    return {
        "jacobian_relative_variation_max": variation,
        "display_psf_axis_correlation_abs": rho,
    }


def approximate_bands(
    frame: np.ndarray,
    sensor_shape: tuple[int, int],
    transform: np.ndarray,
    fill: float,
    sigma_sensor: float,
    layout: str,
    display_samples: int,
    sensor_samples: int,
) -> np.ndarray:
    height, width, _ = frame.shape
    S, Q = display_samples, sensor_samples
    emitted = frame**2.2
    yy, xx = np.indices((height * S, width * S), dtype=np.int32)
    px = xx // S
    py = yy // S
    fx = (xx % S) / S
    fy = (yy % S) / S
    gap = (1 - fill) / 2
    y_on = S * np.maximum(0, np.minimum(fy + 1 / S, 1 - gap) - np.maximum(fy, gap))
    raster = np.empty((height * S, width * S, 3), np.float32)
    for c in range(3):
        if layout == "vertical_rgb":
            left, right = (c + gap) / 3, (c + 1 - gap) / 3
            x_on = S * np.maximum(
                0, np.minimum(fx + 1 / S, right) - np.maximum(fx, left)
            )
            factor = 1.0
        else:
            x_on = S * np.maximum(
                0, np.minimum(fx + 1 / S, 1 - gap) - np.maximum(fx, gap)
            )
            factor = 1 / 3
        raster[..., c] = emitted[py, px, c] * y_on * x_on * factor
    jac = _homography_jacobian(transform, sensor_shape[1] / 2, sensor_shape[0] / 2)
    # J maps isotropic sensor-plane Gaussian to display-plane covariance J J^T.
    # The off-diagonal term and spatial variation are intentionally omitted.
    std_x = sigma_sensor * float(np.linalg.norm(jac[0])) * S
    std_y = sigma_sensor * float(np.linalg.norm(jac[1])) * S
    if sigma_sensor:
        raster = gaussian_filter(raster, (std_y, std_x, 0), mode="constant", cval=0)
    sh, sw = sensor_shape
    base_y, base_x = np.indices((sh, sw), dtype=np.float32)
    output = np.zeros((sh, sw, 3), np.float32)
    for sy in range(Q):
        for sx in range(Q):
            x = base_x + (sx + 0.5) / Q
            y = base_y + (sy + 0.5) / Q
            den = transform[2, 0] * x + transform[2, 1] * y + transform[2, 2]
            u = (transform[0, 0] * x + transform[0, 1] * y + transform[0, 2]) / den
            v = (transform[1, 0] * x + transform[1, 1] * y + transform[1, 2]) / den
            sample = cv2.remap(
                raster,
                (u * S - 0.5).astype(np.float32),
                (v * S - 0.5).astype(np.float32),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            output += sample
    return output / (Q * Q)


def main() -> None:
    rng = np.random.default_rng(20260925)
    frame = rng.random((56, 56, 3), dtype=np.float32)
    runs = []
    for perspective in (False, True):
        for sigma in (0.0, 0.55):
            transform = pose(perspective)
            params = ScreenCaptureParameters(
                sensor_to_display=transform,
                fill_fraction=0.85,
                optical_blur_sigma_sensor_pixels=sigma,
            )
            t0 = time.perf_counter()
            reference = render_screen_capture(
                frame,
                (20, 20),
                params,
                samples_per_sensor_pixel=128,
                tile_size_sensor_pixels=20,
            ).emitter_band_irradiance
            ref_seconds = time.perf_counter() - t0
            cases = []
            for S, Q in ((8, 4), (16, 4), (16, 8)):
                t0 = time.perf_counter()
                predicted = approximate_bands(
                    frame, (20, 20), transform, 0.85, sigma, "vertical_rgb", S, Q
                )
                seconds = time.perf_counter() - t0
                inner = (slice(2, -2), slice(2, -2))
                diff = np.abs(predicted[inner] - reference[inner])
                cases.append(
                    {
                        "display_samples": S,
                        "sensor_samples": Q,
                        "seconds": seconds,
                        "interior_mean_abs_irradiance": float(diff.mean()),
                        "interior_p99_abs_irradiance": float(np.quantile(diff, 0.99)),
                        "interior_max_abs_irradiance": float(diff.max()),
                    }
                )
            runs.append(
                {
                    "perspective": perspective,
                    "sigma_sensor": sigma,
                    "fine128_seconds": ref_seconds,
                    "cases": cases,
                }
            )
            print(f"pose={perspective} sigma={sigma} {cases}", flush=True)
    report = {
        "scope": "20x20 virtual sensor, 56x56 random display, same 128-sample near-reference as convergence probe",
        "method": "display-raster prefilter with center-Jacobian diagonal Gaussian and sensor quadrature",
        "qualification": "approximation, not a validated physical model; real device and large-resolution tests pending",
        "runs": runs,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
