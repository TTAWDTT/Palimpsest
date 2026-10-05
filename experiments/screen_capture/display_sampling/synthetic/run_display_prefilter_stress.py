"""Stress the fast display-prefilter candidate beyond near-front-facing poses."""

from palimpsest.paths import WORK_DIR

import json
import time

import numpy as np

from palimpsest.simulation.screen_capture.capture import (
    ScreenCaptureParameters,
    render_screen_capture,
)
from experiments.screen_capture.display_sampling.synthetic.run_display_prefilter import (
    approximate_bands,
)


OUT = WORK_DIR / "screen_display_prefilter_stress.json"


def pose(degrees: float, perspective: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    return np.asarray(
        [
            [0.77 * np.cos(theta), -0.68 * np.sin(theta), 11.23],
            [0.77 * np.sin(theta), 0.68 * np.cos(theta), 9.47],
            [perspective, -perspective / 2, 1],
        ],
        dtype=np.float64,
    )


def main() -> None:
    rng = np.random.default_rng(20260925)
    frame = rng.random((56, 56, 3), dtype=np.float32)
    runs = []
    for degrees, perspective in ((0, 0), (12, 0.0007), (25, 0.002), (35, 0.004)):
        for sigma in (0, 0.55, 0.8):
            H = pose(degrees, perspective)
            params = ScreenCaptureParameters(
                sensor_to_display=H,
                fill_fraction=0.85,
                optical_blur_sigma_sensor_pixels=sigma,
            )
            start = time.perf_counter()
            reference = render_screen_capture(
                frame,
                (20, 20),
                params,
                samples_per_sensor_pixel=128,
                tile_size_sensor_pixels=20,
            ).emitter_band_irradiance
            ref_seconds = time.perf_counter() - start
            cases = []
            for S, Q in ((8, 4), (16, 8)):
                start = time.perf_counter()
                fast = approximate_bands(
                    frame, (20, 20), H, 0.85, sigma, "vertical_rgb", S, Q
                )
                seconds = time.perf_counter() - start
                diff = np.abs(fast[2:-2, 2:-2] - reference[2:-2, 2:-2])
                cases.append(
                    {
                        "display_samples": S,
                        "sensor_samples": Q,
                        "seconds": seconds,
                        "mean_abs_irradiance": float(diff.mean()),
                        "p99_abs_irradiance": float(np.quantile(diff, 0.99)),
                        "max_abs_irradiance": float(diff.max()),
                    }
                )
            runs.append(
                {
                    "rotation_deg": degrees,
                    "perspective_x": perspective,
                    "sigma_sensor": sigma,
                    "fine128_seconds": ref_seconds,
                    "cases": cases,
                }
            )
            print(
                f"angle {degrees} p={perspective} sigma={sigma} errors "
                f"{[round(x['mean_abs_irradiance'], 5) for x in cases]}",
                flush=True,
            )
    OUT.write_text(
        json.dumps(
            {
                "scope": "20x20 virtual sensor, 56x56 random source, single random seed",
                "reference": "fine128 point quadrature, not exact truth",
                "qualification": "stress probe only; no real device parameter verification",
                "runs": runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
