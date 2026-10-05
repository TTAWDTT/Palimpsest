"""Compare the analytic frontal Gaussian screen renderer with fine quadrature."""

from palimpsest.paths import WORK_DIR

import argparse
import json
import time

import numpy as np

from palimpsest.simulation.screen_capture.capture import (
    ScreenCaptureParameters,
    render_screen_capture,
)


OUTPUT = WORK_DIR / "screen_analytic_benchmark.json"


def timed(
    frame: np.ndarray, setup: ScreenCaptureParameters, method: str, repeats: int
) -> tuple[object, list[float]]:
    durations = []
    last = None
    for _ in range(repeats):
        start = time.perf_counter()
        if method == "fine":
            last = render_screen_capture(
                frame,
                (128, 128),
                setup,
                samples_per_sensor_pixel=32,
                tile_size_sensor_pixels=40,
            )
        else:
            last = render_screen_capture(
                frame, (128, 128), setup, spatial_method="analytic"
            )
        durations.append(time.perf_counter() - start)
    return last, durations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scaling-only",
        action="store_true",
        help="append analytic-only sizes to an existing trusted 128px result",
    )
    args = parser.parse_args()
    frame = np.random.default_rng(20260925).random((240, 240, 3), dtype=np.float32)
    setup = ScreenCaptureParameters(
        sensor_to_display=np.array([[0.75, 0, 30.13], [0, 0.68, 26.37], [0, 0, 1.0]]),
        fill_fraction=0.85,
        display_gamma=2.2,
        optical_blur_sigma_sensor_pixels=0.7,
    )
    if args.scaling_only:
        if not OUTPUT.exists():
            raise FileNotFoundError("full 128px benchmark must run first")
        result = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if result["setup"]["transform"] != setup.sensor_to_display.tolist():
            raise RuntimeError("stored benchmark uses a different transform")
    else:
        fine, fine_seconds = timed(frame, setup, "fine", 2)
        analytic, analytic_seconds = timed(frame, setup, "analytic", 3)
        inner = np.s_[5:-5, 5:-5]
        irradiance_error = np.abs(fine.irradiance[inner] - analytic.irradiance[inner])
        srgb_error = np.abs(fine.srgb[inner] - analytic.srgb[inner])
        result = {
            "scope": "one 128x128 virtual frontal Gaussian capture, CPU, same display frame",
            "setup": {
                "display_shape": list(frame.shape),
                "sensor_shape": [128, 128],
                "transform": setup.sensor_to_display.tolist(),
                "fill_fraction": setup.fill_fraction,
                "display_gamma": setup.display_gamma,
                "gaussian_sigma_sensor_pixels": setup.optical_blur_sigma_sensor_pixels,
                "fine_samples_per_sensor_pixel": 32,
                "fine_tile_size_sensor_pixels": 40,
            },
            "fine_seconds": fine_seconds,
            "analytic_seconds": analytic_seconds,
            "median_speedup": float(
                np.median(fine_seconds) / np.median(analytic_seconds)
            ),
            "interior_absolute_error": {
                "irradiance_mean": float(irradiance_error.mean()),
                "irradiance_max": float(irradiance_error.max()),
                "srgb_mean": float(srgb_error.mean()),
                "srgb_max": float(srgb_error.max()),
            },
            "limitation": "virtual frontal Gaussian case; optical crop boundaries differ and projective/Airy paths use fine quadrature",
        }
    scaling = {}
    for size in (256, 512):
        larger_frame = np.random.default_rng(20260925 + size).random(
            (2 * size, 2 * size, 3), dtype=np.float32
        )
        durations = []
        for _ in range(3):
            start = time.perf_counter()
            larger = render_screen_capture(
                larger_frame, (size, size), setup, spatial_method="analytic"
            )
            durations.append(time.perf_counter() - start)
        scaling[str(size)] = {
            "seconds": durations,
            "median_seconds": float(np.median(durations)),
            "output_shape": list(larger.srgb.shape),
        }
        if size == 512:
            mixed_setup = ScreenCaptureParameters(
                sensor_to_display=setup.sensor_to_display,
                fill_fraction=setup.fill_fraction,
                display_gamma=setup.display_gamma,
                optical_blur_sigma_sensor_pixels=setup.optical_blur_sigma_sensor_pixels,
                sensor_spectral_mix_rgb=(
                    (1, 0.08, 0.02),
                    (0.12, 1, 0.08),
                    (0.01, 0.14, 1),
                ),
            )
            mixed_durations = []
            for _ in range(3):
                start = time.perf_counter()
                render_screen_capture(
                    larger_frame, (size, size), mixed_setup, spatial_method="analytic"
                )
                mixed_durations.append(time.perf_counter() - start)
            scaling[str(size)]["spectral_mix_seconds"] = mixed_durations
            scaling[str(size)]["spectral_mix_median_seconds"] = float(
                np.median(mixed_durations)
            )
    result["analytic_scaling"] = scaling
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
