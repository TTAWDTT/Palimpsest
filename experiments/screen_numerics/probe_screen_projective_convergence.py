"""Check numerical convergence before designing a fast tilted-screen renderer."""

from palimpsest.paths import WORK_DIR

import json
import time

import numpy as np

from palimpsest.simulation.screen_capture import (
    ScreenCaptureParameters,
    render_screen_capture,
)
from palimpsest.simulation.reference import (
    projective_area_reference,
)


OUTPUT = WORK_DIR / "screen_projective_convergence.json"


def pose(perspective: bool) -> np.ndarray:
    theta = np.deg2rad(12)
    matrix = np.array(
        [
            [0.77 * np.cos(theta), -0.68 * np.sin(theta), 11.23],
            [0.77 * np.sin(theta), 0.68 * np.cos(theta), 9.47],
            [0.0007 if perspective else 0, -0.00035 if perspective else 0, 1],
        ]
    )
    return matrix


def main() -> None:
    rng = np.random.default_rng(20260925)
    frame = rng.random((56, 56, 3), dtype=np.float32)
    records = []
    for perspective in (False, True):
        for sigma in (0.0, 0.55):
            parameters = ScreenCaptureParameters(
                sensor_to_display=pose(perspective),
                fill_fraction=0.85,
                optical_blur_sigma_sensor_pixels=sigma,
            )
            runs = {}
            for samples in (32, 64, 128):
                start = time.perf_counter()
                result = render_screen_capture(
                    frame,
                    (20, 20),
                    parameters,
                    samples_per_sensor_pixel=samples,
                    tile_size_sensor_pixels=20,
                )
                runs[samples] = (result, time.perf_counter() - start)
            reference = runs[128][0].irradiance
            record = {
                "perspective": perspective,
                "gaussian_sigma_sensor_pixels": sigma,
                "transform": parameters.sensor_to_display.tolist(),
                "seconds": {str(n): float(runs[n][1]) for n in runs},
                "reference": "128x128 point quadrature per sensor pixel, not an exact physical image",
                "irradiance_error_to_128": {
                    str(n): {
                        "mean_absolute": float(
                            np.abs(runs[n][0].irradiance - reference).mean()
                        ),
                        "max_absolute": float(
                            np.abs(runs[n][0].irradiance - reference).max()
                        ),
                    }
                    for n in (32, 64)
                },
                "srgb_error_to_128": {
                    str(n): {
                        "mean_absolute": float(
                            np.abs(runs[n][0].srgb - runs[128][0].srgb).mean()
                        ),
                        "max_absolute": float(
                            np.abs(runs[n][0].srgb - runs[128][0].srgb).max()
                        ),
                    }
                    for n in (32, 64)
                },
            }
            if sigma == 0:
                start = time.perf_counter()
                area_reference = projective_area_reference(
                    np.power(frame, parameters.display_gamma),
                    (20, 20),
                    parameters.sensor_to_display,
                    parameters.fill_fraction,
                )
                record["exact_area_seconds"] = time.perf_counter() - start
                record["irradiance_error_to_exact_area"] = {
                    str(n): {
                        "mean_absolute": float(
                            np.abs(runs[n][0].irradiance - area_reference).mean()
                        ),
                        "max_absolute": float(
                            np.abs(runs[n][0].irradiance - area_reference).max()
                        ),
                    }
                    for n in (32, 64, 128)
                }
            records.append(record)
    output = {
        "scope": "20x20 virtual sensor, 56x56 random display frame, two tilted poses and two blur settings",
        "seed": 20260925,
        "runs": records,
    }
    OUTPUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
