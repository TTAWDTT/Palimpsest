"""Compare wave prefilter and fine integration away from optical crop edges."""

from palimpsest.paths import WORK_DIR

import json
from time import perf_counter

import numpy as np

from palimpsest.simulation.screen_capture import (
    ScreenCaptureParameters,
    render_screen_capture,
)


OUT = WORK_DIR / "focused_diffraction_boundary_probe.json"


def main() -> None:
    rng = np.random.default_rng(120)
    random = rng.uniform(0.1, 0.9, (72, 72, 3)).astype(np.float32)
    stripe = np.empty((72, 72, 3), dtype=np.float32)
    stripe[:] = 0.1 + 0.8 * (np.arange(72)[None, :, None] % 2)
    rows = []
    for f_number, distance_m in ((11, 1.4452), (13, 1.45)):
        for phase_x, phase_y in ((13.137, 12.219), (13.731, 12.084)):
            params = ScreenCaptureParameters(
                sensor_to_display=np.array(
                    [[1 / 1.3272, 0, phase_x], [0, 1 / 1.3272, phase_y], [0, 0, 1]]
                ),
                fill_fraction=0.85,
                emitter_layout="vertical_rgb",
                display_gamma=1,
                lens_focal_length_mm=30,
                aperture_f_number=f_number,
                screen_distance_m=distance_m,
                focus_distance_m=distance_m,
                sensor_pixel_pitch_um=4.30652,
                defocus_psf_model="wave",
            )
            for name, frame in (("random", random), ("vertical_stripe", stripe)):
                t0 = perf_counter()
                fine = render_screen_capture(frame, (48, 48), params).irradiance
                t1 = perf_counter()
                fast = render_screen_capture(
                    frame, (48, 48), params, spatial_method="wave_prefilter"
                ).irradiance
                t2 = perf_counter()
                for margin in (2, 12, 16):
                    error = np.abs(
                        fine[margin:-margin, margin:-margin]
                        - fast[margin:-margin, margin:-margin]
                    )
                    rows.append(
                        {
                            "f_number": f_number,
                            "pattern": name,
                            "phase_x": phase_x,
                            "phase_y": phase_y,
                            "margin_sensor_pixels": margin,
                            "mae": float(error.mean()),
                            "p99": float(np.quantile(error, 0.99)),
                            "max": float(error.max()),
                            "fine_seconds": t1 - t0,
                            "fast_seconds": t2 - t1,
                        }
                    )
    result = {
        "scope": "72x72 virtual display, 48x48 sensor; inspect 16-pixel central margin to avoid aperture PSF crop edges",
        "rows": rows,
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
