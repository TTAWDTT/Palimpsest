"""Virtual larger-frame wave prefilter tile seam and runtime probe."""

from palimpsest.paths import WORK_DIR

import json
from time import perf_counter

import numpy as np

from palimpsest.simulation.screen_capture.capture import (
    ScreenCaptureParameters,
    render_screen_capture,
)


OUT = WORK_DIR / "wave_prefilter_tiled_benchmark.json"


def main() -> None:
    frame = (
        np.random.default_rng(721).uniform(0.1, 0.9, (512, 512, 3)).astype(np.float32)
    )
    params = ScreenCaptureParameters(
        sensor_to_display=np.array(
            [[1 / 1.3272, 0, 13.731], [0, 1 / 1.3272, 12.084], [0, 0, 1]]
        ),
        fill_fraction=0.85,
        emitter_layout="vertical_rgb",
        display_gamma=1,
        lens_focal_length_mm=30,
        aperture_f_number=13,
        screen_distance_m=1.45,
        focus_distance_m=1.45,
        sensor_pixel_pitch_um=4.30652,
        defocus_psf_model="wave",
    )
    t0 = perf_counter()
    full = render_screen_capture(
        frame, (512, 512), params, spatial_method="wave_prefilter"
    )
    t1 = perf_counter()
    reference = render_screen_capture(
        frame[:192, :192], (160, 160), params, spatial_method="wave_prefilter"
    )
    t2 = perf_counter()
    residual = np.abs(full.irradiance[:160, :160] - reference.irradiance)
    seam = residual[126:130, :160]
    result = {
        "scope": "virtual 512x512 display -> 512x512 sensor; f/13 exact-focus ideal circular pupil; auto tile side 128",
        "auto_tile_seconds": t1 - t0,
        "160x160_full_raster_reference_seconds": t2 - t1,
        "overlap_mae": float(residual.mean()),
        "overlap_max_abs": float(residual.max()),
        "tile_seam_128_max_abs": float(seam.max()),
        "finite_full_output": bool(np.isfinite(full.irradiance).all()),
        "qualification": "single host timing and numerical consistency only; no calibrated screen/camera and no full-resolution natural-image accuracy claim",
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
