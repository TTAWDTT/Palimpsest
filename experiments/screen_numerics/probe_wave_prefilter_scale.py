"""Independent virtual display-scale and emitter-fill stress for wave prefilter."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from origin_simulation.screen_capture import render_screen_capture
from experiments.screen_numerics.probe_physical_focus_display_lattice import camera
from experiments.screen_numerics.probe_wave_display_prefilter import fast_irradiance


OUT = Path("work/wave_prefilter_scale_stress.json")


def main() -> None:
    frame = np.random.default_rng(89).uniform(0.1, 0.9, (56, 56, 3)).astype(np.float32)
    rows = []
    for scale, fill in ((0.7, 0.85), (0.93, 0.7), (0.93, 0.95), (1.1, 0.85)):
        for focus_m in (0.75, 1.0):
            params = replace(
                camera(focus_m, "wave"),
                emitter_layout="vertical_rgb",
                fill_fraction=fill,
            )
            H = params.sensor_to_display.copy()
            H[0, 0] = H[1, 1] = scale
            params = replace(params, sensor_to_display=H)
            reference = render_screen_capture(frame, (20, 20), params).irradiance
            for display_samples, sensor_samples in ((8, 4), (12, 6)):
                fast = fast_irradiance(
                    frame, params, (20, 20), display_samples, sensor_samples
                )
                residual = np.abs(reference[2:-2, 2:-2] - fast[2:-2, 2:-2])
                rows.append(
                    {
                        "display_scale": scale,
                        "fill_fraction": fill,
                        "focus_m": focus_m,
                        "display_samples": display_samples,
                        "sensor_samples": sensor_samples,
                        "mae": float(residual.mean()),
                        "p99": float(np.quantile(residual, 0.99)),
                    }
                )
    report = {
        "scope": "virtual 56x56 display, 20x20 sensor, vertical RGB, S8/Q4 and S12/Q6",
        "rows": rows,
        "qualification": "changing scale corresponds to changing hypothetical display pitch; not camera calibration",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
