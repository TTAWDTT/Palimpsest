"""Measure first/repeated forward-render cost for the two optical branches."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from origin_simulation.optical_psf import circular_pupil_defocus_psf
from origin_simulation.screen_capture import render_screen_capture
from work.probe_physical_focus_display_lattice import camera


OUT = Path("work/wave_defocus_render_benchmark.json")


def timed(frame: np.ndarray, model: str, spatial_method: str = "fine") -> float:
    begin = perf_counter()
    render_screen_capture(frame, (56, 56), camera(1.0, model),
                          spatial_method=spatial_method)
    return perf_counter() - begin


def main() -> None:
    frame = np.random.default_rng(20260925).uniform(.1, .9, size=(96, 96, 3)).astype(np.float32)
    circular_pupil_defocus_psf.cache_clear()
    wave_cold = timed(frame, "wave")
    wave_warm = [timed(frame, "wave") for _ in range(2)]
    wave_fast = [timed(frame, "wave", "wave_prefilter") for _ in range(2)]
    geometric = [timed(frame, "geometric_airy") for _ in range(2)]
    report = {
        "scope": "56x56 sensor crop, 96x96 random virtual display, CPU single-image fine versus wave_prefilter; includes RAW/ISP, no decode or publication",
        "wave_cold_s": wave_cold,
        "wave_warm_s": wave_warm,
        "wave_prefilter_s": wave_fast,
        "geometric_airy_s": geometric,
        "qualification": "one machine, few repeats; NOT a stable end-to-end throughput estimate",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
