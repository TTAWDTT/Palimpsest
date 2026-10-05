"""CPU simulation-kernel timing on one 256px CompenNet input.

Parameters are preloaded/fitted. Input PNG decoding and ZIP reading are timed
separately; this is not an end-to-end camera or origin-detector latency.
"""

from __future__ import annotations

from palimpsest.paths import WORK_DIR

import io
import json
import os
import platform
import time
import zipfile

import numpy as np
from PIL import Image

from experiments.projection_capture.chart_response.compennet.protocol import (
    ARCHIVE,
    AUDIT,
    fit_factorized,
    read_chart,
    simulate,
)
from experiments.projection_capture.chart_response.compennet.run_channel_basis import (
    fit,
    predict as basis_predict,
)
from experiments.projection_capture.chart_response.compennet.run_full_lut import (
    predict as lut_predict,
)


OUT = WORK_DIR / "projector_forward_speed.json"
SETUP = "light1/pos1/stripes"
REPEATS = 64
WARMUP = 8


def measure(function) -> dict:
    for _ in range(WARMUP):
        function()
    samples = []
    for _ in range(REPEATS):
        started = time.perf_counter_ns()
        result = function()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
        if result.shape != (256, 256, 3):
            raise ValueError("unexpected output shape")
    return {
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.quantile(samples, 0.95)),
        "min_ms": float(np.min(samples)),
        "repeats": REPEATS,
        "warmup": WARMUP,
    }


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not audit["all_member_crc_passed"]:
        raise ValueError("official archive CRC audit not passed")
    with zipfile.ZipFile(ARCHIVE) as archive:
        _, chart = read_chart(archive, SETUP)
        black, gain, lut = fit_factorized(chart)
        basis = fit(archive, SETUP, True)
        input_bytes = archive.read("test/img_0001.png")
    with Image.open(io.BytesIO(input_bytes)) as image:
        x = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    methods = {
        "factorized_125_chart": (
            lambda: simulate("factorized", x, (black, gain, lut), 0.9),
            black.nbytes + gain.nbytes + lut.nbytes,
        ),
        "channel_basis_13_chart": (
            lambda: basis_predict(x, basis, 0.9),
            basis[0].nbytes + basis[1].nbytes + basis[2].nbytes,
        ),
        "full_pixel_lut_125_chart": (lambda: lut_predict(chart, x, 0.9), chart.nbytes),
        "decode_input_png": (
            lambda: np.asarray(Image.open(io.BytesIO(input_bytes)).convert("RGB")),
            len(input_bytes),
        ),
    }
    results = {}
    for name, (function, memory) in methods.items():
        if name == "decode_input_png":
            # The kernel timing helper requires a 256x256x3 result, which the
            # decoded byte image also provides.
            pass
        result = measure(function)
        result["parameter_or_input_bytes"] = int(memory)
        results[name] = result
        print(name, round(result["median_ms"], 2), "ms", flush=True)
    OUT.write_text(
        json.dumps(
            {
                "archive_sha256": audit["archive_sha256"],
                "setup": SETUP,
                "input_id": "test/img_0001.png",
                "image_size": [256, 256],
                "sigma_px": 0.9,
                "processor": platform.processor(),
                "cpu_count": os.cpu_count(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "timing_scope": "one Python call per image, preloaded parameters/input; separate PNG decode",
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
