"""Transfer pure-channel tone curves while recalibrating four target charts.

Donor contributes 13-chart channel tone response. Target contributes black and
full red/green/blue captures, which determine a spatial 3x3 mixing field.
The same predeclared four donor/target pairs as the LUT transfer probe are used.
"""

from __future__ import annotations

from palimpsest.paths import WORK_DIR

import json
import time
import zipfile

import numpy as np

from experiments.projection_capture.chart_response.compennet.protocol import (
    ARCHIVE,
    AUDIT,
    load_rgb,
    score,
)
from experiments.projection_capture.chart_response.compennet.run_channel_basis import (
    fit,
    predict,
    ref_index,
    to_linear,
)
from experiments.projection_capture.chart_response.compennet.run_transfer import PAIRS


OUT = WORK_DIR / "projector_channel_transfer_probe.json"
SIGMA = 0.9


def target_endpoints(archive: zipfile.ZipFile, setup: str, linear: bool) -> tuple:
    prefix = f"{setup}/cam/warp/ref/"
    convert = to_linear if linear else lambda image: image
    black = convert(load_rgb(archive, prefix + "img_0001.png"))
    matrix = np.empty((256, 256, 3, 3), np.float32)
    for channel in range(3):
        full = convert(
            load_rgb(archive, prefix + f"img_{ref_index(channel, 4):04d}.png")
        )
        matrix[..., channel] = full - black
    return black, matrix


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not audit["all_member_crc_passed"]:
        raise ValueError("full archive audit not passed")
    rows = []
    started = time.perf_counter()
    with zipfile.ZipFile(ARCHIVE) as archive:
        for donor, target in PAIRS:
            results = {}
            for mode, linear in (("gamma_encoded", False), ("srgb_linearized", True)):
                donor_tone = fit(archive, donor, linear)[2]
                black, matrix = target_endpoints(archive, target, linear)
                params = (black, matrix, donor_tone, linear)
                per_image = []
                for index in range(1, 201):
                    x = load_rgb(archive, f"test/img_{index:04d}.png")
                    y = load_rgb(archive, f"{target}/cam/warp/test/img_{index:04d}.png")
                    per_image.append(score(predict(x, params, SIGMA), y)["mae"])
                results[mode] = {
                    "test_mae": float(np.mean(per_image)),
                    "test_mae_by_image": per_image,
                }
            rows.append(
                {
                    "donor": donor,
                    "target": target,
                    "donor_calibration": "13 pure-color charts",
                    "target_calibration": "black and full RGB primaries (4 charts)",
                    "results": results,
                }
            )
            print(
                donor,
                "->",
                target,
                {m: round(r["test_mae"], 4) for m, r in results.items()},
                flush=True,
            )
            OUT.write_text(
                json.dumps(
                    {
                        "archive_sha256": audit["archive_sha256"],
                        "sigma_px": SIGMA,
                        "elapsed_sec": time.perf_counter() - started,
                        "pairs": rows,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
