"""Can a chart-calibrated effective projector response transfer to another setup?

Target setup contributes only black/white captures for its spatial field;
the RGB response LUT comes exclusively from a donor setup. The four pairs are
predeclared before running this probe. This remains an effective RGB model.
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
    read_chart,
    fit_factorized,
    score,
    simulate,
)


OUT = WORK_DIR / "projector_transfer_probe.json"
PAIRS = [
    ("light1/pos1/stripes", "light1/pos2/stripes"),
    ("light1/pos1/stripes", "light4/pos1/stripes"),
    ("light2/pos1/squares", "light2/pos2/squares"),
    ("light2/pos2/lavender", "light3/pos3/lavender"),
]
SIGMA = 0.9  # fixed from the chart-only probe's training images, not target tests


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not audit["all_member_crc_passed"]:
        raise ValueError("full archive CRC audit not passed")
    started = time.perf_counter()
    rows = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for donor, target in PAIRS:
            _, donor_chart = read_chart(archive, donor)
            _, _, donor_lut = fit_factorized(donor_chart)
            black = load_rgb(archive, f"{target}/cam/warp/ref/img_0001.png")
            white = load_rgb(archive, f"{target}/cam/warp/ref/img_0125.png")
            params = (black, white - black, donor_lut)
            per_image = []
            for index in range(1, 201):
                x = load_rgb(archive, f"test/img_{index:04d}.png")
                y = load_rgb(archive, f"{target}/cam/warp/test/img_{index:04d}.png")
                per_image.append(
                    {
                        "id": index,
                        "input_only": score(x, y),
                        "donor_response": score(
                            simulate("factorized", x, params, SIGMA), y
                        ),
                    }
                )
            aggregate = {
                name: {
                    metric: float(np.mean([row[name][metric] for row in per_image]))
                    for metric in ("mae", "rmse", "psnr")
                }
                for name in ("input_only", "donor_response")
            }
            rows.append(
                {
                    "donor": donor,
                    "target": target,
                    "target_calibration": "black/white chart captures only",
                    "test_count": 200,
                    "aggregate": aggregate,
                    "per_image": per_image,
                }
            )
            print(
                donor,
                "->",
                target,
                {k: round(v["mae"], 4) for k, v in aggregate.items()},
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
