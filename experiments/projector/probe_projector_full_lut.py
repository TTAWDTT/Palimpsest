"""High-capacity pixelwise 3D-LUT control for the chart-only forward probe.

All 125 uniform chart captures are retained at every pixel. This control is
not an optical/surface mechanism and is much larger than the factorized model.
"""

from __future__ import annotations

from experiments.paths import WORK_DIR

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from experiments.projector.probe_projector_chart_model import (
    ARCHIVE,
    AUDIT,
    LEVELS,
    load_rgb,
    read_chart,
    score,
)


OUT = WORK_DIR / "projector_full_lut_probe.json"
SETUPS = [
    "light1/pos1/stripes",
    "light2/pos1/curves",
    "light3/pos3/squares",
    "light4/pos2/curves",
]
SIGMAS = (0.0, 0.45, 0.9, 1.35)


def predict(chart: np.ndarray, source: np.ndarray, sigma: float) -> np.ndarray:
    if sigma:
        source = gaussian_filter(source, (sigma, sigma, 0), mode="reflect")
    cube = chart.reshape(5, 5, 5, 256, 256, 3).transpose(2, 1, 0, 3, 4, 5)
    low = np.clip(np.searchsorted(LEVELS, source, side="right") - 1, 0, 3)
    high = low + 1
    t = (source - LEVELS[low]) / (LEVELS[high] - LEVELS[low])
    yy, xx = np.indices(source.shape[:2])
    result = np.zeros_like(source)
    for ri in (0, 1):
        wr = t[..., 0] if ri else 1 - t[..., 0]
        ii = high[..., 0] if ri else low[..., 0]
        for gi in (0, 1):
            wg = t[..., 1] if gi else 1 - t[..., 1]
            jj = high[..., 1] if gi else low[..., 1]
            for bi in (0, 1):
                wb = t[..., 2] if bi else 1 - t[..., 2]
                kk = high[..., 2] if bi else low[..., 2]
                result += cube[ii, jj, kk, yy, xx] * (wr * wg * wb)[..., None]
    return np.clip(result, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all", action="store_true", help="evaluate all CRC-audited setups"
    )
    args = parser.parse_args()
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not audit["all_member_crc_passed"]:
        raise ValueError("full archive audit not passed")
    rows = []
    setups = sorted(audit["setups"]) if args.all else SETUPS
    started = time.perf_counter()
    with zipfile.ZipFile(ARCHIVE) as archive:
        for setup in setups:
            _, chart = read_chart(archive, setup)
            train = {str(s): [] for s in SIGMAS}
            for index in range(1, 17):
                x = load_rgb(archive, f"train/img_{index:04d}.png")
                y = load_rgb(archive, f"{setup}/cam/warp/train/img_{index:04d}.png")
                for sigma in SIGMAS:
                    train[str(sigma)].append(score(predict(chart, x, sigma), y)["mae"])
            chosen = min(SIGMAS, key=lambda s: np.mean(train[str(s)]))
            per_image = []
            for index in range(1, 201):
                x = load_rgb(archive, f"test/img_{index:04d}.png")
                y = load_rgb(archive, f"{setup}/cam/warp/test/img_{index:04d}.png")
                per_image.append(
                    {"id": index, "mae": score(predict(chart, x, chosen), y)["mae"]}
                )
            row = {
                "setup": setup,
                "train_count": 16,
                "test_count": 200,
                "chosen_sigma": chosen,
                "chart_parameter_bytes_float32": int(chart.nbytes),
                "train_mae_by_sigma": {s: float(np.mean(v)) for s, v in train.items()},
                "test_mae": float(np.mean([v["mae"] for v in per_image])),
                "test_by_image": per_image,
            }
            rows.append(row)
            print(setup, chosen, row["test_mae"], flush=True)
            OUT.write_text(
                json.dumps(
                    {
                        "archive_sha256": audit["archive_sha256"],
                        "elapsed_sec": time.perf_counter() - started,
                        "results": rows,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
