"""Diagnose where uniform-chart prediction fails on real projected textures.

Four named setups and test IDs 1..50 are fixed. Edge/saturation associations
are descriptive: they cannot identify optics versus publication registration.
"""

from __future__ import annotations

from palimpsest.paths import WORK_DIR

import json
import time
import zipfile

import numpy as np
from scipy.ndimage import sobel

from experiments.projector.probe_projector_chart_model import (
    ARCHIVE,
    AUDIT,
    load_rgb,
    read_chart,
)
from experiments.projector.probe_projector_full_lut import SETUPS, predict


OUT = WORK_DIR / "projector_residual_diagnosis.json"


def describe(source: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> dict:
    error = np.abs(prediction - target).mean(2)
    gray = (source * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(2)
    grad = np.hypot(
        sobel(gray, axis=0, mode="reflect"), sobel(gray, axis=1, mode="reflect")
    )
    interior = np.zeros(gray.shape, dtype=bool)
    interior[16:-16, 16:-16] = True
    border = ~interior
    lo, hi = np.quantile(grad[interior], [0.1, 0.9])
    unsaturated = np.max(target, axis=2) < 0.98
    return {
        "all_mae": float(error.mean()),
        "interior_mae": float(error[interior].mean()),
        "border_mae": float(error[border].mean()),
        "low_source_gradient_mae": float(error[interior & (grad <= lo)].mean()),
        "high_source_gradient_mae": float(error[interior & (grad >= hi)].mean()),
        "unsaturated_interior_mae": float(error[interior & unsaturated].mean())
        if (interior & unsaturated).any()
        else None,
        "saturated_fraction": float(np.mean(~unsaturated)),
    }


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    full = json.loads(
        (WORK_DIR / "projector_full_lut_probe.json").read_text(encoding="utf-8")
    )
    sigma_by_setup = {row["setup"]: row["chosen_sigma"] for row in full["results"]}
    if not audit["all_member_crc_passed"]:
        raise ValueError("full archive audit not passed")
    rows = []
    started = time.perf_counter()
    with zipfile.ZipFile(ARCHIVE) as archive:
        for setup in SETUPS:
            _, chart = read_chart(archive, setup)
            sigma = sigma_by_setup[setup]
            images = []
            for index in range(1, 51):
                x = load_rgb(archive, f"test/img_{index:04d}.png")
                y = load_rgb(archive, f"{setup}/cam/warp/test/img_{index:04d}.png")
                images.append(
                    {
                        "id": index,
                        "with_train_chosen_blur": describe(
                            x, y, predict(chart, x, sigma)
                        ),
                        "no_blur": describe(x, y, predict(chart, x, 0)),
                    }
                )
            summary = {
                condition: {
                    key: float(
                        np.mean(
                            [
                                image[condition][key]
                                for image in images
                                if image[condition][key] is not None
                            ]
                        )
                    )
                    for key in images[0][condition]
                }
                for condition in ("with_train_chosen_blur", "no_blur")
            }
            rows.append(
                {
                    "setup": setup,
                    "sigma_px": sigma,
                    "test_ids": "0001..0050",
                    "summary": summary,
                    "per_image": images,
                }
            )
            print(
                setup,
                {k: round(v, 4) for k, v in summary["with_train_chosen_blur"].items()},
                flush=True,
            )
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
