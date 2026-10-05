"""Separate display-placement optimization from photometric-model error.

Fit source-to-projector placement on train textures 1-2 under the 13-color
response, with the identical objective/sampling across two real setups.
Compare to frozen placement from the first setup and to earlier four-color
fits on a common test-content mask. No test texture enters either fit.
"""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT

from palimpsest.paths import WORK_DIR

import json
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from experiments.projection_capture.structured_light.compennetpp.protocol import (
    decode_axis,
    gray,
    load_at,
)
from experiments.projection_capture.raw_response.compennetpp.evaluate_cross_setup_display import (
    PROBE as SECOND_PROBE,
    verify_manifests as verify_second,
)
from experiments.projection_capture.raw_response.compennetpp.evaluate_raw_forward_pilot import (
    NUMBERS,
    PROBE as FIRST_PROBE,
    mae,
    primary_curve_prediction,
    remap_source,
    verify_probe_manifests as verify_first,
)


FIRST_SL = DATA_ROOT / "derived/compennetpp_sl_one_setup"
SECOND_SL = DATA_ROOT / "derived/compennetpp_sl_light3_pos1_cloud"
FIRST_JSON = WORK_DIR / "compennetpp_raw_forward_pilot.json"
SECOND_JSON = WORK_DIR / "compennetpp_cross_setup_display_probe.json"
OUT = WORK_DIR / "compennetpp_display_fit_identifiability.json"
GROUPS = ((1, 2, 3, 4, 5), (1, 6, 11, 16, 21), (1, 26, 51, 76, 101))


def image(folder: Path, setup: str, role: str, category: str, index: int) -> np.ndarray:
    from PIL import Image

    prefix = "" if role == "source" else setup + "__cam__raw__"
    with Image.open(folder / f"{prefix}{category}__img_{index:04d}.png") as im:
        return np.asarray(im.convert("RGB"), np.float32) / 255


def evaluate_one(
    name: str,
    folder: Path,
    sl_folder: Path,
    setup_prefix: str,
    first_transform: np.ndarray,
    earlier_four_transform: np.ndarray,
) -> dict:
    load = lambda role, n: load_at(sl_folder, role, n)
    sh, sw = load("source", 1).shape[:2]
    qx, cx, _ = decode_axis("x", 3, 10, (sh, sw), axis=1, image_loader=load)
    qy, cy, _ = decode_axis("y", 23, 10, (sh, sw), axis=0, image_loader=load)
    valid = (
        ((gray(load("capture", 1)) - gray(load("capture", 2))) > 0.12)
        & (qx >= 0)
        & (qy >= 0)
        & (cx > 0.04)
        & (cy > 0.04)
    )
    black = image(folder, setup_prefix, "camera", "ref", 1)
    stack = [
        np.stack(
            [image(folder, setup_prefix, "camera", "ref", n) - black for n in group],
            axis=2,
        )
        for group in GROUPS
    ]
    yy, xx = np.nonzero(valid)
    rng = np.random.default_rng(1026)
    take = rng.choice(len(xx), size=min(5000, len(xx)), replace=False)
    ytake, xtake = yy[take], xx[take]
    sx_fit = qx[ytake, xtake].astype(np.float32).reshape(-1, 1)
    sy_fit = qy[ytake, xtake].astype(np.float32).reshape(-1, 1)
    black_fit = black[ytake, xtake].reshape(-1, 1, 3)
    stack_fit = [s[ytake, xtake].reshape(-1, 1, 5, 3) for s in stack]
    train = [
        (
            image(folder, setup_prefix, "source", "train", n),
            image(folder, setup_prefix, "camera", "train", n)[ytake, xtake].reshape(
                -1, 1, 3
            ),
        )
        for n in (1, 2)
    ]

    def objective(params: np.ndarray) -> float:
        losses = []
        for source, actual in train:
            warped = remap_source(source, sx_fit, sy_fit, params)
            predicted = np.clip(
                primary_curve_prediction(warped, black_fit, stack_fit), 0, 1
            )
            losses.append(float(np.abs(predicted - actual).mean()))
        return float(np.mean(losses))

    result = differential_evolution(
        objective,
        bounds=[(0.2, 0.6), (0.2, 0.6), (-100, 40), (-100, 40)],
        seed=2409,
        maxiter=40,
        popsize=8,
        polish=True,
        workers=1,
    )
    thirteen_fit = result.x.astype(np.float32)
    candidates = {
        "center_square": np.array(
            [256 / 600, 256 / 600, -100 * 256 / 600, 0], np.float32
        ),
        "first_setup_frozen": first_transform,
        "earlier_four_color_fit": earlier_four_transform,
        "thirteen_color_fit": thirteen_fit,
    }
    masks = {}
    for label, (scale_x, scale_y, offset_x, offset_y) in candidates.items():
        masks[label] = (
            valid
            & (qx * scale_x + offset_x >= 0)
            & (qx * scale_x + offset_x <= 255)
            & (qy * scale_y + offset_y >= 0)
            & (qy * scale_y + offset_y <= 255)
        )
    common = np.logical_and.reduce(list(masks.values()))
    if common.sum() < 10000:
        raise RuntimeError("too few common content pixels for placement comparison")

    rows = []
    for n in NUMBERS:
        source = image(folder, setup_prefix, "source", "test", n)
        actual = image(folder, setup_prefix, "camera", "test", n)
        scores = {}
        for label, params in candidates.items():
            warped = remap_source(
                source, qx.astype(np.float32), qy.astype(np.float32), params
            )
            predicted = primary_curve_prediction(warped, black, stack)
            scores[label] = mae(predicted, actual, common)["mae"]
        rows.append({"number": n, "content_mae": scores})
    averages = {
        label: float(np.mean([r["content_mae"][label] for r in rows]))
        for label in candidates
    }
    return {
        "setup": name,
        "sl_valid_count": int(valid.sum()),
        "common_content_count": int(common.sum()),
        "transforms": {key: value.tolist() for key, value in candidates.items()},
        "train_1_2_objective": {
            key: objective(value) for key, value in candidates.items()
        },
        "thirteen_fit_success": bool(result.success),
        "thirteen_fit_evaluations": int(result.nfev),
        "test_mean_mae": averages,
        "thirteen_fit_better_than_four_fit_count": sum(
            row["content_mae"]["thirteen_color_fit"]
            < row["content_mae"]["earlier_four_color_fit"]
            for row in rows
        ),
        "thirteen_fit_better_than_first_frozen_count": sum(
            row["content_mae"]["thirteen_color_fit"]
            < row["content_mae"]["first_setup_frozen"]
            for row in rows
        ),
        "test_pairs": rows,
    }


def main() -> None:
    verify_first()
    verify_second()
    first = json.loads(FIRST_JSON.read_text(encoding="utf-8"))
    second = json.loads(SECOND_JSON.read_text(encoding="utf-8"))
    frozen = np.asarray(first["display_transform_fitted"], np.float32)
    output = {
        "first": evaluate_one(
            "light1/pos1/cloud_np",
            FIRST_PROBE,
            FIRST_SL,
            "light1__pos1__cloud_np",
            frozen,
            frozen,
        ),
        "second": evaluate_one(
            "light3/pos1/cloud_np",
            SECOND_PROBE,
            SECOND_SL,
            "light3__pos1__cloud_np",
            frozen,
            np.asarray(
                second["second_setup_display_transform_train_fitted"], np.float32
            ),
        ),
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: {
                    field: value[field]
                    for field in (
                        "common_content_count",
                        "transforms",
                        "train_1_2_objective",
                        "test_mean_mae",
                        "thirteen_fit_better_than_four_fit_count",
                        "thirteen_fit_better_than_first_frozen_count",
                    )
                }
                for key, value in output.items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
