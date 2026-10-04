"""Predefined gray-ramp tone hypotheses on frozen CompenNet++ test pairs.

Uses the display placement fixed by prior train images 1-2. Adds only two
gray-reference photos (IDs 32, 94); mixed reference IDs 31 and 95 and all
20 test textures remain held out. These are effective RGB response tests,
not an identification of the camera or projector transfer functions.
"""

from __future__ import annotations

from palimpsest.paths import WORK_DIR

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.projector.decode_compennetpp_sl_one_setup import (
    FOLDER,
    decode_axis,
    gray,
    load_at,
)
from experiments.projector.evaluate_compennetpp_raw_forward_pilot import (
    NUMBERS,
    PROBE,
    camera_image,
    camera_ref,
    image,
    mae,
    primary_curve_prediction,
    remap_source,
    source_image,
    verify_probe_manifests,
)


PRIOR = WORK_DIR / "compennetpp_raw_forward_pilot.json"
NEUTRAL_AUDIT = WORK_DIR / "compennetpp_raw_neutral_ramp_probe.json"
OUT = WORK_DIR / "compennetpp_neutral_tone_probe.json"
GRAY_IDS = (1, 32, 63, 94, 125)
CURVE_GROUPS = ((1, 2, 3, 4, 5), (1, 6, 11, 16, 21), (1, 26, 51, 76, 101))


def verify_neutral_probe() -> dict:
    rowset = json.loads(NEUTRAL_AUDIT.read_text(encoding="utf-8"))
    if not rowset["sample_member_crc_verified"] or rowset["full_zip_verified"]:
        raise RuntimeError("neutral reference archive audit status unexpected")
    names = set()
    for row in rowset["sample_pairs"]:
        path = Path(row["local"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError(f"neutral reference local SHA mismatch: {path}")
        names.add(row["name"])
    expected = {f"ref/img_{n:04d}.png" for n in (32, 94)} | {
        f"light1/pos1/cloud_np/cam/raw/ref/img_{n:04d}.png" for n in (32, 94)
    }
    if names != expected:
        raise RuntimeError(
            f"unexpected neutral reference ZIP members: {names ^ expected}"
        )
    return {"extra_crc_sha_verified_members": len(names)}


def global_tone_knots(
    predicted_grays: np.ndarray, observed_grays: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = np.median(predicted_grays[:, mask, :], axis=1)
    y = np.median(observed_grays[:, mask, :], axis=1)
    x = np.maximum.accumulate(x, axis=0)
    y = np.maximum.accumulate(y, axis=0)
    if np.any(np.diff(x, axis=0) < 1e-4):
        raise RuntimeError("global predicted gray response not strictly increasing")
    return x.astype(np.float32), y.astype(np.float32)


def apply_global(predicted: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.interp(predicted[..., c], x[:, c], y[:, c]) for c in range(3)], axis=-1
    ).astype(np.float32)


def apply_local(predicted: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # x,y: camera H,W,gray-level,RGB. Monotone envelopes avoid reversed bins.
    x = np.maximum.accumulate(x, axis=2)
    y = np.maximum.accumulate(y, axis=2)
    index = np.sum(predicted[..., None, :] >= x[..., 1:, :], axis=2)
    index = np.clip(index, 0, 3)
    a = np.take_along_axis(x, index[..., None, :], axis=2)[:, :, 0]
    b = np.take_along_axis(x, (index + 1)[..., None, :], axis=2)[:, :, 0]
    c = np.take_along_axis(y, index[..., None, :], axis=2)[:, :, 0]
    d = np.take_along_axis(y, (index + 1)[..., None, :], axis=2)[:, :, 0]
    weight = np.clip((predicted - a) / np.maximum(b - a, 1e-6), 0, 1)
    return c + (d - c) * weight


def main() -> None:
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    if prior["test_numbers"] != list(NUMBERS) or prior["content_count"] != 37839:
        raise RuntimeError("prior frozen 20-image protocol has changed")
    verify_probe_manifests()
    extra_audit = verify_neutral_probe()
    loader = lambda kind, index: load_at(FOLDER, kind, index)
    source_h, source_w = loader("source", 1).shape[:2]
    qx, cx, _ = decode_axis(
        "x", 3, 10, (source_h, source_w), axis=1, image_loader=loader
    )
    qy, cy, _ = decode_axis(
        "y", 23, 10, (source_h, source_w), axis=0, image_loader=loader
    )
    valid = (
        ((gray(loader("capture", 1)) - gray(loader("capture", 2))) > 0.12)
        & (qx >= 0)
        & (qy >= 0)
        & (cx > 0.04)
        & (cy > 0.04)
    )
    sx, sy, tx, ty = prior["display_transform_fitted"]
    content = (
        valid
        & (qx * sx + tx >= 0)
        & (qx * sx + tx <= 255)
        & (qy * sy + ty >= 0)
        & (qy * sy + ty <= 255)
    )
    if int(content.sum()) != prior["content_count"]:
        raise RuntimeError("content mask differs from frozen forward experiment")

    black = camera_ref(1)
    stacks = [
        np.stack([camera_ref(n) - black for n in group], axis=2)
        for group in CURVE_GROUPS
    ]
    gray_colors = [
        image(PROBE / f"ref__img_{number:04d}.png")[0, 0] for number in GRAY_IDS
    ]
    predicted_grays = np.stack(
        [
            primary_curve_prediction(np.broadcast_to(color, black.shape), black, stacks)
            for color in gray_colors
        ],
        axis=0,
    )
    observed_grays = np.stack([camera_ref(n) for n in GRAY_IDS], axis=0)
    global_x, global_y = global_tone_knots(predicted_grays, observed_grays, content)
    local_x = np.moveaxis(predicted_grays, 0, 2)
    local_y = np.moveaxis(observed_grays, 0, 2)

    rows = []
    for category, numbers in (("train", (3,)), ("test", NUMBERS)):
        for number in numbers:
            source = source_image(category, number)
            actual = camera_image(category, number)
            warped = remap_source(
                source,
                qx.astype(np.float32),
                qy.astype(np.float32),
                np.asarray((sx, sy, tx, ty), np.float32),
            )
            baseline = primary_curve_prediction(warped, black, stacks)
            pred = {
                "thirteen_primary_code_additive": baseline,
                "global_neutral_five_level_tone": apply_global(
                    baseline, global_x, global_y
                ),
                "pixelwise_neutral_five_level_tone": apply_local(
                    baseline, local_x, local_y
                ),
            }
            rows.append(
                {
                    "category": category,
                    "number": number,
                    "metrics": {
                        name: mae(value, actual, content)
                        for name, value in pred.items()
                    },
                }
            )
    mixed = []
    for number in (31, 95):
        source_color = image(PROBE / f"ref__img_{number:04d}.png")[0, 0]
        actual = camera_ref(number)
        baseline = primary_curve_prediction(
            np.broadcast_to(source_color, black.shape), black, stacks
        )
        pred = {
            "thirteen_primary_code_additive": baseline,
            "global_neutral_five_level_tone": apply_global(
                baseline, global_x, global_y
            ),
            "pixelwise_neutral_five_level_tone": apply_local(
                baseline, local_x, local_y
            ),
        }
        mixed.append(
            {
                "number": number,
                "source_color": source_color.tolist(),
                "metrics": {
                    name: mae(value, actual, content) for name, value in pred.items()
                },
            }
        )
    test_rows = [row for row in rows if row["category"] == "test"]
    methods = list(test_rows[0]["metrics"])
    output = {
        "additional_archive_audit": extra_audit,
        "gray_reference_numbers": list(GRAY_IDS),
        "mixed_reference_holdout_numbers": [31, 95],
        "display_transform_reused": [sx, sy, tx, ty],
        "content_count": int(content.sum()),
        "global_tone_knot_medians_predicted": global_x.tolist(),
        "global_tone_knot_medians_observed": global_y.tolist(),
        "mixed_reference_holdout": mixed,
        "heldout_texture_pairs": rows,
        "mean_test_mae": {
            name: float(np.mean([r["metrics"][name]["mae"] for r in test_rows]))
            for name in methods
        },
        "better_than_primary_count": {
            name: sum(
                r["metrics"][name]["mae"]
                < r["metrics"]["thirteen_primary_code_additive"]["mae"]
                for r in test_rows
            )
            for name in methods
            if name != "thirteen_primary_code_additive"
        },
        "note": "Global median gray tone and pixelwise gray tone are compound effective response hypotheses, not physically identified camera functions.",
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "mean_test_mae": output["mean_test_mae"],
                "better_than_primary_count": output["better_than_primary_count"],
                "mixed_reference_holdout": mixed,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
