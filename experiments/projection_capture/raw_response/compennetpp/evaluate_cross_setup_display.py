"""Does the source-image display transform transfer to a second real setup?

The first setup's frozen 256-to-800x600 placement is used unchanged. Local
four-color references are calibrated in the second setup. A separately fitted
placement using only its train images 1-2 is a diagnostic, not a test fit.
"""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT

from palimpsest.paths import WORK_DIR

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import differential_evolution

from experiments.projection_capture.structured_light.compennetpp.protocol import (
    decode_axis,
    gray,
    load_at,
)
from experiments.projection_capture.raw_response.compennetpp.evaluate_raw_forward_pilot import (
    NUMBERS,
    PROBE as FIRST_PROBE,
    mae,
    primary_curve_prediction,
    remap_source,
)


SETUP = "light3/pos1/cloud_np"
FOLDER = DATA_ROOT / "derived/compennetpp_sl_light3_pos1_cloud"
PROBE = DATA_ROOT / "derived/compennetpp_raw_ref_probe_light3_pos1_cloud_np"
PROBE_MANIFEST = WORK_DIR / "compennetpp_light3_initial_probe.json"
PRIMARY_MANIFEST = WORK_DIR / "compennetpp_light3_primary_curve_probe.json"
TEST_MANIFEST = WORK_DIR / "compennetpp_light3_test20_extension_probe.json"
SL_MANIFEST = WORK_DIR / "compennetpp_sl_light3_pos1_cloud_manifest.json"
FIRST = WORK_DIR / "compennetpp_raw_forward_pilot.json"
OUT = WORK_DIR / "compennetpp_cross_setup_display_probe.json"


def local_image(role: str, category: str, number: int) -> np.ndarray:
    from PIL import Image

    name = (
        f"{category}__img_{number:04d}.png"
        if role == "source"
        else f"light3__pos1__cloud_np__cam__raw__{category}__img_{number:04d}.png"
    )
    with Image.open(PROBE / name) as im:
        return np.asarray(im.convert("RGB"), np.float32) / 255


def verify_manifests() -> dict:
    rows = json.loads(PROBE_MANIFEST.read_text(encoding="utf-8"))
    primary = json.loads(PRIMARY_MANIFEST.read_text(encoding="utf-8"))
    extension = json.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    sl = json.loads(SL_MANIFEST.read_text(encoding="utf-8"))
    if rows["setup"] != SETUP or not rows["sample_member_crc_verified"]:
        raise RuntimeError("second setup source/camera members lack CRC audit")
    if sl["setup"] != SETUP or not sl["pairs_complete"] or len(sl["rows"]) != 84:
        raise RuntimeError("second setup structured-light pairs lack CRC audit")
    if len(rows["sample_pairs"]) != 20:
        raise RuntimeError("second setup pilot is not 20 members")
    if (
        primary["setup"] != SETUP
        or not primary["sample_member_crc_verified"]
        or len(primary["sample_pairs"]) != 18
    ):
        raise RuntimeError("second setup primary curve refs lack CRC audit")
    if (
        extension["setup"] != SETUP
        or not extension["sample_member_crc_verified"]
        or len(extension["sample_pairs"]) != 34
    ):
        raise RuntimeError("second setup 17 extra test pairs lack CRC audit")
    all_rows = (
        rows["sample_pairs"] + primary["sample_pairs"] + extension["sample_pairs"]
    )
    source_matches = 0
    for row in all_rows:
        if hashlib.sha256(Path(row["local"]).read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError(
                f"local sample differs from CRC-audited ZIP member: {row['name']}"
            )
        if row["role"] == "source":
            other = FIRST_PROBE / row["name"].replace("/", "__")
            if hashlib.sha256(other.read_bytes()).hexdigest() != row["sha256"]:
                raise RuntimeError(
                    f"source input differs between setups: {row['name']}"
                )
            source_matches += 1
    return {
        "source_camera_member_count": len(all_rows),
        "source_inputs_matching_first_setup": source_matches,
        "structured_light_member_count": len(sl["rows"]),
    }


def main() -> None:
    audits = verify_manifests()
    first = json.loads(FIRST.read_text(encoding="utf-8"))
    prior_transform = np.asarray(first["display_transform_fitted"], np.float32)
    loader = lambda kind, index: load_at(FOLDER, kind, index)
    sh, sw = loader("source", 1).shape[:2]
    qx, cx, _ = decode_axis("x", 3, 10, (sh, sw), axis=1, image_loader=loader)
    qy, cy, _ = decode_axis("y", 23, 10, (sh, sw), axis=0, image_loader=loader)
    valid = (
        ((gray(loader("capture", 1)) - gray(loader("capture", 2))) > 0.12)
        & (qx >= 0)
        & (qy >= 0)
        & (cx > 0.04)
        & (cy > 0.04)
    )
    black = local_image("camera", "ref", 1)
    basis = np.stack(
        [local_image("camera", "ref", n) - black for n in (5, 21, 101)], axis=-2
    )
    primary_groups = ((1, 2, 3, 4, 5), (1, 6, 11, 16, 21), (1, 26, 51, 76, 101))
    primary_stacks = [
        np.stack([local_image("camera", "ref", n) - black for n in group], axis=2)
        for group in primary_groups
    ]

    ys, xs = np.nonzero(valid)
    rng = np.random.default_rng(1026)
    take = rng.choice(len(xs), size=min(5000, len(xs)), replace=False)
    yy, xx = ys[take], xs[take]
    qx_fit, qy_fit = (
        qx[yy, xx].astype(np.float32).reshape(-1, 1),
        qy[yy, xx].astype(np.float32).reshape(-1, 1),
    )
    black_fit, basis_fit = black[yy, xx], basis[yy, xx]
    train = [
        (local_image("source", "train", n), local_image("camera", "train", n)[yy, xx])
        for n in (1, 2)
    ]

    def objective(params: np.ndarray) -> float:
        errors = []
        for source, actual in train:
            input_rgb = remap_source(source, qx_fit, qy_fit, params)[:, 0]
            prediction = np.clip(
                black_fit + np.einsum("nkc,nk->nc", basis_fit, input_rgb), 0, 1
            )
            errors.append(float(np.abs(prediction - actual).mean()))
        return float(np.mean(errors))

    optimized = differential_evolution(
        objective,
        bounds=[(0.2, 0.6), (0.2, 0.6), (-100, 40), (-100, 40)],
        seed=2409,
        maxiter=40,
        popsize=8,
        polish=True,
        workers=1,
    )
    fitted = optimized.x.astype(np.float32)
    map_take = rng.choice(len(xs), size=min(10000, len(xs)), replace=False)
    camera_points = np.stack((xs[map_take], ys[map_take]), axis=-1).astype(np.float32)
    projector_points = np.stack(
        (qx[ys[map_take], xs[map_take]], qy[ys[map_take], xs[map_take]]), axis=-1
    ).astype(np.float32)
    homography, inliers = cv2.findHomography(
        camera_points, projector_points, cv2.RANSAC, 3.0
    )
    if homography is None:
        raise RuntimeError("second setup homography fit failed")
    grid_y, grid_x = np.indices(qx.shape, dtype=np.float32)
    camera_grid = np.stack((grid_x, grid_y), axis=-1)
    hmap = cv2.perspectiveTransform(camera_grid.reshape(-1, 1, 2), homography).reshape(
        *qx.shape, 2
    )

    def content_mask(params: np.ndarray) -> np.ndarray:
        sx, sy, tx, ty = params
        return (
            valid
            & (qx * sx + tx >= 0)
            & (qx * sx + tx <= 255)
            & (qy * sy + ty >= 0)
            & (qy * sy + ty <= 255)
        )

    common_content = content_mask(prior_transform) & content_mask(fitted)
    rows = []
    for number in NUMBERS:
        source, actual = (
            local_image("source", "test", number),
            local_image("camera", "test", number),
        )
        methods = {}
        for name, transform in (
            ("first_setup_frozen_display", prior_transform),
            ("second_setup_train_fitted_display", fitted),
        ):
            source_warp = remap_source(
                source, qx.astype(np.float32), qy.astype(np.float32), transform
            )
            four = black + np.einsum("hwkc,hwk->hwc", basis, source_warp)
            thirteen = primary_curve_prediction(source_warp, black, primary_stacks)
            methods[name] = {
                "four_color_common_content_mae": mae(four, actual, common_content)[
                    "mae"
                ],
                "thirteen_color_common_content_mae": mae(
                    thirteen, actual, common_content
                )["mae"],
                "four_color_sl_valid_mae": mae(four, actual, valid)["mae"],
                "thirteen_color_sl_valid_mae": mae(thirteen, actual, valid)["mae"],
            }
        homography_source = remap_source(
            source, hmap[..., 0], hmap[..., 1], prior_transform
        )
        homography_prediction = primary_curve_prediction(
            homography_source, black, primary_stacks
        )
        methods["first_setup_frozen_display"][
            "thirteen_color_homography_common_content_mae"
        ] = mae(homography_prediction, actual, common_content)["mae"]
        rows.append({"number": number, "methods": methods})
    output = {
        "setup": SETUP,
        "audits": audits,
        "first_setup_display_transform": prior_transform.tolist(),
        "second_setup_display_transform_train_fitted": fitted.tolist(),
        "display_transform_difference": (fitted - prior_transform).tolist(),
        "train_1_2_objective_frozen": objective(prior_transform),
        "train_1_2_objective_fitted": objective(fitted),
        "fit_success": bool(optimized.success),
        "sl_valid_count": int(valid.sum()),
        "common_content_count": int(common_content.sum()),
        "homography_inlier_fraction_at_3px": float(inliers.mean()),
        "test_numbers": list(NUMBERS),
        "test_pairs": rows,
        "mean_common_content_mae": {
            f"{placement}.{color}": float(
                np.mean(
                    [
                        row["methods"][placement][f"{color}_common_content_mae"]
                        for row in rows
                    ]
                )
            )
            for placement in (
                "first_setup_frozen_display",
                "second_setup_train_fitted_display",
            )
            for color in ("four_color", "thirteen_color")
        },
        "frozen_thirteen_better_than_four_count": sum(
            row["methods"]["first_setup_frozen_display"][
                "thirteen_color_common_content_mae"
            ]
            < row["methods"]["first_setup_frozen_display"][
                "four_color_common_content_mae"
            ]
            for row in rows
        ),
        "frozen_display_better_than_second_refit_with_thirteen_count": sum(
            row["methods"]["first_setup_frozen_display"][
                "thirteen_color_common_content_mae"
            ]
            < row["methods"]["second_setup_train_fitted_display"][
                "thirteen_color_common_content_mae"
            ]
            for row in rows
        ),
        "frozen_local_geometry_better_than_homography_count": sum(
            row["methods"]["first_setup_frozen_display"][
                "thirteen_color_common_content_mae"
            ]
            < row["methods"]["first_setup_frozen_display"][
                "thirteen_color_homography_common_content_mae"
            ]
            for row in rows
        ),
        "mean_frozen_thirteen_homography_mae": float(
            np.mean(
                [
                    row["methods"]["first_setup_frozen_display"][
                        "thirteen_color_homography_common_content_mae"
                    ]
                    for row in rows
                ]
            )
        ),
        "note": "Second setup changes lighting/geometry jointly; only model form and first display placement are being tested.",
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: output[key]
                for key in (
                    "setup",
                    "audits",
                    "sl_valid_count",
                    "common_content_count",
                    "mean_common_content_mae",
                    "frozen_thirteen_better_than_four_count",
                    "frozen_display_better_than_second_refit_with_thirteen_count",
                    "frozen_local_geometry_better_than_homography_count",
                    "mean_frozen_thirteen_homography_mae",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
