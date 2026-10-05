"""Pilot forward projection test in native camera coordinates.

Geometry comes only from the 42 structured-light pairs. Photometry uses only
four uniform references: projector black and primary R/G/B endpoints. Three
texture pairs and intermediate colors are withheld from calibration. This is
an effective RGB response, not projector/surface/camera parameter separation.
"""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT

from palimpsest.paths import WORK_DIR

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import differential_evolution

from experiments.projection_capture.structured_light.compennetpp.protocol import (
    FOLDER,
    decode_axis,
    gray,
    load_at,
)


PROBE = DATA_ROOT / "derived/compennetpp_raw_ref_probe"
OUT = WORK_DIR / "compennetpp_raw_forward_pilot.json"
SETUP = "light1__pos1__cloud_np__cam__raw"
NUMBERS = (
    1,
    2,
    3,
    10,
    35,
    37,
    40,
    48,
    88,
    97,
    98,
    112,
    113,
    114,
    116,
    153,
    183,
    185,
    189,
    198,
)
TRAIN_NUMBERS = (1, 2, 3)
PROBE_MANIFESTS = (
    "compennetpp_raw_ref_probe.json",
    "compennetpp_raw_primary_response_probe.json",
    "compennetpp_raw_primary_curve_probe.json",
    "compennetpp_raw_train_geometry_probe.json",
    "compennetpp_raw_test20_extension_probe.json",
)


def image(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255


def camera_ref(index: int) -> np.ndarray:
    return image(PROBE / f"{SETUP}__ref__img_{index:04d}.png")


def source_image(category: str, index: int) -> np.ndarray:
    return image(PROBE / f"{category}__img_{index:04d}.png")


def camera_image(category: str, index: int) -> np.ndarray:
    return image(PROBE / f"{SETUP}__{category}__img_{index:04d}.png")


def remap_source(
    source: np.ndarray, qx: np.ndarray, qy: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    sx, sy, tx, ty = transform
    return cv2.remap(
        source,
        (qx * sx + tx).astype(np.float32),
        (qy * sy + ty).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def primary_curve_prediction(
    source_rgb: np.ndarray, black: np.ndarray, primary_stacks: list[np.ndarray]
) -> np.ndarray:
    """Add three independently measured camera-RGB primary responses.

    This is a falsifiable effective-code model; channel additivity and
    piecewise interpolation are assumptions, not identified physics.
    """
    levels = np.array([0, 64, 128, 191, 255], dtype=np.float32) / 255
    predicted = black.copy()
    for channel, stack in enumerate(primary_stacks):
        value = source_rgb[..., channel]
        low = np.clip(np.searchsorted(levels, value, side="right") - 1, 0, 3)
        weight = (value - levels[low]) / (levels[low + 1] - levels[low])
        v0 = np.take_along_axis(stack, low[..., None, None], axis=2)[:, :, 0]
        v1 = np.take_along_axis(stack, (low + 1)[..., None, None], axis=2)[:, :, 0]
        predicted += (1 - weight[..., None]) * v0 + weight[..., None] * v1
    return predicted


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0, 1)
    return np.where(
        rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1 / 2.4) - 0.055
    )


def mae(pred: np.ndarray, actual: np.ndarray, mask: np.ndarray) -> dict:
    delta = np.abs(np.clip(pred, 0, 1) - actual)[mask]
    return {
        "mae": float(delta.mean()),
        "p50_pixel_mae": float(np.median(delta.mean(1))),
        "p90_pixel_mae": float(np.quantile(delta.mean(1), 0.9)),
    }


def verify_probe_manifests() -> dict:
    rows = {}
    for filename in PROBE_MANIFESTS:
        document = json.loads((WORK_DIR / filename).read_text(encoding="utf-8"))
        if not document["sample_member_crc_verified"] or document["full_zip_verified"]:
            raise RuntimeError(f"unexpected audit status in {filename}")
        if document["setup"].replace("/", "__") != SETUP.replace("__cam__raw", ""):
            raise RuntimeError(f"setup mismatch in {filename}")
        for row in document["sample_pairs"]:
            path = Path(row["local"])
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
            ):
                raise RuntimeError(
                    f"local image differs from CRC-audited member: {path}"
                )
            if row["name"] in rows and row["sha256"] != rows[row["name"]]["sha256"]:
                raise RuntimeError(
                    f"conflicting ZIP member in probe manifests: {row['name']}"
                )
            rows[row["name"]] = row
    required = []
    for number in (1, 2, 3, 4, 5, 6, 11, 16, 21, 26, 31, 51, 63, 76, 95, 101, 125):
        required.extend(
            (
                f"ref/img_{number:04d}.png",
                f"light1/pos1/cloud_np/cam/raw/ref/img_{number:04d}.png",
            )
        )
    for category, numbers in (("train", TRAIN_NUMBERS), ("test", NUMBERS)):
        for number in numbers:
            required.extend(
                (
                    f"{category}/img_{number:04d}.png",
                    f"light1/pos1/cloud_np/cam/raw/{category}/img_{number:04d}.png",
                )
            )
    missing = sorted(set(required) - rows.keys())
    if missing:
        raise RuntimeError(f"CRC-audited member manifest missing: {missing}")
    return {
        "manifest_files": list(PROBE_MANIFESTS),
        "unique_verified_members": len(rows),
        "required_verified_members": len(set(required)),
    }


def main() -> None:
    audit = verify_probe_manifests()
    loader = lambda kind, index: load_at(FOLDER, kind, index)
    projector_h, projector_w = loader("source", 1).shape[:2]
    x, cx, _ = decode_axis(
        "x", 3, 10, (projector_h, projector_w), axis=1, image_loader=loader
    )
    y, cy, _ = decode_axis(
        "y", 23, 10, (projector_h, projector_w), axis=0, image_loader=loader
    )
    white_sl = gray(loader("capture", 1))
    black_sl = gray(loader("capture", 2))
    valid = (
        ((white_sl - black_sl) > 0.12) & (x >= 0) & (y >= 0) & (cx > 0.04) & (cy > 0.04)
    )
    if int(valid.sum()) < 10000:
        raise RuntimeError("insufficient decoded camera field")

    refs = {number: camera_ref(number) for number in (1, 5, 21, 101, 125)}
    if (
        len({arr.shape for arr in refs.values()}) != 1
        or next(iter(refs.values())).shape[:2] != x.shape
    ):
        raise RuntimeError("reference camera sizes do not match decoded geometry")
    black = refs[1]
    white = refs[125]
    basis = np.stack([refs[number] - black for number in (5, 21, 101)], axis=-2)
    curve_indices = ((1, 2, 3, 4, 5), (1, 6, 11, 16, 21), (1, 26, 51, 76, 101))
    primary_stacks = [
        np.stack([camera_ref(n) - black for n in group], axis=2)
        for group in curve_indices
    ]
    black_linear = srgb_to_linear(black)
    primary_stacks_linear = [
        np.stack([srgb_to_linear(camera_ref(n)) - black_linear for n in group], axis=2)
        for group in curve_indices
    ]

    def linearized_primary_prediction(source_rgb: np.ndarray) -> np.ndarray:
        return linear_to_srgb(
            primary_curve_prediction(source_rgb, black_linear, primary_stacks_linear)
        )

    # Original textures are 256^2 while structured-light input is 800x600.
    # Estimate the unknown display scaling/placement from train 1-2 only.
    source_h, source_w = source_image("train", 1).shape[:2]
    if (source_w, source_h) != (256, 256):
        raise RuntimeError("unexpected texture source size")
    yy, xx = np.nonzero(valid)
    rng = np.random.default_rng(1026)
    fit = rng.choice(len(xx), size=min(5000, len(xx)), replace=False)
    fit_y, fit_x = yy[fit], xx[fit]
    qx_fit = x[fit_y, fit_x].astype(np.float32).reshape(-1, 1)
    qy_fit = y[fit_y, fit_x].astype(np.float32).reshape(-1, 1)
    black_fit = black[fit_y, fit_x]
    basis_fit = basis[fit_y, fit_x]
    calibration = [
        (source_image("train", n), camera_image("train", n)[fit_y, fit_x])
        for n in (1, 2)
    ]

    def objective(params: np.ndarray) -> float:
        losses = []
        for source, actual in calibration:
            rgb = remap_source(source, qx_fit, qy_fit, params)[:, 0]
            pred = np.clip(black_fit + np.einsum("nkc,nk->nc", basis_fit, rgb), 0, 1)
            losses.append(float(np.mean(np.abs(pred - actual))))
        return float(np.mean(losses))

    fit_result = differential_evolution(
        objective,
        bounds=[(0.2, 0.6), (0.2, 0.6), (-100, 40), (-100, 40)],
        seed=2409,
        maxiter=40,
        popsize=8,
        polish=True,
        workers=1,
    )
    transform = fit_result.x.astype(np.float32)
    content_mask = (
        valid
        & (x * transform[0] + transform[2] >= 0)
        & (x * transform[0] + transform[2] <= source_w - 1)
        & (y * transform[1] + transform[3] >= 0)
        & (y * transform[1] + transform[3] <= source_h - 1)
    )
    candidate_transforms = {
        "centered_square_600px": [0.42666667, 0.42666667, -42.666667, 0.0],
        "stretch_800x600": [0.32, 0.42666667, 0.0, 0.0],
        "fill_800px_crop_height": [0.32, 0.32, 0.0, 32.0],
        "fitted_train_1_2": transform.tolist(),
    }

    sample = rng.choice(len(xx), size=min(10000, len(xx)), replace=False)
    camera_pts = np.stack((xx[sample], yy[sample]), axis=1).astype(np.float32)
    projector_pts = np.stack(
        (x[yy[sample], xx[sample]], y[yy[sample], xx[sample]]), axis=1
    ).astype(np.float32)
    homography, inliers = cv2.findHomography(camera_pts, projector_pts, cv2.RANSAC, 3.0)
    if homography is None:
        raise RuntimeError("homography fit failed")
    grid_y, grid_x = np.indices(x.shape, dtype=np.float32)
    grid = np.stack((grid_x, grid_y), axis=-1)
    hmap = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), homography).reshape(
        *x.shape, 2
    )

    base = {
        "calibration_ref_numbers_four_color": [1, 5, 21, 101],
        "image_audit": audit,
        "calibration_ref_numbers_primary_curves": sorted(
            {n for group in curve_indices for n in group}
        ),
        "withheld_ref_numbers_primary_curves": [31, 63, 95, 125],
        "train_numbers_for_display_fit": [1, 2],
        "withheld_train_numbers": [3],
        "test_numbers": list(NUMBERS),
        "test_selection": "first three inspected before the 13-color extension; remaining 17 preselected by NumPy default_rng(1026) without replacement from integers 4..200",
        "camera_size": [x.shape[1], x.shape[0]],
        "structured_light_source_size": [projector_w, projector_h],
        "texture_source_size": [source_w, source_h],
        "display_transform_candidate_development_losses": {
            name: objective(np.asarray(params))
            for name, params in candidate_transforms.items()
        },
        "display_transform_fitted": transform.tolist(),
        "display_fit_success": bool(fit_result.success),
        "display_fit_evaluations": int(fit_result.nfev),
        "valid_count": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "content_count": int(content_mask.sum()),
        "fitted_display_inside_fraction_of_valid": float(
            content_mask.sum() / valid.sum()
        ),
        "homography_inlier_fraction_at_3px": float(inliers.mean()),
        "measurement_note": "Primary MAE on high-confidence pixels within fitted 256-square content display; whole-SL-valid MAE also saved; RGB PNG is not sensor RAW",
    }

    ref_errors = []
    for number in (2, 3, 31, 63, 95):
        source = image(PROBE / f"ref__img_{number:04d}.png")
        actual = camera_ref(number)
        color = source[0, 0]
        predicted = black + np.einsum("hwkc,k->hwc", basis, color)
        ref_errors.append(
            {
                "number": number,
                "source_color": color.tolist(),
                "rgb_basis": mae(predicted, actual, content_mask),
                "primary_curves": mae(
                    primary_curve_prediction(
                        np.broadcast_to(color, actual.shape), black, primary_stacks
                    ),
                    actual,
                    content_mask,
                ),
                "primary_curves_srgb_linear_proxy": mae(
                    linearized_primary_prediction(np.broadcast_to(color, actual.shape)),
                    actual,
                    content_mask,
                ),
                "black_white_diagonal": mae(
                    black + (white - black) * color, actual, content_mask
                ),
            }
        )
    base["withheld_uniform_reference_errors"] = ref_errors

    rows = []
    for category, numbers in (("train", (3,)), ("test", NUMBERS)):
        for number in numbers:
            source = source_image(category, number)
            actual = camera_image(category, number)
            local = remap_source(
                source, x.astype(np.float32), y.astype(np.float32), transform
            )
            global_warp = remap_source(source, hmap[..., 0], hmap[..., 1], transform)
            shifted = remap_source(
                source, x.astype(np.float32) + 16, y.astype(np.float32), transform
            )
            predictions = {
                "black_only": black,
                "direct_projector_rgb_local_code": local,
                "black_white_diagonal_local_code": black + (white - black) * local,
                "four_color_rgb_basis_local_code": black
                + np.einsum("hwkc,hwk->hwc", basis, local),
                "thirteen_color_primary_curves_local_code": primary_curve_prediction(
                    local, black, primary_stacks
                ),
                "thirteen_color_primary_curves_srgb_linear_proxy": linearized_primary_prediction(
                    local
                ),
                "four_color_rgb_basis_homography": black
                + np.einsum("hwkc,hwk->hwc", basis, global_warp),
                "four_color_rgb_basis_shift_16px": black
                + np.einsum("hwkc,hwk->hwc", basis, shifted),
                "thirteen_color_primary_curves_homography": primary_curve_prediction(
                    global_warp, black, primary_stacks
                ),
                "thirteen_color_primary_curves_shift_16px": primary_curve_prediction(
                    shifted, black, primary_stacks
                ),
            }
            rows.append(
                {
                    "category": category,
                    "number": number,
                    "source_sha256": hashlib.sha256(
                        (PROBE / f"{category}__img_{number:04d}.png").read_bytes()
                    ).hexdigest(),
                    "camera_sha256": hashlib.sha256(
                        (
                            PROBE / f"{SETUP}__{category}__img_{number:04d}.png"
                        ).read_bytes()
                    ).hexdigest(),
                    "metrics": {
                        name: mae(pred, actual, content_mask)
                        for name, pred in predictions.items()
                    },
                    "whole_sl_valid_mae": {
                        name: mae(pred, actual, valid)["mae"]
                        for name, pred in predictions.items()
                    },
                }
            )
    base["heldout_pairs"] = rows
    base["mean_test_mae_by_method"] = {
        name: float(
            np.mean(
                [
                    row["metrics"][name]["mae"]
                    for row in rows
                    if row["category"] == "test"
                ]
            )
        )
        for name in rows[0]["metrics"]
    }
    base["median_test_mae_by_method"] = {
        name: float(
            np.median(
                [
                    row["metrics"][name]["mae"]
                    for row in rows
                    if row["category"] == "test"
                ]
            )
        )
        for name in rows[0]["metrics"]
    }
    test_rows = [row for row in rows if row["category"] == "test"]
    base["primary_curve_better_than_four_color_count"] = sum(
        row["metrics"]["thirteen_color_primary_curves_local_code"]["mae"]
        < row["metrics"]["four_color_rgb_basis_local_code"]["mae"]
        for row in test_rows
    )
    base["local_geometry_better_than_homography_with_primary_curves_count"] = sum(
        row["metrics"]["thirteen_color_primary_curves_local_code"]["mae"]
        < row["metrics"]["thirteen_color_primary_curves_homography"]["mae"]
        for row in test_rows
    )
    OUT.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "valid_count": base["valid_count"],
                "withheld_ref": ref_errors,
                "mean_test_mae": base["mean_test_mae_by_method"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
