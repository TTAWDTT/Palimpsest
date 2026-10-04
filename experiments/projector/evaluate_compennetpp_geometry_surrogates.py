"""Hold out camera tiles when fitting effective projector-camera geometry.

This distinguishes a single planar homography from a low-degree smooth warp
on one decoded real setup. Neither model is a calibrated 3D surface renderer;
incorrect structured-light bits and foreground/background boundaries remain.
"""

from __future__ import annotations

from palimpsest.paths import WORK_DIR

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from experiments.projector.decode_compennetpp_sl_one_setup import (
    FOLDER,
    MANIFEST,
    decode_axis,
    gray,
    load_at,
)


OUT = WORK_DIR / "compennetpp_geometry_surrogate_holdout.json"
SEED = 9182
TILE_SIZE = 40
CONTRAST = 0.04


def design(coords: np.ndarray, degree: int) -> np.ndarray:
    u = (coords[:, 0] - 319.5) / 319.5
    v = (coords[:, 1] - 239.5) / 239.5
    terms = [
        u**p * v**q
        for total in range(degree + 1)
        for p in range(total + 1)
        for q in [total - p]
    ]
    return np.stack(terms, axis=1).astype(np.float64)


def robust_polynomial(
    train_camera: np.ndarray, train_projector: np.ndarray, degree: int
) -> np.ndarray:
    matrix = design(train_camera, degree)
    target = train_projector.astype(np.float64)
    weights = np.ones(len(matrix), np.float64)
    coefficients = None
    for _ in range(15):
        root = np.sqrt(weights)
        coefficients = np.linalg.lstsq(
            matrix * root[:, None], target * root[:, None], rcond=None
        )[0]
        residual = np.linalg.norm(matrix @ coefficients - target, axis=1)
        weights = np.minimum(1.0, 3.0 / np.maximum(residual, 1e-6))
    return coefficients


def summarize(prediction: np.ndarray, target: np.ndarray) -> dict:
    residual = np.linalg.norm(prediction - target, axis=1)
    return {
        "count": int(len(residual)),
        "median_projector_px": float(np.median(residual)),
        "p90_projector_px": float(np.quantile(residual, 0.9)),
        "below_3px_fraction": float(np.mean(residual < 3)),
        "below_10px_fraction": float(np.mean(residual < 10)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--folder", type=Path, default=FOLDER)
    parser.add_argument("--output-json", type=Path, default=OUT)
    args = parser.parse_args()
    image_loader = lambda kind, index: load_at(args.folder, kind, index)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest["pairs_complete"] or len(manifest["rows"]) != 84:
        raise ValueError(
            "structured-light selected members not completely CRC verified"
        )
    shape = image_loader("source", 1).shape[:2]
    x, cx, _ = decode_axis("x", 3, 10, shape, axis=1, image_loader=image_loader)
    y, cy, _ = decode_axis("y", 23, 10, shape, axis=0, image_loader=image_loader)
    contrast = gray(image_loader("capture", 1)) - gray(image_loader("capture", 2))
    yy, xx = np.nonzero(
        (contrast > 0.12) & (x >= 0) & (y >= 0) & (cx > CONTRAST) & (cy > CONTRAST)
    )
    camera = np.column_stack([xx, yy]).astype(np.float32)
    projector = np.column_stack([x[yy, xx], y[yy, xx]]).astype(np.float32)
    tiles_x = xx // TILE_SIZE
    tiles_y = yy // TILE_SIZE
    heldout = (tiles_x * 17 + tiles_y * 31 + SEED) % 5 == 0
    rng = np.random.default_rng(SEED)
    train_indices = rng.choice(
        np.flatnonzero(~heldout), size=min(10000, int((~heldout).sum())), replace=False
    )
    test_indices = rng.choice(
        np.flatnonzero(heldout), size=min(10000, int(heldout.sum())), replace=False
    )
    train_cam, train_prj = camera[train_indices], projector[train_indices]
    test_cam, test_prj = camera[test_indices], projector[test_indices]
    homography, inliers = cv2.findHomography(
        train_cam, train_prj, cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000
    )
    if homography is None:
        raise RuntimeError("homography fit failed")
    predictions = {
        "homography": cv2.perspectiveTransform(test_cam[:, None, :], homography)[
            :, 0, :
        ]
    }
    for degree in (2, 3):
        coefficients = robust_polynomial(train_cam, train_prj, degree)
        predictions[f"robust_polynomial_degree_{degree}"] = (
            design(test_cam, degree) @ coefficients
        ).astype(np.float32)
    result = {
        "setup": manifest["setup"],
        "threshold": CONTRAST,
        "tile_size_camera_px": TILE_SIZE,
        "tile_rule": "(tile_x*17+tile_y*31+9182)%5==0",
        "total_confident_decodes": int(len(camera)),
        "train_points": int(len(train_indices)),
        "heldout_points": int(len(test_indices)),
        "homography_train_inlier_fraction": float(inliers.mean()),
        "heldout": {
            name: summarize(pred, test_prj) for name, pred in predictions.items()
        },
    }
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
