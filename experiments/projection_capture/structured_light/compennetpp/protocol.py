"""Decode one real projector-camera Gray-code capture sequence.

The source pattern stack itself defines 10-bit codebooks. No assumption about
Gray-code polarity/order is needed. Output coordinates are projector *pixels*,
not world depth. Only run after 84 selected members pass ZIP CRC.
"""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT

from palimpsest.paths import WORK_DIR

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


FOLDER = DATA_ROOT / "derived/compennetpp_sl_one_setup"
MANIFEST = WORK_DIR / "compennetpp_sl_one_setup_manifest.json"
OUT = WORK_DIR / "compennetpp_sl_decoding.json"
VIS = WORK_DIR / "compennetpp_sl_decoding_preview.png"


def load_at(folder: Path, kind: str, index: int) -> np.ndarray:
    with Image.open(folder / f"{kind}_{index:04d}.png") as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255


def load(kind: str, index: int) -> np.ndarray:
    return load_at(FOLDER, kind, index)


def gray(image: np.ndarray) -> np.ndarray:
    return image @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def decode_axis(
    kind: str,
    first: int,
    count: int,
    source_shape: tuple[int, int],
    axis: int,
    image_loader=load,
) -> tuple[np.ndarray, np.ndarray, dict]:
    h, w = source_shape
    source_codes = np.zeros(w if axis == 1 else h, dtype=np.uint16)
    captured_codes = None
    min_contrast = None
    complement = []
    invariant = []
    for bit in range(count):
        pos_idx = first + 2 * bit
        neg_idx = pos_idx + 1
        pos_source = gray(image_loader("source", pos_idx))
        neg_source = gray(image_loader("source", neg_idx))
        if axis == 1:
            line = pos_source[h // 2] > neg_source[h // 2]
            invariant.append(bool(np.all((pos_source > neg_source) == line[None, :])))
        else:
            line = pos_source[:, w // 2] > neg_source[:, w // 2]
            invariant.append(bool(np.all((pos_source > neg_source) == line[:, None])))
        complement.append(float(np.mean(np.abs(pos_source + neg_source - 1))))
        source_codes |= line.astype(np.uint16) << bit
        pos_camera = gray(image_loader("capture", pos_idx))
        neg_camera = gray(image_loader("capture", neg_idx))
        diff = pos_camera - neg_camera
        if captured_codes is None:
            captured_codes = np.zeros(pos_camera.shape, dtype=np.uint16)
            min_contrast = np.abs(diff)
        captured_codes |= (diff > 0).astype(np.uint16) << bit
        min_contrast = np.minimum(min_contrast, np.abs(diff))
    if not all(invariant):
        raise ValueError(f"source patterns do not vary along expected axis {axis}")
    if len(np.unique(source_codes)) != len(source_codes):
        raise ValueError(f"source coordinate codes not unique along axis {axis}")
    lookup = np.full(1 << count, -1, dtype=np.int16)
    lookup[source_codes] = np.arange(len(source_codes), dtype=np.int16)
    decoded = lookup[captured_codes]
    info = {
        "coordinate_count": len(source_codes),
        "source_code_unique": True,
        "source_pair_complement_mae": complement,
        "source_axis_invariant": invariant,
        "camera_min_pair_contrast_quantiles": np.quantile(
            min_contrast, [0.1, 0.5, 0.9]
        ).tolist(),
    }
    return decoded, min_contrast, info


def geometry(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict:
    yy, xx = np.nonzero(mask)
    if len(xx) < 1000:
        return {"valid_points": int(len(xx)), "homography": None}
    rng = np.random.default_rng(425)
    take = rng.choice(len(xx), size=min(len(xx), 10000), replace=False)
    projector = np.column_stack([x[yy[take], xx[take]], y[yy[take], xx[take]]]).astype(
        np.float32
    )
    camera = np.column_stack([xx[take], yy[take]]).astype(np.float32)
    transform, inliers = cv2.findHomography(
        projector, camera, cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000
    )
    if transform is None:
        return {
            "valid_points": int(len(xx)),
            "sample_count": len(take),
            "homography": None,
        }
    projected = cv2.perspectiveTransform(projector[:, None, :], transform)[:, 0, :]
    residual = np.linalg.norm(projected - camera, axis=1)
    return {
        "valid_points": int(len(xx)),
        "sample_count": len(take),
        "homography": transform.tolist(),
        "ransac_inlier_fraction": float(inliers.mean()),
        "all_sample_residual_px_quantiles": np.quantile(
            residual, [0.1, 0.5, 0.9]
        ).tolist(),
        "ransac_inlier_residual_px_quantiles": np.quantile(
            residual[inliers[:, 0] > 0], [0.1, 0.5, 0.9]
        ).tolist(),
    }


def decode_setup(
    manifest_path: Path, folder: Path, output_path: Path, preview_path: Path
) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest["pairs_complete"] or len(manifest["rows"]) != 84:
        raise ValueError("selected structured-light pairs are not fully CRC verified")
    image_loader = lambda kind, index: load_at(folder, kind, index)
    source_shape = image_loader("source", 1).shape[:2]
    x, contrast_x, xinfo = decode_axis(
        "x", 3, 10, source_shape, axis=1, image_loader=image_loader
    )
    y, contrast_y, yinfo = decode_axis(
        "y", 23, 10, source_shape, axis=0, image_loader=image_loader
    )
    white = gray(image_loader("capture", 1))
    black = gray(image_loader("capture", 2))
    illuminated = (white - black) > 0.12
    valid_base = illuminated & (x >= 0) & (y >= 0)
    thresholds = [0.0, 0.01, 0.02, 0.04]
    masks = {
        str(value): valid_base & (contrast_x > value) & (contrast_y > value)
        for value in thresholds
    }
    result = {
        "setup": manifest["setup"],
        "source_size": [source_shape[1], source_shape[0]],
        "camera_size": [x.shape[1], x.shape[0]],
        "source_axis_x": xinfo,
        "source_axis_y": yinfo,
        "white_minus_black_above_0_12_fraction": float(illuminated.mean()),
        "valid_fraction_of_camera_by_min_pair_contrast": {
            key: float(mask.mean()) for key, mask in masks.items()
        },
        "valid_fraction_of_white_black_mask_by_min_pair_contrast": {
            key: float(mask.sum() / illuminated.sum()) for key, mask in masks.items()
        },
        "geometry_at_0_02_contrast": geometry(x, y, masks["0.02"]),
    }
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    preview = np.zeros((*x.shape, 3), dtype=np.uint8)
    mask = masks["0.02"]
    preview[..., 0][mask] = np.clip(
        x[mask] / (source_shape[1] - 1) * 255, 0, 255
    ).astype(np.uint8)
    preview[..., 1][mask] = np.clip(
        y[mask] / (source_shape[0] - 1) * 255, 0, 255
    ).astype(np.uint8)
    preview[..., 2][mask] = 128
    Image.fromarray(preview).save(preview_path)
    return result
