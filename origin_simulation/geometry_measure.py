"""Measure the 2-D screen projection from a controlled checkerboard photo.

This does not infer camera distance, physical panel pixel pitch, or lens MTF.
The frame's colored corners disambiguate checkerboard orientation, so all four
markers must remain visible. OpenCV is an optional ``analysis`` dependency.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from origin_simulation.capture_kit import sha256, validate_kit


def _marker_centers(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    marker_size = max(12, min(width, height) // 30)
    positions = [
        (round(width * 0.025), round(height * 0.025)),
        (width - round(width * 0.025) - marker_size, round(height * 0.025)),
        (
            width - round(width * 0.025) - marker_size,
            height - round(height * 0.025) - marker_size,
        ),
        (round(width * 0.025), height - round(height * 0.025) - marker_size),
    ]
    centers = np.float32(
        [[x + marker_size / 2, y + marker_size / 2] for x, y in positions]
    )
    expected_rgb = np.float32([(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0)])
    return centers, expected_rgb


def _marker_score(
    image_rgb: np.ndarray,
    homography: np.ndarray,
    centers: np.ndarray,
    expected_rgb: np.ndarray,
) -> float:
    points = cv2.perspectiveTransform(centers.reshape(-1, 1, 2), homography).reshape(
        -1, 2
    )
    h, w = image_rgb.shape[:2]
    if not np.all(np.isfinite(points)) or any(
        x < 8 or x >= w - 8 or y < 8 or y >= h - 8 for x, y in points
    ):
        return float("inf")
    observed = []
    for x, y in points:
        x, y = round(float(x)), round(float(y))
        patch = image_rgb[y - 5 : y + 6, x - 5 : x + 6]
        rgb = np.median(patch.reshape(-1, 3), axis=0).astype(np.float64)
        observed.append(rgb / max(float(rgb.sum()), 1))
    return float(np.mean((np.asarray(observed) - expected_rgb) ** 2))


def measure(kit_dir: Path, image_path: Path) -> dict:
    manifest, errors = validate_kit(kit_dir)
    if errors:
        raise RuntimeError("; ".join(errors))
    info = next(row for row in manifest["patterns"] if row["pattern_id"] == "geometry")
    roi = next(row for row in info["rois"] if row["name"] == "checkerboard")
    x0, y0, x1, y1 = roi["xyxy"]
    cols, rows = roi["tiles"]
    grid_x = np.linspace(x0, x1, cols + 1, dtype=int)[1:-1]
    grid_y = np.linspace(y0, y1, rows + 1, dtype=int)[1:-1]
    expected = np.float32([(x, y) for y in grid_y for x in grid_x])
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot decode camera image: {image_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray, (cols - 1, rows - 1), flags=cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found or corners is None or len(corners) != len(expected):
        raise ValueError(
            f"expected {len(expected)} checkerboard inner corners; not found"
        )
    detected = corners.reshape(rows - 1, cols - 1, 2)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    centers, expected_rgb = _marker_centers(manifest["width"], manifest["height"])
    candidates = []
    for row_flip in (False, True):
        for col_flip in (False, True):
            candidate = detected[
                :: -1 if row_flip else 1, :: -1 if col_flip else 1
            ].reshape(-1, 2)
            H, mask = cv2.findHomography(expected, candidate, cv2.RANSAC, 3.0)
            if H is None or mask is None:
                continue
            score = _marker_score(rgb, H, centers, expected_rgb)
            projected = cv2.perspectiveTransform(expected.reshape(-1, 1, 2), H).reshape(
                -1, 2
            )
            error = np.linalg.norm(projected - candidate, axis=1)
            candidates.append(
                (score, float(np.median(error)), H, int(mask.sum()), row_flip, col_flip)
            )
    if not candidates:
        raise ValueError("could not fit a checkerboard homography")
    score, median_error, H, inliers, row_flip, col_flip = min(
        candidates, key=lambda row: row[0]
    )
    if not math.isfinite(score) or score >= 0.12:
        raise ValueError(
            f"corner colors cannot resolve checkerboard orientation; best score={score:.3f}"
        )
    cx, cy = manifest["width"] / 2, manifest["height"] / 2
    reference = np.float32([[[cx, cy]], [[cx + 1, cy]], [[cx, cy + 1]]])
    projected = cv2.perspectiveTransform(reference, H).reshape(3, 2)
    dx, dy = projected[1] - projected[0], projected[2] - projected[0]
    return {
        "kit_manifest_sha256": sha256(kit_dir / "pattern_manifest.json"),
        "camera_file": str(image_path),
        "camera_file_sha256": sha256(image_path),
        "image_size_px": [bgr.shape[1], bgr.shape[0]],
        "detected_inner_corners": len(expected),
        "homography_inliers": inliers,
        "median_reprojection_error_px": median_error,
        "marker_color_score": score,
        "detector_row_flip": row_flip,
        "detector_col_flip": col_flip,
        "display_to_camera_homography": H.tolist(),
        "projected_center_xy_px": projected[0].tolist(),
        "projected_x_axis_px_per_display_px": dx.tolist(),
        "projected_y_axis_px_per_display_px": dy.tolist(),
        "projected_x_scale_px_per_display_px": float(np.linalg.norm(dx)),
        "projected_y_scale_px_per_display_px": float(np.linalg.norm(dy)),
        "scope": "2-D projection from this image only; no physical pose or optics inferred",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.kit_dir, args.image)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "detected_inner_corners",
                    "homography_inliers",
                    "median_reprojection_error_px",
                    "marker_color_score",
                    "projected_x_scale_px_per_display_px",
                    "projected_y_scale_px_per_display_px",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
