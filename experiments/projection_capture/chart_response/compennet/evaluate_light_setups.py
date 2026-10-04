"""Compare true structured-light decodes on two cloud_np CompenNet++ setups.

The paper varies global lighting and pro-cam distance between setups, so this
is an observed setup comparison, not an isolated illumination intervention.
"""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT, WORK_DIR

import json
from pathlib import Path

import cv2
import numpy as np

from experiments.projection_capture.structured_light.compennetpp.protocol import (
    decode_axis,
    gray,
    load_at,
)
from experiments.projection_capture.structured_light.compennetpp.evaluate_geometry_surrogates import (
    design,
    robust_polynomial,
)


FIRST_MANIFEST = WORK_DIR / "compennetpp_sl_one_setup_manifest.json"
SECOND_MANIFEST = WORK_DIR / "compennetpp_sl_light3_pos1_cloud_manifest.json"
FIRST_FOLDER = DATA_ROOT / "derived/compennetpp_sl_one_setup"
SECOND_FOLDER = DATA_ROOT / "derived/compennetpp_sl_light3_pos1_cloud"
OUT = WORK_DIR / "compennetpp_cloud_two_setups_comparison.json"


def decode(folder: Path) -> dict:
    loader = lambda kind, index: load_at(folder, kind, index)
    shape = loader("source", 1).shape[:2]
    x, cx, _ = decode_axis("x", 3, 10, shape, axis=1, image_loader=loader)
    y, cy, _ = decode_axis("y", 23, 10, shape, axis=0, image_loader=loader)
    white = gray(loader("capture", 1))
    black = gray(loader("capture", 2))
    baseline = white - black
    valid = (baseline > 0.12) & (x >= 0) & (y >= 0)
    masks = {
        threshold: valid & (cx > threshold) & (cy > threshold)
        for threshold in (0.02, 0.04)
    }
    return {
        "x": x,
        "y": y,
        "cx": cx,
        "cy": cy,
        "white": white,
        "black": black,
        "baseline": baseline,
        "masks": masks,
    }


def image_registration(first: np.ndarray, second: np.ndarray) -> dict:
    image_a = np.uint8(np.clip(first * 255, 0, 255))
    image_b = np.uint8(np.clip(second * 255, 0, 255))
    detector = cv2.SIFT_create(nfeatures=3000)
    point_a, desc_a = detector.detectAndCompute(image_a, None)
    point_b, desc_b = detector.detectAndCompute(image_b, None)
    if desc_a is None or desc_b is None:
        return {"status": "no descriptors"}
    pairs = cv2.BFMatcher().knnMatch(desc_a, desc_b, k=2)
    good = [one for one, two in pairs if one.distance < 0.75 * two.distance]
    if len(good) < 12:
        return {"status": "too few ratio matches", "ratio_matches": len(good)}
    camera_a = np.float32([point_a[row.queryIdx].pt for row in good])
    camera_b = np.float32([point_b[row.trainIdx].pt for row in good])
    transform, inliers = cv2.findHomography(
        camera_a, camera_b, cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if transform is None:
        return {"status": "homography failed", "ratio_matches": len(good)}
    if int(inliers.sum()) < 30 or float(inliers.mean()) < 0.5:
        return {
            "status": "insufficient reliable correspondences for a camera-pose claim",
            "ratio_matches": len(good),
            "ransac_inliers": int(inliers.sum()),
        }
    transformed = cv2.perspectiveTransform(camera_a[:, None, :], transform)[:, 0, :]
    residual = np.linalg.norm(transformed - camera_b, axis=1)
    center = np.array([319.5, 239.5], np.float32)
    center_after = cv2.perspectiveTransform(center[None, None, :], transform)[0, 0]
    return {
        "status": "ok",
        "ratio_matches": len(good),
        "ransac_inliers": int(inliers.sum()),
        "inlier_median_camera_px": float(np.median(residual[inliers[:, 0] > 0])),
        "center_displacement_camera_px": (center_after - center).tolist(),
        "homography": transform.tolist(),
    }


def camera_locations_by_projector_code(
    decoded: dict, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    mask = decoded["masks"][threshold]
    camera_y, camera_x = np.nonzero(mask)
    width, height = 800, 600
    codes = decoded["y"][mask].astype(np.int32) * width + decoded["x"][mask]
    count = np.bincount(codes, minlength=width * height)
    sum_x = np.bincount(codes, weights=camera_x, minlength=width * height)
    sum_y = np.bincount(codes, weights=camera_y, minlength=width * height)
    locations = np.empty((width * height, 2), np.float32)
    locations[:, 0] = np.divide(sum_x, count, out=np.zeros_like(sum_x), where=count > 0)
    locations[:, 1] = np.divide(sum_y, count, out=np.zeros_like(sum_y), where=count > 0)
    return count > 0, locations


def code_matched_geometry(one: dict, two: dict, threshold: float) -> dict:
    found_one, camera_one = camera_locations_by_projector_code(one, threshold)
    found_two, camera_two = camera_locations_by_projector_code(two, threshold)
    shared = found_one & found_two
    if shared.sum() < 100:
        return {
            "shared_projector_codes": int(shared.sum()),
            "status": "too few shared codes",
        }
    first, second = camera_one[shared], camera_two[shared]
    raw_displacement = np.linalg.norm(first - second, axis=1)
    rng = np.random.default_rng(944)
    take = rng.choice(len(first), size=min(10000, len(first)), replace=False)
    transform, inliers = cv2.findHomography(
        first[take], second[take], cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000
    )
    if transform is None:
        raise RuntimeError("projector-code matched camera geometry failed")
    prediction = cv2.perspectiveTransform(first[take, None, :], transform)[:, 0, :]
    error = np.linalg.norm(prediction - second[take], axis=1)
    return {
        "shared_projector_codes": int(shared.sum()),
        "first_unique_projector_codes": int(found_one.sum()),
        "second_unique_projector_codes": int(found_two.sum()),
        "raw_camera_displacement_median_px": float(np.median(raw_displacement)),
        "raw_camera_displacement_p90_px": float(np.quantile(raw_displacement, 0.9)),
        "sampled_codes": len(take),
        "single_homography_ransac_inlier_fraction": float(inliers.mean()),
        "single_homography_all_residual_median_camera_px": float(np.median(error)),
        "single_homography_all_residual_p90_camera_px": float(np.quantile(error, 0.9)),
        "homography": transform.tolist(),
    }


def transfer_polynomial(one: dict, two: dict, threshold: float) -> dict:
    def points(decoded: dict) -> tuple[np.ndarray, np.ndarray]:
        yy, xx = np.nonzero(decoded["masks"][threshold])
        camera = np.column_stack([xx, yy]).astype(np.float32)
        projector = np.column_stack(
            [decoded["x"][yy, xx], decoded["y"][yy, xx]]
        ).astype(np.float32)
        return camera, projector

    camera_one, projector_one = points(one)
    camera_two, projector_two = points(two)
    rng = np.random.default_rng(944)
    calibration = rng.choice(
        len(camera_one), size=min(10000, len(camera_one)), replace=False
    )
    evaluation = rng.choice(
        len(camera_two), size=min(10000, len(camera_two)), replace=False
    )
    coefficients = robust_polynomial(
        camera_one[calibration], projector_one[calibration], degree=3
    )
    predicted = design(camera_two[evaluation], degree=3) @ coefficients
    error = np.linalg.norm(predicted - projector_two[evaluation], axis=1)
    return {
        "degree": 3,
        "first_setup_calibration_points": len(calibration),
        "second_setup_evaluation_points": len(evaluation),
        "second_setup_projector_residual_median_px": float(np.median(error)),
        "second_setup_projector_residual_p90_px": float(np.quantile(error, 0.9)),
        "warning": "second camera coverage differs, so this includes extrapolation as well as physical setup shift",
    }


def main() -> None:
    first_manifest = json.loads(FIRST_MANIFEST.read_text(encoding="utf-8"))
    second_manifest = json.loads(SECOND_MANIFEST.read_text(encoding="utf-8"))
    for manifest in (first_manifest, second_manifest):
        if not manifest["pairs_complete"] or len(manifest["rows"]) != 84:
            raise ValueError("a setup lacks 42 CRC-verified source/capture pairs")
    source_a = {
        row["index"]: row["sha256"]
        for row in first_manifest["rows"]
        if row["kind"] == "source"
    }
    source_b = {
        row["index"]: row["sha256"]
        for row in second_manifest["rows"]
        if row["kind"] == "source"
    }
    if source_a != source_b or len(source_a) != 42:
        raise ValueError("projected pattern sources differ between setups")
    one, two = decode(FIRST_FOLDER), decode(SECOND_FOLDER)
    comparisons = {}
    for threshold in (0.02, 0.04):
        first = one["masks"][threshold]
        second = two["masks"][threshold]
        shared = first & second
        delta = np.hypot(
            one["x"][shared].astype(float) - two["x"][shared],
            one["y"][shared].astype(float) - two["y"][shared],
        )
        comparisons[str(threshold)] = {
            "first_valid": int(first.sum()),
            "second_valid": int(second.sum()),
            "shared_valid": int(shared.sum()),
            "jaccard_valid_masks": float(shared.sum() / (first | second).sum()),
            "same_camera_pixel_exact_projector_coordinate_fraction": float(
                np.mean(delta == 0)
            )
            if len(delta)
            else None,
            "same_camera_pixel_projector_coordinate_delta_median": float(
                np.median(delta)
            )
            if len(delta)
            else None,
            "same_camera_pixel_projector_coordinate_delta_p90": float(
                np.quantile(delta, 0.9)
            )
            if len(delta)
            else None,
            "same_projector_code_camera_geometry": code_matched_geometry(
                one, two, threshold
            ),
        }
    result = {
        "first_setup": first_manifest["setup"],
        "second_setup": second_manifest["setup"],
        "identical_42_projected_patterns_sha256": True,
        "warning": "setup labels do not prove only lighting changed; paper states camera-projector distance can vary between setups",
        "white_capture_sift_registration": image_registration(
            one["white"], two["white"]
        ),
        "direct_same_camera_pixel_comparisons": comparisons,
        "light1_geometry_surrogate_applied_to_light3": transfer_polynomial(
            one, two, 0.04
        ),
        "first_white_mean": float(one["white"].mean()),
        "second_white_mean": float(two["white"].mean()),
        "first_black_mean": float(one["black"].mean()),
        "second_black_mean": float(two["black"].mean()),
    }
    OUT.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
