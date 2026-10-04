"""Independent full-frame feature check of frozen calibration RAW/RGB affine.

Raw/RGB content is used only to diagnose geometry. No simulation weights or
held-out RAW prediction are fitted here.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from experiments.raw2event.audit_raw2event_probe import ROOT, extract_frame


GEOM = Path("work/raw2event_sensor_geometry_audit.json")
OUT = Path("work/raw2event_content_feature_geometry_audit.json")


def gray_views(prefix: str) -> tuple[np.ndarray, np.ndarray]:
    raw = extract_frame(ROOT / "frames_raw" / f"{prefix}.mkv", 0, "gray16le", 1, "<u2")
    rgb = extract_frame(ROOT / "frames_rgb" / f"{prefix}.mkv", 0, "rgb24", 3, "u1")
    raw_view = np.clip(raw.astype(np.float32) / 1023 * 255, 0, 255).astype(np.uint8)
    rgb_view = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(raw_view), clahe.apply(rgb_view)


def masked_tag(shape: tuple[int, int], points: list, margin: int = 15) -> np.ndarray:
    mask = np.full(shape, 255, np.uint8)
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 0)
    mask = cv2.erode(mask, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    return mask


def main() -> None:
    geom = json.loads(GEOM.read_text(encoding="utf-8"))
    global_affine = np.asarray(geom["fit_matrices"]["global_affine"], dtype=np.float64)
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.01)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    results = []
    for row in geom["samples"]:
        if row["role"] != "phase_confirmation":
            continue
        raw_view, rgb_view = gray_views(row["prefix"])
        kp_raw, des_raw = sift.detectAndCompute(
            raw_view, masked_tag(raw_view.shape, row["raw_tag"])
        )
        kp_rgb, des_rgb = sift.detectAndCompute(
            rgb_view, masked_tag(rgb_view.shape, row["rgb_tag"])
        )
        item = {
            "prefix": row["prefix"],
            "raw_keypoints": len(kp_raw),
            "rgb_keypoints": len(kp_rgb),
        }
        if des_raw is None or des_rgb is None:
            item["status"] = "no_descriptors"
            results.append(item)
            continue
        matches = [
            a
            for a, b in matcher.knnMatch(des_raw, des_rgb, k=2)
            if a.distance < 0.7 * b.distance
        ]
        item["ratio_matches"] = len(matches)
        if len(matches) < 6:
            item["status"] = "insufficient_matches"
            results.append(item)
            continue
        src = np.asarray([kp_raw[m.queryIdx].pt for m in matches], dtype=np.float32)
        dst = np.asarray([kp_rgb[m.trainIdx].pt for m in matches], dtype=np.float32)
        fitted, inliers = cv2.estimateAffine2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=3000
        )
        if fitted is None:
            item["status"] = "ransac_failed"
            results.append(item)
            continue
        mask = inliers.ravel().astype(bool)
        tag_h = cv2.getPerspectiveTransform(
            np.asarray(row["raw_tag"], np.float32),
            np.asarray(row["rgb_tag"], np.float32),
        )

        def pred(matrix: np.ndarray) -> np.ndarray:
            if matrix.shape == (2, 3):
                return cv2.transform(src[None], matrix)[0]
            return cv2.perspectiveTransform(src[None], matrix)[0]

        item.update(
            {
                "status": "ok",
                "ransac_inliers": int(mask.sum()),
                "inlier_span_raw_xy": (
                    src[mask].max(axis=0) - src[mask].min(axis=0)
                ).tolist(),
                "inlier_bbox_raw_xyxy": [
                    *src[mask].min(axis=0).tolist(),
                    *src[mask].max(axis=0).tolist(),
                ],
                "global_affine_inlier_reprojection_median_px": float(
                    np.median(
                        np.linalg.norm(pred(global_affine)[mask] - dst[mask], axis=1)
                    )
                ),
                "per_tag_homography_inlier_reprojection_median_px": float(
                    np.median(np.linalg.norm(pred(tag_h)[mask] - dst[mask], axis=1))
                ),
                "feature_affine_inlier_reprojection_median_px": float(
                    np.median(np.linalg.norm(pred(fitted)[mask] - dst[mask], axis=1))
                ),
                "feature_affine_vs_global_at_inliers_median_px": float(
                    np.median(
                        np.linalg.norm(
                            pred(fitted)[mask] - pred(global_affine)[mask], axis=1
                        )
                    )
                ),
                "feature_affine": fitted.tolist(),
            }
        )
        results.append(item)
        print(
            f"{len(results)}/10 {row['prefix']} inliers={int(mask.sum())}", flush=True
        )
    report = {
        "scope": "ten phase-confirmation first-frame RAW/ISP-RGB pairs; Tag excluded from SIFT matching",
        "geometry_source": "global affine fit on prior 10 calibration Tag corner sets",
        "diagnostic_only": True,
        "results": results,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            [
                {
                    k: v
                    for k, v in r.items()
                    if k
                    in (
                        "prefix",
                        "status",
                        "ratio_matches",
                        "ransac_inliers",
                        "inlier_span_raw_xy",
                        "global_affine_inlier_reprojection_median_px",
                        "per_tag_homography_inlier_reprojection_median_px",
                    )
                }
                for r in results
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
