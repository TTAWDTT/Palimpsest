"""Check whether one calibration-only RAW-to-ISP transform transfers to new captures.

This is a geometric diagnostic, not a source-to-RAW simulation score. Only
AprilTag corners are used; original 10 reserved and 3 cross-day stress
recordings are excluded before image reads.
"""

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from work.audit_raw2event_probe import ROOT, detect_tag, extract_frame


SPLIT = Path("E:/ai_image_origin_research/data/manifests/raw2event_process_split_v1.csv")
NEW = Path("E:/ai_image_origin_research/data/manifests/raw2event_phase_confirmation_v1.csv")
OUT = Path("work/raw2event_sensor_geometry_audit.json")


def read_rows() -> list[dict]:
    rows = [r for r in csv.DictReader(SPLIT.open(encoding="utf-8", newline=""))
            if r["role"] in ("calibration", "development")]
    rows += list(csv.DictReader(NEW.open(encoding="utf-8", newline="")))
    roles = [r["role"] for r in rows]
    if len(rows) != 30 or any(roles.count(role) != 10 for role in
                            ("calibration", "development", "phase_confirmation")):
        raise RuntimeError("expected 10+10+10 permitted recordings")
    if len({r["prefix"] for r in rows}) != 30:
        raise RuntimeError("duplicate prefix")
    return rows


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (2, 3):
        return cv2.transform(points[None].astype(np.float32), matrix)[0]
    return cv2.perspectiveTransform(points[None].astype(np.float32), matrix)[0]


def main() -> None:
    rows = read_rows()
    samples = []
    for i, row in enumerate(rows, 1):
        prefix = row["prefix"]
        raw = extract_frame(ROOT / "frames_raw" / f"{prefix}.mkv", 0, "gray16le", 1, "<u2")
        rgb = extract_frame(ROOT / "frames_rgb" / f"{prefix}.mkv", 0, "rgb24", 3, "u1")
        raw_view = np.minimum(raw.astype(np.float32) * (255 / 1023), 255).astype(np.uint8)
        raw_tag, rid = detect_tag(raw_view)
        rgb_tag, iid = detect_tag(rgb)
        if rid != 0 or iid != 0:
            raise RuntimeError(f"unexpected Tag IDs: {prefix}")
        samples.append({"prefix": prefix, "role": row["role"], "capture_day": row["capture_day"],
                        "raw_tag": raw_tag.tolist(), "rgb_tag": rgb_tag.tolist()})
        print(f"tags {i}/30", flush=True)
    cal = [r for r in samples if r["role"] == "calibration"]
    raw_cal = np.concatenate([np.asarray(r["raw_tag"], dtype=np.float32) for r in cal])
    rgb_cal = np.concatenate([np.asarray(r["rgb_tag"], dtype=np.float32) for r in cal])
    affine, _ = cv2.estimateAffine2D(raw_cal, rgb_cal, method=cv2.LMEDS, refineIters=10)
    similar, _ = cv2.estimateAffinePartial2D(raw_cal, rgb_cal, method=cv2.LMEDS, refineIters=10)
    projective, _ = cv2.findHomography(raw_cal, rgb_cal, method=0)
    if any(x is None for x in (affine, similar, projective)):
        raise RuntimeError("fit failed")
    fits = {"global_affine": affine, "global_similarity": similar, "global_projective": projective}
    rgb_source_corners = {}
    prior = Path("work/raw2event_source_to_raw_split_v1_source_rgb_refined.json")
    if prior.exists():
        old = json.loads(prior.read_text(encoding="utf-8"))
        for item in old.get("per_source", old.get("sources", [])):
            geometry = item.get("geometry", {})
            if "source_rgb_refined_corners" in geometry:
                rgb_source_corners[item["prefix"]] = geometry["source_rgb_refined_corners"]
    newer = Path("work/raw2event_phase_confirmation_evaluation.json")
    for item in json.loads(newer.read_text(encoding="utf-8"))["per_source"]:
        rgb_source_corners[item["prefix"]] = item["geometry"]["source_rgb_refined_corners"]
    for row in samples:
        raw = np.asarray(row["raw_tag"], dtype=np.float32)
        rgb = np.asarray(row["rgb_tag"], dtype=np.float32)
        row["tag_corner_rmse_rgb_px"] = {name: float(np.sqrt(np.mean(np.sum((transform(raw, matrix) - rgb)**2, axis=1))))
                                         for name, matrix in fits.items()}
        row["raw_tag_center"] = raw.mean(axis=0).tolist()
        row["rgb_tag_center"] = rgb.mean(axis=0).tolist()
        if row["prefix"] in rgb_source_corners:
            source = np.asarray(rgb_source_corners[row["prefix"]], dtype=np.float32)
            tag_h = cv2.getPerspectiveTransform(raw, rgb)
            source_raw_by_tag = transform(source, np.linalg.inv(tag_h))
            source_raw_by_global = transform(source, cv2.invertAffineTransform(affine))
            row["content_quad_global_vs_per_recording_tag_mean_distance_raw_px"] = float(
                np.mean(np.linalg.norm(source_raw_by_tag - source_raw_by_global, axis=1)))
        # A per-image Tag homography has zero fitting error by construction,
        # so it is not counted as an independent held-out geometry estimate.
    aggregates = {}
    for role in ("calibration", "development", "phase_confirmation"):
        subset = [r for r in samples if r["role"] == role]
        aggregates[role] = {name: {"mean_rmse": float(np.mean([r["tag_corner_rmse_rgb_px"][name] for r in subset])),
                                   "median_rmse": float(np.median([r["tag_corner_rmse_rgb_px"][name] for r in subset])),
                                   "max_rmse": float(max(r["tag_corner_rmse_rgb_px"][name] for r in subset))}
                            for name in fits}
        distances = [r["content_quad_global_vs_per_recording_tag_mean_distance_raw_px"] for r in subset
                     if "content_quad_global_vs_per_recording_tag_mean_distance_raw_px" in r]
        if distances:
            aggregates[role]["source_content_quad_mapping_disagreement"] = {
                "n": len(distances), "mean_raw_px": float(np.mean(distances)),
                "median_raw_px": float(np.median(distances)), "max_raw_px": float(max(distances))}
    report = {"scope": "first-frame AprilTag RAW to ISP-RGB coordinate mapping",
              "fit_role": "10 calibration prefixes only, 40 Tag corner pairs",
              "excluded": "original reserved 10 and cross-day stress reserved 3",
              "fit_matrices": {name: matrix.tolist() for name, matrix in fits.items()},
              "aggregate": aggregates, "samples": samples,
              "qualification": "Tag corner residual is a local geometric diagnostic; no content-corner or RAW-pixel simulation claim"}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregates, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
