"""Diagnose content registration using known source and captured ISP-RGB only.

This refinement never reads the RAW target. It is an exploratory nuisance
registration after inspecting the first development results, not a device
geometry measurement or a predeclared evaluation protocol.
"""

from palimpsest.paths import DATA_ROOT, WORK_DIR

import csv
import json

import cv2
import numpy as np
from scipy.optimize import minimize

from experiments.raw2event.audit_raw2event_probe import ROOT, extract_frame
from experiments.raw2event.audit_raw2event_split_first_frames import load_originals
from experiments.raw2event.match_raw2event_cifar_source import normalized_gray


SPLIT = DATA_ROOT / "manifests/raw2event_process_split_v1.csv"
BASE = WORK_DIR / "raw2event_process_split_first_frame_audit.json"
OUT = WORK_DIR / "raw2event_content_registered_geometry.json"
DEST = np.asarray([[0, 0], [31, 0], [31, 31], [0, 31]], dtype=np.float32)
BOUNDS = [(-20, 20), (-20, 20), (0.90, 1.10), (-0.12, 0.12)]


def adjusted_corners(base: np.ndarray, params: np.ndarray) -> np.ndarray:
    dx, dy, scale, theta = params
    rotation = np.asarray(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    center = base.mean(axis=0)
    return (
        (base - center) @ rotation.T * scale + center + np.asarray([dx, dy])
    ).astype(np.float32)


def score(rgb: np.ndarray, source_feature: np.ndarray, corners: np.ndarray) -> float:
    transform = cv2.getPerspectiveTransform(corners, DEST)
    query = cv2.warpPerspective(rgb, transform, (32, 32), flags=cv2.INTER_LINEAR)
    return float(normalized_gray(query[None])[0] @ source_feature)


def main() -> None:
    split = [
        row
        for row in csv.DictReader(SPLIT.open(encoding="utf-8", newline=""))
        if row["role"] in ("calibration", "development")
    ]
    originals = load_originals(split)
    audit = json.loads(BASE.read_text(encoding="utf-8"))
    first = {row["prefix"]: row for row in audit["rows"]}
    if (
        len(split) != 20
        or audit["count"] != 20
        or set(first) != {row["prefix"] for row in split}
    ):
        raise RuntimeError("complete frozen split registration required")
    records = []
    for row in split:
        prefix = row["prefix"]
        rgb = extract_frame(ROOT / "frames_rgb" / f"{prefix}.mkv", 0, "rgb24", 3, "u1")
        source_feature = normalized_gray(originals[prefix][None])[0]
        base = np.asarray(
            first[prefix]["tag_similarity_content_corners"], dtype=np.float32
        )
        before = score(rgb, source_feature, base)

        def objective(params: np.ndarray) -> float:
            return -score(rgb, source_feature, adjusted_corners(base, params))

        result = minimize(
            objective,
            [0, 0, 1, 0],
            method="Powell",
            bounds=BOUNDS,
            options={"xtol": 1e-3, "ftol": 1e-5, "maxiter": 80},
        )
        refined = adjusted_corners(base, result.x)
        after = score(rgb, source_feature, refined)
        if after < before - 1e-4:
            raise RuntimeError(f"registration became worse: {prefix}")
        records.append(
            {
                "prefix": prefix,
                "role": row["role"],
                "class_name": row["class_name"],
                "initial_zncc": before,
                "refined_zncc": after,
                "parameters_dx_dy_scale_radians": result.x.tolist(),
                "refined_corners": refined.tolist(),
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
            }
        )
        print(f"{len(records)}/20 {prefix}: {before:.3f} -> {after:.3f}", flush=True)
    report = {
        "scope": "known-source geometric nuisance registration from real ISP-RGB frame0",
        "bounds": BOUNDS,
        "raw_target_used": False,
        "selection_warning": "optimization designed after inspecting development RAW errors; exploratory only",
        "n": len(records),
        "records": records,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "min_before": min(r["initial_zncc"] for r in records),
                "min_after": min(r["refined_zncc"] for r in records),
                "median_after": float(np.median([r["refined_zncc"] for r in records])),
            }
        )
    )


if __name__ == "__main__":
    main()
