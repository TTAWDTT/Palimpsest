"""Target-RAW geometry upper bound; diagnostic only, never a heldout score.

The true RAW frame is explicitly used to optimize source alignment. This can
locate a registration bottleneck but cannot validate a forward simulator.
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
from experiments.raw2event.probe_raw2event_source_to_raw import prepare
from experiments.raw2event.refine_raw2event_content_geometry import adjusted_corners


SPLIT = DATA_ROOT / "manifests/raw2event_process_split_v1.csv"
REFINED_RGB = WORK_DIR / "raw2event_content_registered_geometry.json"
OUT = WORK_DIR / "raw2event_raw_target_registration_diagnostic.json"
DEST = np.asarray([[0, 0], [31, 0], [31, 31], [0, 31]], dtype=np.float32)
BOUNDS = [(-20, 20), (-20, 20), (0.90, 1.10), (-0.12, 0.12)]


def main() -> None:
    split = [
        row
        for row in csv.DictReader(SPLIT.open(encoding="utf-8", newline=""))
        if row["role"] in ("calibration", "development")
    ]
    source = load_originals(split)
    rgb_refined = {
        row["prefix"]: row
        for row in json.loads(REFINED_RGB.read_text(encoding="utf-8"))["records"]
    }
    if len(split) != 20 or set(rgb_refined) != {row["prefix"] for row in split}:
        raise RuntimeError("complete split required")
    records = []
    for row in split:
        prefix = row["prefix"]
        prepared = prepare(
            prefix,
            None,
            source[prefix],
            np.asarray(rgb_refined[prefix]["refined_corners"]),
        )
        raw = extract_frame(
            ROOT / "frames_raw" / f"{prefix}.mkv", 0, "gray16le", 1, "<u2"
        )
        # Low pass suppresses Bayer checkerboard and sensor noise for location only.
        raw_view = cv2.GaussianBlur(raw.astype(np.float32), (0, 0), 1.5)
        raw_view = np.clip(raw_view / 1023 * 255, 0, 255).astype(np.uint8)
        target = normalized_gray(source[prefix][None])[0]
        base = np.asarray(prepared["raw_corners"], dtype=np.float32)

        def score(corners: np.ndarray) -> float:
            homography = cv2.getPerspectiveTransform(corners, DEST)
            query = cv2.warpPerspective(raw_view, homography, (32, 32))
            flat = query.astype(np.float32).ravel() / 255
            flat -= flat.mean()
            flat /= np.linalg.norm(flat) + 1e-9
            return float(flat @ target)

        before = score(base)
        result = minimize(
            lambda params: -score(adjusted_corners(base, params)),
            [0, 0, 1, 0],
            method="Powell",
            bounds=BOUNDS,
            options={"xtol": 1e-3, "ftol": 1e-5, "maxiter": 80},
        )
        after = score(adjusted_corners(base, result.x))
        records.append(
            {
                "prefix": prefix,
                "role": row["role"],
                "class_name": row["class_name"],
                "raw_zncc_before": before,
                "raw_zncc_after": after,
                "raw_target_adjustment_dx_dy_scale_radians": result.x.tolist(),
                "target_refined_raw_corners": adjusted_corners(base, result.x).tolist(),
            }
        )
        print(f"{len(records)}/20 {prefix}: {before:.3f} -> {after:.3f}", flush=True)
    report = {
        "n": len(records),
        "uses_target_raw": True,
        "purpose": "diagnose registration error only; not validation of simulator",
        "bounds": BOUNDS,
        "records": records,
        "median_before_after": [
            float(np.median([r["raw_zncc_before"] for r in records])),
            float(np.median([r["raw_zncc_after"] for r in records])),
        ],
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["median_before_after"]))


if __name__ == "__main__":
    main()
