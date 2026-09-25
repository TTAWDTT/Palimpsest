"""Audit the first frames of the frozen Raw2Event calibration/development split.

The fixed content quadrilateral is a preregistered approximate registration
probe. Low scores are flags for follow-up, not grounds for dropping images.
"""

import argparse
import csv
import io
import json
from pathlib import Path
import pickle
import tarfile
import warnings

import cv2
import numpy as np

from work.audit_raw2event_probe import ROOT, WIDTH, HEIGHT, detect_tag, extract_frame
from work.fetch_cifar10_python import TARGET as CIFAR_ARCHIVE
from work.match_raw2event_cifar_source import query_from_photo, normalized_gray


SPLIT = Path("E:/ai_image_origin_research/data/manifests/raw2event_process_split_v1.csv")
AUDIT_DIR = Path("work/raw2event_process_split_downloads")
OUT = Path("work/raw2event_process_split_first_frame_audit.json")
SOURCE_QUAD = [[237, 180], [409, 195], [397, 370], [219, 353]]


def load_originals(rows: list[dict]) -> dict[str, np.ndarray]:
    by_batch = {}
    with tarfile.open(CIFAR_ARCHIVE, "r:gz") as archive:
        for batch in sorted({row["cifar_batch"] for row in rows}):
            with archive.extractfile(f"cifar-10-batches-py/{batch}") as stream:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=r"dtype\(\): align should be passed.*")
                    payload = pickle.load(io.BytesIO(stream.read()), encoding="bytes")
            by_batch[batch] = np.asarray(payload[b"data"], dtype=np.uint8)
    return {row["prefix"]: by_batch[row["cifar_batch"]][int(row["row"])].reshape(3, 32, 32).transpose(1, 2, 0)
            for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    rows = [row for row in csv.DictReader(SPLIT.open(encoding="utf-8", newline=""))
            if row["role"] in ("calibration", "development")]
    if len(rows) != 20:
        raise RuntimeError("expected the frozen 20-prefix calibration/development split")
    originals = load_originals(rows)
    results = []
    for row in rows:
        prefix = row["prefix"]
        audit_path = AUDIT_DIR / f"{prefix}.json"
        if not audit_path.exists():
            if args.allow_partial:
                continue
            raise RuntimeError(f"download audit missing for {prefix}")
        download_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not download_audit.get("complete_verified"):
            raise RuntimeError(f"unverified download for {prefix}")
        rgb_path = ROOT / "frames_rgb" / f"{prefix}.mkv"
        rgb = extract_frame(rgb_path, 0, "rgb24", 3, "u1")
        tag_corners, tag_id = detect_tag(rgb)
        if tag_id != 0:
            raise RuntimeError(f"unexpected tag ID for {prefix}")
        tmp = Path("work/raw2event_split_audit_frame0.png")
        cv2.imwrite(str(tmp), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        query = query_from_photo(tmp, SOURCE_QUAD)
        source = originals[prefix]
        corr = float(normalized_gray(query[None])[0] @ normalized_gray(source[None])[0])
        results.append({"prefix": prefix, "role": row["role"], "class_name": row["class_name"],
                        "capture_day": row["capture_day"], "source_key": row["cifar_source_key"],
                        "fixed_quad_gray_zncc": corr,
                        "tag_center_xy": tag_corners.mean(axis=0).tolist(),
                        "tag_edge_px_mean": float(np.linalg.norm(np.roll(tag_corners, -1, axis=0)
                                                                  - tag_corners, axis=1).mean()),
                        "rgb_first_frame_shape": list(rgb.shape)})
    scores = [row["fixed_quad_gray_zncc"] for row in results]
    report = {"scope": "frozen calibration/development prefixes, first RGB frame and fixed approximate content quad",
              "count": len(results), "scores_min_median_max": [float(v) for v in (min(scores), np.median(scores), max(scores))],
              "below_0_6": [row["prefix"] for row in results if row["fixed_quad_gray_zncc"] < 0.6],
              "rows": results,
              "complete_split": len(results) == 20,
              "qualification": "association/geometry quality flag only; do not reject low-score images post hoc"}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("count", "scores_min_median_max", "below_0_6")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
