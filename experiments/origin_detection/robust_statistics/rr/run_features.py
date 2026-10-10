"""Extract registered candidates once, with audited coverage and a timed pilot."""

import argparse
import csv
from datetime import datetime, timezone
import io
import random
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from palimpsest.detection.algorithms.local_statistics.features import FEATURE_NAMES, bounded_rgb, extract_features
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT
from .protocol import CONFIG, RESULT, ROOT, settings, verified_inventory, write_json


def code_fingerprints():
    return {name: file_sha256(REPO_ROOT / name) for name in (
        "src/palimpsest/detection/algorithms/local_statistics/features.py",
        "src/palimpsest/detection/algorithms/local_statistics/detector.py",
        "experiments/origin_detection/robust_statistics/rr/run_features.py",
        "experiments/origin_detection/robust_statistics/rr/protocol.py",
        "experiments/origin_detection/robust_statistics/rr/README.md",
        "configs/evaluation/rr_robust_statistics.toml",
    )}


def jpeg_control(rgb, edge):
    # Encoding control acts after the registered area resize. It does not undo
    # earlier JPEG history and does not stand in for a real propagation path.
    stream = io.BytesIO()
    Image.fromarray(bounded_rgb(rgb, edge)).save(stream, "JPEG", quality=90, subsampling=0)
    stream.seek(0)
    with Image.open(stream) as opened:
        return np.asarray(opened.convert("RGB"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="0=full; positive=independent timing pilot")
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("Limit must be nonnegative")
    config = settings()
    rows, audit = verified_inventory()
    random.Random(config["seed"]).shuffle(rows)
    rows = rows[:args.limit] if args.limit else rows
    prefix = f"features_pilot_{args.limit}" if args.limit else "features"
    csv_path, receipt_path = RESULT / f"{prefix}.csv", RESULT / f"{prefix}.json"
    if csv_path.exists() or receipt_path.exists():
        raise FileExistsError("Feature run exists; no silent overwrite or unverified resume")
    cv2.setNumThreads(1)
    expected_rows = len(rows) * len(config["long_edges"]) * 2
    fields = ["filename", "src", "label", "condition", "role", "long_edge", "variant",
              "width", "height", "patches", "decode_ms", "preprocess_ms", "statistics_ms",
              "analysis_end_to_end_ms", *FEATURE_NAMES]
    started = datetime.now(timezone.utc).isoformat()
    start, completed, progress = perf_counter(), 0, []
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            path = ROOT / row["filename"]
            if file_sha256(path) != row["sha256"]:
                raise ValueError(f"Image fingerprint changed: {path}")
            decode_start = perf_counter()
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"))
            decode_ms = (perf_counter() - decode_start) * 1000
            if rgb.shape[1] != int(row["width"]) or rgb.shape[0] != int(row["height"]):
                raise ValueError("Decoded dimensions differ from audit")
            for edge in config["long_edges"]:
                for variant in ("raw", "jpeg90_444_after_resize"):
                    analyze_start = perf_counter()
                    array = rgb if variant == "raw" else jpeg_control(rgb, edge)
                    result = extract_features(array, edge)
                    writer.writerow({
                        "filename": row["filename"], "src": row["source_group"],
                        "label": "FAKE" if row["label"] == "ai" else "REAL",
                        "condition": row["condition"], "role": row["algorithm_role"],
                        "long_edge": edge, "variant": variant, "width": rgb.shape[1],
                        "height": rgb.shape[0], "patches": result.patches, "decode_ms": decode_ms,
                        "preprocess_ms": result.preprocess_ms, "statistics_ms": result.statistics_ms,
                        "analysis_end_to_end_ms": decode_ms + (perf_counter() - analyze_start) * 1000,
                        **dict(zip(FEATURE_NAMES, result.values.tolist())),
                    })
                    completed += 1
            if index % 30 == 0 or index == len(rows):
                stream.flush()
                item = {"images": index, "rows": completed, "elapsed_seconds": perf_counter() - start}
                progress.append(item)
                print(f"{datetime.now(timezone.utc).isoformat()} {index}/{len(rows)} images "
                      f"{completed}/{expected_rows} rows {item['elapsed_seconds']:.1f}s", flush=True)
    if completed != expected_rows:
        raise ValueError("Incomplete feature output")
    write_json(receipt_path, {
        "scope": "timing pilot" if args.limit else "registered development screening, not final test",
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "images": len(rows), "rows": completed, "csv_sha256": file_sha256(csv_path),
        "development_manifest_sha256": audit["development_manifest_sha256"],
        "config_sha256": file_sha256(CONFIG), "code_fingerprints": code_fingerprints(),
        "elapsed_seconds": perf_counter() - start, "progress": progress,
        "opencv_threads": cv2.getNumThreads(), "feature_names": list(FEATURE_NAMES),
        "timing_scope": "decode plus all three feature families; input SHA verification excluded; "
                        "JPEG diagnostic also includes in-memory encoding; no calibrated decision here",
    })


if __name__ == "__main__":
    main()
