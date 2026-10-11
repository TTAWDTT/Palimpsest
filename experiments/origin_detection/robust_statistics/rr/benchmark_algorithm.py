"""Measure the selected prototype after extraction ends, checking score parity."""

import argparse
import csv
import json
import platform
import random
from time import perf_counter

import cv2
import numpy as np

from palimpsest.detection.algorithms.local_statistics.detector import LocalStatisticsDetector, StatisticsRule
from palimpsest.detection.algorithms.local_statistics.projection import PairedProjectionDetector, ProjectionRule
from palimpsest.detection.algorithms.local_statistics.features import FEATURE_NAMES
from palimpsest.detection.files import predict_file
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from .protocol import RESULT, ROOT, read_csv, settings, verified_inventory, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", action="store_true", help="Benchmark the gated paired projection on the same 60 images")
    args = parser.parse_args()
    if args.projection:
        output_dir = WORK_DIR / "robust_statistics/rr_paired_projection"
        iteration = json.loads((output_dir / "iteration.json").read_text(encoding="utf-8"))
        chosen = iteration["chosen"]
        if chosen is None or not iteration["candidates"][chosen]["selection_gate"]:
            raise ValueError("No projection candidate passed")
        for name, fingerprint in iteration["code_fingerprints"].items():
            if file_sha256(REPO_ROOT / name) != fingerprint:
                raise ValueError("Registered projection code/config changed")
        rule_path = output_dir / f"{chosen}_rule.json"
        if file_sha256(rule_path) != iteration["rule_files"][chosen]["sha256"]:
            raise ValueError("Frozen projection rule changed")
        rule = ProjectionRule.load(rule_path)
        detector = PairedProjectionDetector(rule)
    else:
        output_dir, rule_path = RESULT, RESULT / "selected_rule.json"
        rule = StatisticsRule.load(rule_path)
        detector = LocalStatisticsDetector(rule)
    if (output_dir / "benchmark.json").exists():
        raise FileExistsError("Benchmark exists; preserve measured run")
    inventory, audit = verified_inventory()
    config = settings()
    random.Random(config["seed"] + 2).shuffle(inventory)
    selected = []
    for condition in config["conditions"]:
        for label in ("ai", "real"):
            selected.extend([r for r in inventory if r["condition"] == condition and r["label"] == label][:10])
    feature_receipt = json.loads((RESULT / "features.json").read_text(encoding="utf-8"))
    if feature_receipt["csv_sha256"] != file_sha256(RESULT / "features.csv"):
        raise ValueError("Benchmark feature cache changed")
    if feature_receipt["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Benchmark feature schema differs")
    cached = {r["filename"]: np.array([float(r[f]) for f in FEATURE_NAMES])
        for r in read_csv(RESULT / "features.csv")
        if r["variant"] == "raw" and int(r["long_edge"]) == rule.long_edge}
    cv2.setNumThreads(1)
    # Explicit warm-up; production timings exclude startup and SHA verification.
    detector.predict(np.full((256, 256, 3), 128, np.uint8))
    rows, start = [], perf_counter()
    for item in selected:
        path = ROOT / item["filename"]
        if file_sha256(path) != item["sha256"]:
            raise ValueError("Benchmark image fingerprint changed")
        for repeat in range(3):
            predicted = predict_file(detector, path)
            expected_score = float(rule.score(cached[item["filename"]]))
            if not np.isclose(predicted.prediction.score, expected_score, rtol=1e-10, atol=1e-10):
                raise ValueError("Prototype changed cached full-feature score")
            rows.append({"filename": item["filename"], "condition": item["condition"], "repeat": repeat,
                         "pixels": predicted.width * predicted.height,
                         "decode_ms": predicted.decode_ms, "end_to_end_ms": predicted.end_to_end_ms,
                         "predict_ms": predicted.prediction.timing_ms["predict_ms"]})
    summaries = {}
    for name, chosen in {"all": rows,
                         "at_most_2MP": [r for r in rows if r["pixels"] <= 2_000_000],
                         "above_2MP": [r for r in rows if r["pixels"] > 2_000_000]}.items():
        summaries[name] = {"measurements": len(chosen), "images": len({r["filename"] for r in chosen})}
        if chosen:
            summaries[name].update({key: dict(zip(("p50", "p95"), np.quantile(
                [r[key] for r in chosen], [0.5, 0.95]).tolist()))
                                   for key in ("decode_ms", "predict_ms", "end_to_end_ms")})
    path = output_dir / "benchmark.csv"
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "benchmark.json", {
        "selected_rule_sha256": file_sha256(rule_path),
        "rule_kind": "paired_projection" if args.projection else "color_rule",
        "development_manifest_sha256": audit["development_manifest_sha256"],
        "processor": platform.processor(), "platform": platform.platform(),
        "opencv_threads": cv2.getNumThreads(), "opencv_version": cv2.__version__,
        "numpy_version": np.__version__, "python_version": platform.python_version(),
        "summaries": summaries, "csv_sha256": file_sha256(path),
        "elapsed_seconds": perf_counter() - start,
        "scope": "60 development images, 3 repeats, CPU batch1; filesystem cache warm after SHA read; "
                 "same cached scores; parameter/model startup excluded; not whole RR or phone throughput",
    })
    print(summaries, flush=True)


if __name__ == "__main__":
    main()
