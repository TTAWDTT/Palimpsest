"""Frozen-rule external diagnostic; no refitting or condition-specific threshold."""

import argparse
import csv
from datetime import datetime, timezone
import json
import random
from time import perf_counter

import cv2

from palimpsest.contracts import Origin
from palimpsest.detection.algorithms.local_statistics.detector import LocalStatisticsDetector, StatisticsRule
from palimpsest.detection.algorithms.local_statistics.projection import PairedProjectionDetector, ProjectionRule
from palimpsest.detection.files import predict_file
from palimpsest.evaluation.cached import CachedRun, ExpectedImage, evaluate_cached_method
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import DATA_ROOT, REPO_ROOT, WORK_DIR

MANIFEST = DATA_ROOT / "manifests/chimera_bfree_manifest.csv"
AUDIT = DATA_ROOT / "manifests/chimera_paired_image_audit.csv"
ROOT = DATA_ROOT / "derived/chimera_paired"
RULE = WORK_DIR / "robust_statistics/rr_first_iteration/selected_rule.json"
SCREENING = WORK_DIR / "robust_statistics/rr_first_iteration/screening.json"
OUTPUT = WORK_DIR / "robust_statistics/chimera_first_iteration"
CONDITIONS = {"stylegan2_orig": "original", "recap_mac": "mac_iphone", "recap_monitor": "lg_blackfly"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--projection", action="store_true", help="Use the gated second-iteration frozen rule")
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("Limit must be nonnegative")
    if file_sha256(MANIFEST) != "39da9e9eac3bd2d133c53769f9c5a158c1769039c67560c0bbfc4660d09a7d24":
        raise ValueError("Chimera manifest changed")
    if args.projection:
        iteration_dir = WORK_DIR / "robust_statistics/rr_paired_projection"
        selection_path = iteration_dir / "iteration.json"
        screening = json.loads(selection_path.read_text(encoding="utf-8"))
        chosen = screening["chosen"]
        if chosen is None or not screening["candidates"][chosen]["selection_gate"]:
            raise ValueError("No projection candidate passed the registered development gate")
        for name, fingerprint in screening["code_fingerprints"].items():
            if file_sha256(REPO_ROOT / name) != fingerprint:
                raise ValueError("Registered projection code/config changed")
        rule_path = iteration_dir / f"{chosen}_rule.json"
        fingerprint = screening["rule_files"][chosen]["sha256"]
        rule = ProjectionRule.load(rule_path)
        detector = PairedProjectionDetector(rule)
        output_dir = WORK_DIR / "robust_statistics/chimera_paired_projection"
    else:
        selection_path = SCREENING
        screening = json.loads(SCREENING.read_text(encoding="utf-8"))
        rule_path, fingerprint = RULE, screening["selected_rule_sha256"]
        rule = StatisticsRule.load(rule_path)
        detector = LocalStatisticsDetector(rule)
        output_dir = OUTPUT
    if file_sha256(rule_path) != fingerprint:
        raise ValueError("Frozen RR rule changed")
    rows = read_rows(MANIFEST)
    audited = {row["filename"]: row for row in read_rows(AUDIT)}
    if len(rows) != 3600 or len(audited) != 3600 or {r["filename"] for r in rows} != audited.keys():
        raise ValueError("Chimera inventory differs")
    if len({r["src"] for r in rows}) != 1200:
        raise ValueError("Chimera source counts differ")
    for row in rows:
        audit = audited[row["filename"]]
        if any(row[key] != audit[key] for key in ("src", "label", "condition")):
            raise ValueError("Chimera label/condition differs from audit")
    expected = [ExpectedImage(r["src"], CONDITIONS[r["condition"]],
                              Origin.AI if r["label"] == "FAKE" else Origin.NATURAL,
                              r["filename"]) for r in rows]
    random.Random(20261005).shuffle(rows)
    rows = rows[:args.limit] if args.limit else rows
    prefix = f"pilot_{args.limit}" if args.limit else "frozen_rule"
    path, receipt = output_dir / f"{prefix}.csv", output_dir / f"{prefix}.json"
    if path.exists() or receipt.exists():
        raise FileExistsError("External run exists; no silent overwrite/resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    started, start, progress = datetime.now(timezone.utc).isoformat(), perf_counter(), []
    with path.open("x", encoding="utf-8", newline="") as stream:
        fields = ["filename", "src", "label", "condition", "score", "threshold", "width", "height",
                  "decode_ms", "predict_ms", "end_to_end_ms", "error"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            image_path = ROOT / row["filename"]
            if file_sha256(image_path) != audited[row["filename"]]["sha256"]:
                raise ValueError("Chimera image fingerprint changed")
            prediction = predict_file(detector, image_path)
            writer.writerow({**row, "score": prediction.prediction.score, "threshold": rule.threshold,
                             "width": prediction.width, "height": prediction.height,
                             "decode_ms": prediction.decode_ms, "predict_ms": prediction.prediction.timing_ms["predict_ms"],
                             "end_to_end_ms": prediction.end_to_end_ms, "error": ""})
            if index % 30 == 0 or index == len(rows):
                stream.flush()
                item = {"images": index, "elapsed_seconds": perf_counter() - start}
                progress.append(item)
                print(f"{datetime.now(timezone.utc).isoformat()} {index}/{len(rows)} "
                      f"{item['elapsed_seconds']:.1f}s", flush=True)
    result = {"scope": "timing pilot" if args.limit else "frozen RR rule, full external dataset diagnostic",
              "images": len(rows), "started_utc": started, "elapsed_seconds": perf_counter() - start,
              "progress": progress, "rule_sha256": file_sha256(rule_path), "csv_sha256": file_sha256(path),
              "rule_kind": "paired_projection" if args.projection else "color_rule",
              "selection_receipt_sha256": file_sha256(selection_path),
              "runner_sha256": file_sha256(REPO_ROOT / "experiments/origin_detection/robust_statistics/chimera/run_algorithm.py"),
              "manifest_sha256": file_sha256(MANIFEST), "image_audit_sha256": file_sha256(AUDIT),
              "opencv_threads": cv2.getNumThreads(),
              "limitations": "Chimera previously used by project; only StyleGAN2/three scenes/two capture rigs; "
                             "RR parameters untouched; no new blind test or mobile throughput claim"}
    if not args.limit:
        result["algorithm"], _ = evaluate_cached_method(
            expected, [CachedRun(path, condition_map=CONDITIONS, expected_sha256=result["csv_sha256"])],
            method=detector.name, threshold=rule.threshold, score_kind="statistical_score")
        result["B-Free"], _ = evaluate_cached_method(
            expected, [CachedRun(WORK_DIR / "chimera_bfree_full.csv", condition_map=CONDITIONS,
                                expected_sha256="24012eacbd52b3d23c698c8f631edbf688e1560b836e859173413460f67e0d6e")],
            method="B-Free", score_kind="logit")
        if args.projection:
            old_receipt = json.loads((OUTPUT / "frozen_rule.json").read_text(encoding="utf-8"))
            result["previous_color_rule"], _ = evaluate_cached_method(
                expected, [CachedRun(OUTPUT / "frozen_rule.csv", condition_map=CONDITIONS,
                                    expected_sha256=old_receipt["csv_sha256"])],
                method="previous-color-rule", threshold=old_receipt["algorithm"]["threshold"],
                score_kind="statistical_score")
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
