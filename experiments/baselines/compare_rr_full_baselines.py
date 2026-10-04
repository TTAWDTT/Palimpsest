"""Audit matched RRDataset predictions and compare full baseline results."""

from experiments.paths import REPO_ROOT

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = REPO_ROOT
WORK = ROOT / "work"
FILES = {
    "D3": WORK / "d3_rr_full.csv",
    "B-Free": WORK / "rr_bfree_complete.csv",
    "Benford-RF": WORK / "benford_rr_full.csv",
}
TEST_MANIFEST = Path(r"E:\ai_image_origin_research\data\manifests\rr_test_files.csv")
EXPECTED_COUNTS = {"original": 16986, "transfer": 16986, "redigital": 16985}
PAIRS = (("D3", "B-Free"), ("D3", "Benford-RF"), ("B-Free", "Benford-RF"))
EXCLUDED_SOURCES = set(
    json.loads((WORK / "rr_bfree_evaluation.json").read_text(encoding="utf-8"))[
        "trainval_overlap_audit"
    ]["excluded_source_ids"]
)


def read_predictions(path):
    predictions = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            filename = row["filename"]
            if filename in predictions:
                raise ValueError(f"duplicate filename in {path}: {filename}")
            score = float(row["score"])
            latency = float(row["end_to_end_ms"])
            if not math.isfinite(score) or not math.isfinite(latency) or latency < 0:
                raise ValueError(f"invalid score or timing in {path}: {filename}")
            condition = filename.split("/", 1)[0]
            if condition not in EXPECTED_COUNTS:
                raise ValueError(f"invalid condition: {filename}")
            if "condition" in row and row["condition"] != condition:
                raise ValueError(f"condition mismatch: {filename}")
            if row["label"] not in {"REAL", "FAKE"}:
                raise ValueError(f"invalid label: {filename}")
            if row["src"] in EXCLUDED_SOURCES:
                continue
            predictions[filename] = {
                "condition": condition,
                "source": row["src"],
                "label": row["label"],
                "score": score,
                "correct": (score > 0) == (row["label"] == "FAKE"),
                "latency_ms": latency,
            }
    return predictions


def bootstrap_difference(first, second, filenames, seed):
    generator = np.random.default_rng(seed)
    class_differences = {}
    for label in ("REAL", "FAKE"):
        values = np.asarray(
            [
                int(first[name]["correct"]) - int(second[name]["correct"])
                for name in filenames
                if first[name]["label"] == label
            ],
            dtype=np.int8,
        )
        if not len(values):
            raise ValueError(f"empty class {label}")
        estimates = np.empty(2000, dtype=np.float64)
        for begin in range(0, len(estimates), 100):
            indices = generator.integers(0, len(values), size=(100, len(values)))
            estimates[begin : begin + 100] = values[indices].mean(axis=1)
        class_differences[label] = {
            "observed": float(values.mean()),
            "bootstrap": estimates,
        }
    observed = (
        class_differences["REAL"]["observed"] + class_differences["FAKE"]["observed"]
    ) / 2
    replicate = (
        class_differences["REAL"]["bootstrap"] + class_differences["FAKE"]["bootstrap"]
    ) / 2
    return {
        "ba_difference": observed,
        "ci95": np.quantile(replicate, [0.025, 0.975]).tolist(),
    }


def main():
    predictions = {method: read_predictions(path) for method, path in FILES.items()}
    reference = predictions["D3"]
    expected = {}
    with TEST_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source = f"{row['label']}/{row['source_id']}"
            if source in EXCLUDED_SOURCES:
                continue
            filename = row["filename"]
            if filename in expected:
                raise ValueError(f"duplicate manifest filename: {filename}")
            expected[filename] = {
                "condition": row["condition"],
                "source": source,
                "label": {"ai": "FAKE", "real": "REAL"}[row["label"]],
            }
    if reference.keys() != expected.keys():
        raise ValueError(
            f"D3 filenames differ from independent manifest: "
            f"missing={len(expected.keys() - reference.keys())}, "
            f"extra={len(reference.keys() - expected.keys())}"
        )
    for filename, row in reference.items():
        if any(
            row[key] != expected[filename][key]
            for key in ("condition", "source", "label")
        ):
            raise ValueError(
                f"D3 metadata differs from independent manifest: {filename}"
            )
    for method, rows in predictions.items():
        if rows.keys() != reference.keys():
            raise ValueError(f"filename coverage differs: {method}")
        for filename, row in rows.items():
            if (row["condition"], row["source"], row["label"]) != (
                reference[filename]["condition"],
                reference[filename]["source"],
                reference[filename]["label"],
            ):
                raise ValueError(f"metadata mismatch: {method}: {filename}")
    counts = {
        condition: sum(row["condition"] == condition for row in reference.values())
        for condition in EXPECTED_COUNTS
    }
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"condition counts differ: {counts}")
    result = {
        "images": len(reference),
        "condition_counts": counts,
        "test_manifest_sha256": hashlib.sha256(TEST_MANIFEST.read_bytes()).hexdigest(),
        "csv_sha256": {
            method: hashlib.sha256(path.read_bytes()).hexdigest()
            for method, path in FILES.items()
        },
        "method_latencies_ms": {},
        "paired_method_ba_differences": {},
    }
    for method, rows in predictions.items():
        latencies = np.asarray([row["latency_ms"] for row in rows.values()])
        result["method_latencies_ms"][method] = dict(
            zip(("p50", "p95"), np.quantile(latencies, [0.5, 0.95]).tolist())
        )
    for condition in EXPECTED_COUNTS:
        names = [
            name for name, row in reference.items() if row["condition"] == condition
        ]
        result["paired_method_ba_differences"][condition] = {}
        for index, (first, second) in enumerate(PAIRS):
            result["paired_method_ba_differences"][condition][
                f"{first} minus {second}"
            ] = bootstrap_difference(
                predictions[first], predictions[second], names, 20260924 + index
            )
    output = WORK / "rr_full_baseline_comparison.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
