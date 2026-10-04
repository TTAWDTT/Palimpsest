"""Compare simulated and released transfer effects on fixed B-Free scores."""

from __future__ import annotations

from experiments.paths import REPO_ROOT

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


BASE = REPO_ROOT
INDEX = BASE / "work" / "rr_transfer_simulation_dev_index.json"
ACTUAL = BASE / "work" / "rr_bfree_complete.csv"
SIMULATED = BASE / "work" / "rr_transfer_simulation_dev_bfree.csv"
SIMULATION_ROOT = Path(r"E:\ai_image_origin_research\data\derived\rr_simulation_dev")
SIMULATION_MANIFEST = Path(
    r"E:\ai_image_origin_research\data\manifests\rr_transfer_simulation_dev_bfree.csv"
)
OUTPUT = BASE / "work" / "rr_transfer_simulation_dev_score_evaluation.json"
EXPECTED_ACTUAL_SHA256 = (
    "0e4b82ea9a3134ea464cc89575303a76de4d9420056f34bb2601731351570a9b"
)
EXPECTED_SIMULATION_MANIFEST_SHA256 = (
    "c158195de733a121df6c6b2c5ad1ab1af00a5446c7193fd218eaa5532019509c"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_scores(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if any(row["error"] or not math.isfinite(float(row["score"])) for row in rows):
        raise ValueError(f"invalid scores in {path}")
    return rows


def correctness(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return (scores > 0) == (labels == "FAKE")


def accuracy_by_class(correct: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    by_class = {
        label: float(correct[labels == label].mean()) for label in ("FAKE", "REAL")
    }
    by_class["balanced_accuracy"] = (by_class["FAKE"] + by_class["REAL"]) / 2
    return by_class


def bootstrap_mae_gain(
    actual_change: np.ndarray, simulated_change: np.ndarray, labels: np.ndarray
) -> list[float]:
    rng = np.random.default_rng(20260924)
    groups = [np.flatnonzero(labels == label) for label in ("FAKE", "REAL")]
    estimates = []
    for _ in range(2000):
        indices = np.concatenate(
            [rng.choice(group, len(group), replace=True) for group in groups]
        )
        estimates.append(
            float(
                np.abs(actual_change[indices]).mean()
                - np.abs(actual_change[indices] - simulated_change[indices]).mean()
            )
        )
    return [
        float(np.quantile(estimates, probability)) for probability in (0.025, 0.975)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--simulated", type=Path, default=SIMULATED)
    parser.add_argument("--simulation-root", type=Path, default=SIMULATION_ROOT)
    parser.add_argument("--simulation-manifest", type=Path, default=SIMULATION_MANIFEST)
    parser.add_argument(
        "--expected-manifest-sha256", default=EXPECTED_SIMULATION_MANIFEST_SHA256
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if file_sha256(ACTUAL) != EXPECTED_ACTUAL_SHA256:
        raise ValueError("official RR B-Free result fingerprint changed")
    if file_sha256(args.simulation_manifest) != args.expected_manifest_sha256:
        raise ValueError("simulated score manifest fingerprint changed")
    selected = json.loads(args.index.read_text(encoding="utf-8"))["rows"]
    expected_sources = [row["source"] for row in selected]
    if len(expected_sources) != 1000 or len(set(expected_sources)) != 1000:
        raise ValueError("expected 1000 unique development sources")
    actual_scores = {}
    for row in read_scores(ACTUAL):
        condition = row["filename"].split("/", 1)[0]
        key = row["src"], condition
        if key in actual_scores:
            raise ValueError(f"duplicate released score: {key}")
        actual_scores[key] = row
    simulated_rows = read_scores(args.simulated)
    simulated_scores = {row["src"]: row for row in simulated_rows}
    if len(simulated_rows) != 1000 or set(simulated_scores) != set(expected_sources):
        raise ValueError("simulated score coverage mismatch")
    for expected in selected:
        row = simulated_scores[expected["source"]]
        if row["filename"] != expected["filename"]:
            raise ValueError(f"simulated filename mismatch: {expected['source']}")
        if (
            file_sha256(args.simulation_root / expected["filename"])
            != expected["sha256"]
        ):
            raise ValueError(
                f"simulated image fingerprint mismatch: {expected['source']}"
            )
    labels = np.asarray(
        ["FAKE" if source.startswith("ai/") else "REAL" for source in expected_sources]
    )
    original = np.empty(1000, dtype=float)
    actual = np.empty(1000, dtype=float)
    simulated = np.empty(1000, dtype=float)
    for index, source in enumerate(expected_sources):
        for condition, output in (("original", original), ("transfer", actual)):
            row = actual_scores[source, condition]
            if row["label"] != labels[index]:
                raise ValueError(f"released label mismatch: {source}")
            output[index] = float(row["score"])
        if simulated_scores[source]["label"] != labels[index]:
            raise ValueError(f"simulated label mismatch: {source}")
        simulated[index] = float(simulated_scores[source]["score"])
    original_correct = correctness(original, labels)
    actual_correct = correctness(actual, labels)
    simulated_correct = correctness(simulated, labels)
    actual_change, simulated_change = actual - original, simulated - original
    actual_failures = original_correct & ~actual_correct
    predicted_failures = original_correct & ~simulated_correct
    true_positive = int((actual_failures & predicted_failures).sum())
    result = {
        "scope": "1000-source RR internal development check; fixed official B-Free weights; not independent evidence",
        "sources": len(expected_sources),
        "fake_sources": int((labels == "FAKE").sum()),
        "real_sources": int((labels == "REAL").sum()),
        "simulated_csv_sha256": file_sha256(args.simulated),
        "index_sha256": file_sha256(args.index),
        "accuracy": {
            "original": accuracy_by_class(original_correct, labels),
            "released_transfer": accuracy_by_class(actual_correct, labels),
            "simulated_transfer": accuracy_by_class(simulated_correct, labels),
        },
        "score_change": {
            "released_mean": float(actual_change.mean()),
            "simulated_mean": float(simulated_change.mean()),
            "released_mean_absolute": float(np.abs(actual_change).mean()),
            "simulated_mean_absolute": float(np.abs(simulated_change).mean()),
            "mean_absolute_error_simulated_vs_released": float(
                np.abs(actual_change - simulated_change).mean()
            ),
            "mean_absolute_error_no_change_vs_released": float(
                np.abs(actual_change).mean()
            ),
            "mae_gain_over_no_change": float(
                np.abs(actual_change).mean()
                - np.abs(actual_change - simulated_change).mean()
            ),
            "mae_gain_stratified_source_bootstrap_95pct": bootstrap_mae_gain(
                actual_change, simulated_change, labels
            ),
            "pearson_correlation": float(
                np.corrcoef(actual_change, simulated_change)[0, 1]
            ),
            "sign_agreement": float(
                (np.sign(actual_change) == np.sign(simulated_change)).mean()
            ),
        },
        "origin_correct_to_transfer_failure": {
            "released_failures": int(actual_failures.sum()),
            "simulated_failures": int(predicted_failures.sum()),
            "both": true_positive,
            "recall": true_positive / int(actual_failures.sum())
            if actual_failures.any()
            else None,
            "precision": true_positive / int(predicted_failures.sum())
            if predicted_failures.any()
            else None,
        },
        "processed_decision_agreement": float(
            (actual_correct == simulated_correct).mean()
        ),
        "simulated_latency_ms": {
            "p50": float(
                np.quantile(
                    [float(row["end_to_end_ms"]) for row in simulated_rows], 0.5
                )
            ),
            "p95": float(
                np.quantile(
                    [float(row["end_to_end_ms"]) for row in simulated_rows], 0.95
                )
            ),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
