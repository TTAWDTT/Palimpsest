"""Measure whether simple paired features distinguish real RR processing from simulation."""

from __future__ import annotations

from experiments.paths import REPO_ROOT

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


BASE = REPO_ROOT
NUMERIC_FEATURES = (
    "coarse_luminance_correlation",
    "mean_rgb_absolute_change",
    "coarse_gradient_energy_ratio",
    "width_ratio",
    "height_ratio",
)


def fold(source):
    digest = hashlib.sha256(f"rr-simulator-critic-20260924/{source}".encode()).digest()
    return "train" if int.from_bytes(digest[:8], "big") % 2 == 0 else "test"


def evaluate(condition_rows, include_jpeg, numeric_features=NUMERIC_FEATURES):
    examples = {"train": [], "test": []}
    labels = {"train": [], "test": []}
    seen_sources = set()
    for row in condition_rows:
        source = row["source"]
        if source in seen_sources:
            raise ValueError(f"duplicate source: {source}")
        seen_sources.add(source)
        destination = fold(source)
        for mode, label in (("real", 1), ("simulated", 0)):
            features = {name: float(row[mode][name]) for name in numeric_features}
            if include_jpeg:
                features["jpeg_luma_fingerprint"] = row[f"{mode}_jpeg_luma_fingerprint"]
            examples[destination].append(features)
            labels[destination].append(label)
    vectorizer = DictVectorizer(sparse=False)
    train_features = vectorizer.fit_transform(examples["train"])
    test_features = vectorizer.transform(examples["test"])
    estimator = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        random_state=20260924,
        n_jobs=1,
    )
    estimator.fit(train_features, labels["train"])
    scores = estimator.predict_proba(test_features)[:, 1]
    predictions = scores > 0.5
    auc = float(roc_auc_score(labels["test"], scores))
    source_count = len(scores) // 2
    rng = np.random.default_rng(20260924)
    bootstrap_aucs = []
    test_labels = np.asarray(labels["test"])
    for _ in range(1000):
        source_indices = rng.integers(source_count, size=source_count)
        row_indices = (2 * source_indices[:, None] + np.arange(2)).ravel()
        bootstrap_aucs.append(
            roc_auc_score(test_labels[row_indices], scores[row_indices])
        )
    return {
        "train_sources": len(labels["train"]) // 2,
        "test_sources": len(labels["test"]) // 2,
        "features": len(vectorizer.feature_names_),
        "test_auc_real_vs_simulated": auc,
        "test_auc_source_bootstrap_95pct": [
            float(np.quantile(bootstrap_aucs, q)) for q in (0.025, 0.975)
        ],
        "test_auc_distance_from_chance": abs(auc - 0.5),
        "test_balanced_accuracy_at_half": float(
            balanced_accuracy_score(labels["test"], predictions)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=BASE / "work" / "rr_simulator_v0_evaluation.json"
    )
    parser.add_argument(
        "--output", type=Path, default=BASE / "work" / "rr_simulator_v0_critic.json"
    )
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data["per_source_features"]
    counts = Counter(row["condition"] for row in rows)
    expected = 2 * int(data["selected_validation_sources_per_class"])
    if counts != {"transfer": expected, "redigital": expected}:
        raise ValueError(f"unexpected row counts: {counts}")
    source_sets = {
        condition: {row["source"] for row in rows if row["condition"] == condition}
        for condition in counts
    }
    if source_sets["transfer"] != source_sets["redigital"]:
        raise ValueError("conditions do not have the same source set")
    result = {
        "scope": "limited-feature two-sample critic on source-disjoint split; chance AUC is 0.5",
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "conditions": {
            condition: {
                "numeric_only": evaluate(
                    [row for row in rows if row["condition"] == condition], False
                ),
                "numeric_plus_jpeg": evaluate(
                    [row for row in rows if row["condition"] == condition], True
                ),
            }
            for condition in ("transfer", "redigital")
        },
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["conditions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
