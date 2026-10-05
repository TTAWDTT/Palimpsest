"""Stratify frozen origin detectors by observed RR transfer geometry."""

from __future__ import annotations

from palimpsest.paths import REPO_ROOT

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from palimpsest.data.rr import load_pairs
from experiments.origin_detection.platform_statistics.rr.protocol import infer_geometry


BASE = REPO_ROOT
EVALUATION = BASE / "work" / "rr_simulator_crop_1000_evaluation.json"
OUTPUT = BASE / "work" / "rr_transfer_geometry_effect.json"


def scores(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mapping = {row["filename"]: float(row["score"]) for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"duplicate filenames: {path}")
    return mapping


def balanced_accuracy(rows: list[dict], condition: str) -> float:
    rates = []
    for label in ("ai", "real"):
        subset = [row for row in rows if row["label"] == label]
        if not subset:
            raise ValueError(f"no {label} images")
        rates.append(sum(row[f"{condition}_correct"] for row in subset) / len(subset))
    return float(np.mean(rates))


def summarize(rows: list[dict]) -> dict:
    original = balanced_accuracy(rows, "original")
    transfer = balanced_accuracy(rows, "transfer")
    return {
        "sources": len(rows),
        "ai_sources": sum(row["label"] == "ai" for row in rows),
        "real_sources": sum(row["label"] == "real" for row in rows),
        "original_balanced_accuracy": original,
        "transfer_balanced_accuracy": transfer,
        "paired_balanced_accuracy_change": transfer - original,
    }


def main() -> None:
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    selected = {row["source"] for row in evaluation["per_source_features"]}
    expected = 2 * int(evaluation["selected_validation_sources_per_class"])
    if len(selected) != expected:
        raise ValueError(f"expected {expected} sources, got {len(selected)}")
    pairs = load_pairs()
    methods = {
        "B-Free": scores(BASE / "work" / "rr_bfree_complete.csv"),
        "D3": scores(BASE / "work" / "d3_rr_full.csv"),
    }
    geometry_by_source = {
        source: infer_geometry(pairs[source]["original"], pairs[source]["transfer"])
        for source in sorted(selected)
    }
    result = {
        "scope": "RR internal analysis; test images used only for post-hoc geometry stratification",
        "selected_source_sha256": hashlib.sha256(
            "\n".join(sorted(selected)).encode()
        ).hexdigest(),
        "geometry_counts": {
            mode: sum(value == mode for value in geometry_by_source.values())
            for mode in ("center_crop", "resize")
        },
        "methods": {},
    }
    for method, score_mapping in methods.items():
        groups = defaultdict(list)
        for source, mode in geometry_by_source.items():
            label = source.split("/", 1)[0]
            original_name = pairs[source]["original"]["filename"]
            transfer_name = pairs[source]["transfer"]["filename"]
            original_score = score_mapping[original_name]
            transfer_score = score_mapping[transfer_name]
            groups[mode].append(
                {
                    "label": label,
                    "original_correct": (original_score > 0) == (label == "ai"),
                    "transfer_correct": (transfer_score > 0) == (label == "ai"),
                }
            )
        result["methods"][method] = {
            mode: summarize(rows) for mode, rows in groups.items()
        }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
