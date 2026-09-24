"""Measure how much ReWIND labels leak through format and image dimensions."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from score_published_logits import LABELS, evaluate


DATA_PATH = Path(r"E:\ai_image_origin_research\data\manifests\rewind_official.csv")
OUTPUT_PATH = Path("work/metadata_shortcut_diagnostic.json")


def format_key(row: dict[str, str]) -> str:
    return row["format"].upper()


def size_key(row: dict[str, str]) -> int:
    shortest_side = min(int(row["w"]), int(row["h"]))
    return min(max(int(math.log2(shortest_side)), 8), 15)


def feature_key(row: dict[str, str], kind: str):
    if kind == "format":
        return format_key(row)
    if kind == "size":
        return size_key(row)
    return (format_key(row), size_key(row))


def main() -> None:
    with DATA_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    source_labels = {}
    for row in rows:
        source = row["src"]
        label = LABELS[row["label"]]
        if source in source_labels and source_labels[source] != label:
            raise ValueError(f"Conflicting origin labels for source {source}")
        source_labels[source] = label

    source_fold = {}
    for label in (0, 1):
        sources = [source for source, value in source_labels.items() if value == label]
        sources.sort(key=lambda source: hashlib.sha256(source.encode()).digest())
        for index, source in enumerate(sources):
            source_fold[source] = index % 5

    results = {}
    for kind, category_count in (("format", 3), ("size", 8), ("format_size", 24)):
        for weighting in ("image", "source"):
            predictions = []
            for fold in range(5):
                training = [row for row in rows if source_fold[row["src"]] != fold]
                testing = [row for row in rows if source_fold[row["src"]] == fold]
                source_counts = Counter(row["src"] for row in training)
                counts = [defaultdict(float), defaultdict(float)]
                totals = [0.0, 0.0]
                for row in training:
                    label = LABELS[row["label"]]
                    weight = 1.0 if weighting == "image" else 1.0 / source_counts[row["src"]]
                    counts[label][feature_key(row, kind)] += weight
                    totals[label] += weight
                smoothing = 1.0 if weighting == "image" else 1.0 / 10
                for row in testing:
                    key = feature_key(row, kind)
                    real_likelihood = (counts[0][key] + smoothing) / (
                        totals[0] + smoothing * category_count
                    )
                    fake_likelihood = (counts[1][key] + smoothing) / (
                        totals[1] + smoothing * category_count
                    )
                    predictions.append(
                        {**row, "shortcut_score": str(math.log(fake_likelihood / real_likelihood))}
                    )
            results[f"{kind}_{weighting}"] = evaluate(predictions, "shortcut_score")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for kind, metrics in results.items():
        print(
            f"{kind}: AUC={metrics['auc']:.3f} "
            f"BA={metrics['balanced_accuracy_at_zero']:.3f} "
            f"sourceBA={metrics['source_macro_balanced_accuracy_at_zero']:.3f}"
        )


if __name__ == "__main__":
    main()
