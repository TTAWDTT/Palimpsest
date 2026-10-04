"""Quantify the RRDataset train/val label shortcut from decoded file format."""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT, WORK_DIR

import csv
import json
from collections import Counter, defaultdict


MANIFEST = DATA_ROOT / "manifests/rr_trainval_files.csv"
OUTPUT = WORK_DIR / "rr_format_shortcut.json"


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "val"]

    train_counts = defaultdict(Counter)
    for row in train:
        train_counts[row["format"]][row["label"]] += 1
    mapping = {
        image_format: counts.most_common(1)[0][0]
        for image_format, counts in train_counts.items()
    }

    correct_by_label = Counter()
    total_by_label = Counter()
    confusion = Counter()
    for row in validation:
        prediction = mapping[row["format"]]
        label = row["label"]
        correct_by_label[label] += int(prediction == label)
        total_by_label[label] += 1
        confusion[f"{label}->{prediction}"] += 1

    class_accuracy = {
        label: correct_by_label[label] / total_by_label[label]
        for label in ("real", "ai")
    }
    result = {
        "train_counts_by_format_and_label": {
            image_format: dict(counts) for image_format, counts in train_counts.items()
        },
        "format_to_label_mapping": mapping,
        "validation_confusion": dict(confusion),
        "validation_class_accuracy": class_accuracy,
        "validation_balanced_accuracy": sum(class_accuracy.values()) / 2,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
