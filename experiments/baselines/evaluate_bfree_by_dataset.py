"""Summarize local and published B-Free decisions by ReWIND upstream dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from experiments.baselines.score_published_logits import evaluate


def summarize(rows: list[dict[str, str]], score_key: str) -> dict[str, object]:
    label_counts = defaultdict(int)
    correct_counts = defaultdict(int)
    source_decisions = defaultdict(list)
    source_labels = {}
    for row in rows:
        label = row["label"]
        source = row["src"]
        if source in source_labels and source_labels[source] != label:
            raise ValueError(f"Conflicting labels for source {source}")
        source_labels[source] = label
        correct = int((float(row[score_key]) > 0) == (label == "FAKE"))
        label_counts[label] += 1
        correct_counts[label] += correct
        source_decisions[source].append(correct)

    result: dict[str, object] = {
        "images": len(rows),
        "sources": len(source_decisions),
        "class_accuracy": {
            label: correct_counts[label] / count
            for label, count in label_counts.items()
        },
        "class_source_macro_accuracy": {
            label: sum(
                sum(source_decisions[source]) / len(source_decisions[source])
                for source, source_label in source_labels.items()
                if source_label == label
            )
            / sum(source_label == label for source_label in source_labels.values())
            for label in label_counts
        },
    }
    if len(label_counts) == 2:
        score_rows = [
            {"src": row["src"], "label": row["label"], "B-Free": row[score_key]}
            for row in rows
        ]
        result["two_class_metrics"] = evaluate(score_rows, "B-Free")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    with args.inference_csv.open(newline="", encoding="utf-8-sig") as handle:
        all_rows = list(csv.DictReader(handle))
    if len(all_rows) != 5582 or any(
        not row["score"] or row["error"] for row in all_rows
    ):
        raise ValueError("Expected all 5,582 successful ReWIND ZIP inferences")
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest = {row["filename"]: row for row in manifest_rows}
    if (
        len(manifest_rows) != 5582
        or len(manifest) != len(manifest_rows)
        or len({row["filename"] for row in all_rows}) != len(all_rows)
        or {row["filename"] for row in all_rows} != manifest.keys()
    ):
        raise ValueError(
            "Inference filenames must exactly match the verified ZIP manifest"
        )
    for row in all_rows:
        expected = manifest[row["filename"]]
        if (row["src"], row["label"], row["published_bfree_score"]) != (
            expected["src"],
            expected["label"],
            expected["B-Free"],
        ):
            raise ValueError(f"Manifest metadata mismatch: {row['filename']}")

    by_dataset = defaultdict(list)
    for row in all_rows:
        by_dataset[manifest[row["filename"]]["src_dataset"]].append(row)

    result = {
        dataset: {
            "local": summarize(rows, "score"),
            "published_on_same_files": summarize(rows, "published_bfree_score"),
        }
        for dataset, rows in sorted(by_dataset.items())
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
