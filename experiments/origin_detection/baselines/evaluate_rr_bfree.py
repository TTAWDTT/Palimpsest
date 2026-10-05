"""Evaluate B-Free on every verified RRDataset test condition and paired source."""

from __future__ import annotations

from palimpsest.evaluation.timing import summarize_timing
from palimpsest.evaluation.pairing import paired_change

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from palimpsest.evaluation.classification import evaluate


CONDITIONS = ("original", "transfer", "redigital")


def summarize_cohort(inference_rows: list[dict[str, str]]) -> dict[str, object]:
    by_condition = defaultdict(list)
    source_by_condition = defaultdict(dict)
    by_execution_mode = defaultdict(list)
    for row in inference_rows:
        condition = row["filename"].split("/", 1)[0]
        if condition not in CONDITIONS:
            raise ValueError(f"Unexpected condition: {condition}")
        by_condition[condition].append(row)
        by_execution_mode[row["preprocess_mode"]].append(row)
        if row["src"] in source_by_condition[condition]:
            raise ValueError(f"Duplicate source within condition: {row['src']}")
        source_by_condition[condition][row["src"]] = row
    if set(by_condition) != set(CONDITIONS):
        raise ValueError("All three RRDataset conditions are required")
    return {
        "timing_overall_ms": summarize_timing(inference_rows),
        "execution_modes": {
            mode: {"images": len(rows), "timing_ms": summarize_timing(rows)}
            for mode, rows in sorted(by_execution_mode.items())
        },
        "conditions": {
            condition: {
                "metrics": evaluate(
                    [
                        {
                            "src": row["src"],
                            "label": row["label"],
                            "B-Free": row["score"],
                        }
                        for row in by_condition[condition]
                    ],
                    "B-Free",
                ),
                "timing_ms": summarize_timing(by_condition[condition]),
            }
            for condition in CONDITIONS
        },
        "paired_changes_from_original": {
            condition: paired_change(
                source_by_condition["original"], source_by_condition[condition]
            )
            for condition in ("transfer", "redigital")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--test-image-manifest", type=Path, required=True)
    parser.add_argument("--trainval-image-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest = {row["filename"]: row for row in manifest_rows}
    with args.inference_csv.open(newline="", encoding="utf-8-sig") as handle:
        inference_rows = list(csv.DictReader(handle))
    if (
        not manifest
        or len(manifest) != len(manifest_rows)
        or len(inference_rows) != len(manifest)
        or {row["filename"] for row in inference_rows} != manifest.keys()
        or any(row["error"] or not row["score"] for row in inference_rows)
    ):
        raise ValueError(
            "Inference CSV must contain exactly one successful row per test image"
        )

    for row in inference_rows:
        expected = manifest[row["filename"]]
        if (row["src"], row["label"]) != (expected["src"], expected["label"]):
            raise ValueError(f"Inference metadata mismatch: {row['filename']}")
        score = float(row["score"])
        if not math.isfinite(score):
            raise ValueError(f"Nonfinite score: {row['filename']}")
    with args.trainval_image_manifest.open(newline="", encoding="utf-8-sig") as handle:
        trainval_hashes = {row["sha256"] for row in csv.DictReader(handle)}
    with args.test_image_manifest.open(newline="", encoding="utf-8-sig") as handle:
        image_rows = list(csv.DictReader(handle))
    if (
        len(image_rows) != len(manifest)
        or {row["filename"] for row in image_rows} != manifest.keys()
    ):
        raise ValueError("Test image manifest is not aligned with inference manifest")
    for image_row in image_rows:
        matching = manifest[image_row["filename"]]
        if image_row["label"] not in ("ai", "real"):
            raise ValueError(f"Unexpected source label: {image_row['filename']}")
        expected_label = "FAKE" if image_row["label"] == "ai" else "REAL"
        if (
            matching["src"] != f"{image_row['label']}/{image_row['source_id']}"
            or matching["label"] != expected_label
        ):
            raise ValueError(f"Test source mapping mismatch: {image_row['filename']}")
    overlapping_rows = [row for row in image_rows if row["sha256"] in trainval_hashes]
    excluded_sources = {
        f"{row['label']}/{row['source_id']}" for row in overlapping_rows
    }
    if len(overlapping_rows) != 14 or len(excluded_sources) != 14:
        raise ValueError("Expected 14 distinct overlapping RR test sources")
    result = {
        "exact_overlap_excluded": summarize_cohort(
            [row for row in inference_rows if row["src"] not in excluded_sources]
        ),
        "full_archive_descriptive": summarize_cohort(inference_rows),
        "trainval_overlap_audit": {
            "exact_overlap_test_images": len(overlapping_rows),
            "excluded_source_groups": len(excluded_sources),
            "excluded_source_ids": sorted(excluded_sources),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
