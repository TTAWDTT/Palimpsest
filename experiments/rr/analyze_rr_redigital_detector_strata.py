"""Describe fixed-detector degradation in RR JPEG subsampling strata."""

from __future__ import annotations

from experiments.paths import REPO_ROOT

import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, JpegImagePlugin


BASE = REPO_ROOT
ROOT = Path(r"E:\ai_image_origin_research\data\derived\rr_test")
MANIFEST = Path(r"E:\ai_image_origin_research\data\manifests\rr_test_files.csv")
TRAINVAL = Path(r"E:\ai_image_origin_research\data\manifests\rr_trainval_files.csv")
METHODS = {
    "B-Free": BASE / "work" / "rr_bfree_complete.csv",
    "D3": BASE / "work" / "d3_rr_full.csv",
}
OUTPUT = BASE / "work" / "rr_redigital_detector_strata.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(group: list[tuple[str, float, float]]) -> dict[str, object]:
    accuracy = {}
    for label in ("ai", "real"):
        rows = [
            (original, transformed)
            for row_label, original, transformed in group
            if row_label == label
        ]
        expected_positive = label == "ai"
        original_correct = sum(
            (original > 0) == expected_positive for original, _ in rows
        )
        transformed_correct = sum(
            (transformed > 0) == expected_positive for _, transformed in rows
        )
        accuracy[label] = {
            "count": len(rows),
            "original": original_correct / len(rows),
            "redigital": transformed_correct / len(rows),
        }
    return {
        "images": len(group),
        "class_accuracy": accuracy,
        "original_balanced_accuracy": (
            accuracy["ai"]["original"] + accuracy["real"]["original"]
        )
        / 2,
        "redigital_balanced_accuracy": (
            accuracy["ai"]["redigital"] + accuracy["real"]["redigital"]
        )
        / 2,
    }


def main() -> None:
    manifest = read_csv(MANIFEST)
    trainval_hashes = {row["sha256"] for row in read_csv(TRAINVAL)}
    excluded = {
        f"{row['label']}/{row['source_id']}"
        for row in manifest
        if row["condition"] == "original" and row["sha256"] in trainval_hashes
    }
    if len(excluded) != 14:
        raise ValueError("expected 14 exact train/val overlaps")
    strata = {}
    for row in manifest:
        if row["condition"] != "redigital":
            continue
        source = f"{row['label']}/{row['source_id']}"
        if source in excluded:
            continue
        with Image.open(ROOT / row["filename"]) as image:
            strata[source] = str(JpegImagePlugin.get_sampling(image))
    result = {
        "scope": "redigital JPEG header strata, post-14 exclusion; observational, no method labels",
        "sources": len(strata),
        "methods": {},
    }
    for name, path in METHODS.items():
        scores = {}
        for row in read_csv(path):
            condition = row["filename"].split("/", 1)[0]
            if condition not in ("original", "redigital"):
                continue
            source = row["src"]
            if source in strata:
                scores[source, condition] = float(row["score"])
        grouped = defaultdict(list)
        for source, sampling in strata.items():
            label = source.split("/", 1)[0]
            grouped[sampling].append(
                (label, scores[source, "original"], scores[source, "redigital"])
            )
        result["methods"][name] = {
            sampling: summarize(group) for sampling, group in grouped.items()
        }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
