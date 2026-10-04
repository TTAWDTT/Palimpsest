"""Compare locally inferred B-Free logits with QuAD's published per-file scores."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from experiments.baselines.score_published_logits import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    with args.inference_csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    completed = [row for row in rows if row["score"] and not row["error"]]
    if not completed:
        raise ValueError("No successful local predictions")
    if len({row["filename"] for row in rows}) != len(rows):
        raise ValueError("Duplicate filename in inference output")
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest = {row["filename"]: row for row in manifest_rows}
    if (
        len(manifest_rows) != 5582
        or len(manifest) != len(manifest_rows)
        or not {row["filename"] for row in rows} <= manifest.keys()
    ):
        raise ValueError("Inference rows do not match the 5,582-image ZIP manifest")
    for row in rows:
        expected = manifest[row["filename"]]
        if (row["src"], row["label"], row["published_bfree_score"]) != (
            expected["src"],
            expected["label"],
            expected["B-Free"],
        ):
            raise ValueError(f"Manifest metadata mismatch: {row['filename']}")

    errors = [
        abs(float(row["score"]) - float(row["published_bfree_score"]))
        for row in completed
    ]
    local_rows = [
        {"src": row["src"], "label": row["label"], "B-Free": row["score"]}
        for row in completed
    ]
    published_rows = [
        {
            "src": row["src"],
            "label": row["label"],
            "B-Free": row["published_bfree_score"],
        }
        for row in completed
    ]
    sign_disagreements = sum(
        (float(row["score"]) > 0) != (float(row["published_bfree_score"]) > 0)
        for row in completed
    )
    result = {
        "input_images": len(manifest),
        "inference_rows": len(rows),
        "successful_images": len(completed),
        "coverage": len(completed) / len(manifest),
        "mean_absolute_logit_error": statistics.mean(errors),
        "max_absolute_logit_error": max(errors),
        "sign_disagreements": sign_disagreements,
        "local_metrics": evaluate(local_rows, "B-Free"),
        "published_metrics_on_same_files": evaluate(published_rows, "B-Free"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
