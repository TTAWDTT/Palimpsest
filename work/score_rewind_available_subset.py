"""Score the QuAD-published logits on files expected in its public ReWIND ZIP."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from score_published_logits import DETECTORS, evaluate


MANIFEST_DIR = Path(r"E:\ai_image_origin_research\data\manifests")
OUTPUT_PATH = Path("work/rewind_available_subset_scores.json")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    official = read_csv(MANIFEST_DIR / "rewind_official.csv")
    expected = read_csv(MANIFEST_DIR / "rewind_archive_expected.csv")
    by_filename = {row["filename"]: row for row in official}
    if len(by_filename) != len(official):
        raise ValueError("Duplicate filenames in official CSV")

    subset = []
    for row in expected:
        matched = by_filename[row["filename"]]
        for key in ("src", "label", "md5", "src_dataset"):
            if row[key] != matched[key]:
                raise ValueError(f"Metadata mismatch for {row['filename']}: {key}")
        if abs(float(row["B-Free"]) - float(matched["B-Free"])) > 1e-6:
            raise ValueError(f"Published B-Free score mismatch: {row['filename']}")
        subset.append(matched)

    by_dataset = {}
    for dataset in sorted({row["src_dataset"] for row in subset}):
        group = [row for row in subset if row["src_dataset"] == dataset]
        by_dataset[dataset] = {
            "images": len(group),
            "sources": len({row["src"] for row in group}),
            "real_images": sum(row["label"] == "REAL" for row in group),
            "fake_images": sum(row["label"] == "FAKE" for row in group),
            "accuracy_at_zero": sum(
                (float(row["B-Free"]) > 0) == (row["label"] == "FAKE")
                for row in group
            ) / len(group),
        }

    result = {
        "expected_files": len(subset),
        "source_datasets": dict(Counter(row["src_dataset"] for row in subset)),
        "published_scores": {detector: evaluate(subset, detector) for detector in DETECTORS},
        "bfree_by_source_dataset": by_dataset,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "files={expected_files} sources={sources} B-Free AUC={auc:.3f} sourceBA={ba:.3f}".format(
            expected_files=len(subset),
            sources=result["published_scores"]["B-Free"]["sources"],
            auc=result["published_scores"]["B-Free"]["auc"],
            ba=result["published_scores"]["B-Free"]["source_macro_balanced_accuracy_at_zero"],
        )
    )


if __name__ == "__main__":
    main()
