"""Summarize paired RR test provenance signals available from the verified manifest."""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT

from palimpsest.paths import REPO_ROOT

import csv
import json
from collections import Counter, defaultdict


BASE = REPO_ROOT
MANIFEST = DATA_ROOT / "manifests/rr_test_files.csv"
OUTPUT = BASE / "work" / "rr_manifest_provenance.json"


def main() -> None:
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_source = defaultdict(dict)
    condition_counts = Counter()
    dimensions = defaultdict(Counter)
    formats = defaultdict(Counter)
    for row in rows:
        source = (row["label"], row["source_id"])
        condition = row["condition"]
        if condition in by_source[source]:
            raise ValueError(f"duplicate condition for {source}: {condition}")
        by_source[source][condition] = row
        condition_counts[(condition, row["label"])] += 1
        dimensions[condition][(int(row["width"]), int(row["height"]))] += 1
        formats[condition][row["format"]] += 1
    result = {
        "manifest": str(MANIFEST),
        "sources": len(by_source),
        "condition_class_counts": {
            f"{condition}/{label}": count
            for (condition, label), count in sorted(condition_counts.items())
        },
        "conditions": {},
    }
    for condition in ("transfer", "redigital"):
        pairs = [
            (group["original"], group[condition])
            for group in by_source.values()
            if "original" in group and condition in group
        ]
        identical = [
            (original["label"], original["source_id"])
            for original, processed in pairs
            if original["sha256"] == processed["sha256"]
        ]
        result["conditions"][condition] = {
            "pairs": len(pairs),
            "same_dimensions_as_original": sum(
                original["width"] == processed["width"]
                and original["height"] == processed["height"]
                for original, processed in pairs
            ),
            "output_700x700": dimensions[condition][(700, 700)],
            "exact_byte_identical_to_original": len(identical),
            "exact_byte_identical_sources": [
                f"{label}/{source}" for label, source in identical
            ],
            "formats": dict(formats[condition]),
            "top_dimensions": [
                {"width": width, "height": height, "count": count}
                for (width, height), count in dimensions[condition].most_common(12)
            ],
        }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
