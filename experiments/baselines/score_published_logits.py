"""Recompute single-image baselines from QuAD's published detector logits."""

from __future__ import annotations

from palimpsest.evaluation.classification import evaluate
from palimpsest.io.tables import read_rows

import argparse
import json
from pathlib import Path


DETECTORS = ("DMID", "CoDE", "D3", "B-Free", "DRCT", "CO-SPY")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rewind_rows = read_rows(args.manifest_dir / "rewind_official.csv")
    tree_rows = read_rows(args.manifest_dir / "ancestree_test_official.csv")
    source_rows = read_rows(args.manifest_dir / "ancestree_sources_official.csv")
    test_sources = {row["src"] for row in tree_rows}
    clean_test_rows = [row for row in source_rows if row["src"] in test_sources]
    if len(clean_test_rows) != len(test_sources):
        raise ValueError("AncesTree source table does not cover every test source")

    conditions = {
        "ReWIND_all": rewind_rows,
        "ReWIND_without_RRDataset": [
            row for row in rewind_rows if row["src_dataset"] != "RRDataset"
        ],
        "AncesTree_clean_test_sources": clean_test_rows,
    }
    for level in range(1, 6):
        conditions[f"AncesTree_level_{level}"] = [
            row for row in tree_rows if int(row["level"]) == level
        ]

    results = {
        condition: {detector: evaluate(rows, detector) for detector in DETECTORS}
        for condition, rows in conditions.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(conditions)} conditions to {args.output}")


if __name__ == "__main__":
    main()
