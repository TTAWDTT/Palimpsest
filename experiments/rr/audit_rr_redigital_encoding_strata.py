"""Describe RR redigital geometry by observable JPEG subsampling stratum."""

from __future__ import annotations

from experiments.paths import REPO_ROOT

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, JpegImagePlugin


BASE = REPO_ROOT
ROOT = Path(r"E:\ai_image_origin_research\data\derived\rr_test")
MANIFEST = Path(r"E:\ai_image_origin_research\data\manifests\rr_test_files.csv")
OUTPUT = BASE / "work" / "rr_redigital_encoding_strata.json"


def main() -> None:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    originals = {
        (row["label"], row["source_id"]): row
        for row in rows
        if row["condition"] == "original"
    }
    by_sampling = defaultdict(list)
    for row in rows:
        if row["condition"] != "redigital":
            continue
        original = originals[row["label"], row["source_id"]]
        with Image.open(ROOT / row["filename"]) as image:
            sampling = str(JpegImagePlugin.get_sampling(image))
        by_sampling[sampling].append(
            {
                "label": row["label"],
                "same_dimensions": row["width"] == original["width"]
                and row["height"] == original["height"],
                "width_ratio": int(row["width"]) / int(original["width"]),
                "height_ratio": int(row["height"]) / int(original["height"]),
            }
        )
    result = {}
    for sampling, group in by_sampling.items():
        result[sampling] = {
            "count": len(group),
            "labels": dict(Counter(row["label"] for row in group)),
            "same_dimensions": sum(row["same_dimensions"] for row in group),
            "width_ratio_p10_p50_p90": [
                float(np.quantile([row["width_ratio"] for row in group], probability))
                for probability in (0.1, 0.5, 0.9)
            ],
            "height_ratio_p10_p50_p90": [
                float(np.quantile([row["height_ratio"] for row in group], probability))
                for probability in (0.1, 0.5, 0.9)
            ],
        }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
