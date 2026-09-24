"""Audit JPEG header profiles by source label in released RR redigital images."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, JpegImagePlugin


BASE = Path(__file__).resolve().parents[1]
ROOT = Path(r"E:\ai_image_origin_research\data\derived\rr_test")
MANIFEST = Path(r"E:\ai_image_origin_research\data\manifests\rr_test_files.csv")
OUTPUT = BASE / "work" / "rr_redigital_label_encoding.json"


def main() -> None:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] != "redigital":
                continue
            with Image.open(ROOT / row["filename"]) as image:
                sampling = JpegImagePlugin.get_sampling(image)
                profile = str(sampling)
            counts[row["label"]][profile] += 1
    result = {
        "condition": "redigital",
        "rows": sum(sum(values.values()) for values in counts.values()),
        "jpeg_subsampling_by_original_label": {
            label: dict(values) for label, values in counts.items()
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
