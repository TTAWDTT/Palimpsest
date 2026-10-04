"""Apply the train/val format-only PNG→AI, JPEG→real rule to RR test."""

import csv
import json
from collections import Counter
from pathlib import Path


MANIFEST = Path("E:/ai_image_origin_research/data/manifests/rr_test_files.csv")
OUTPUT = Path("work/rr_test_format_control.json")


def main():
    with MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter((row["condition"], row["label"]) for row in rows)
    correct = Counter(
        (row["condition"], row["label"])
        for row in rows
        if ("ai" if row["format"] == "PNG" else "real") == row["label"]
    )
    result = {}
    for condition in ("original", "transfer", "redigital"):
        real_accuracy = correct[(condition, "real")] / counts[(condition, "real")]
        ai_accuracy = correct[(condition, "ai")] / counts[(condition, "ai")]
        result[condition] = {
            "real_images": counts[(condition, "real")],
            "ai_images": counts[(condition, "ai")],
            "real_accuracy": real_accuracy,
            "ai_accuracy": ai_accuracy,
            "balanced_accuracy": (real_accuracy + ai_accuracy) / 2,
        }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
