"""Verify rendered JPEGs reproduce the simulator's recorded paired features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from palimpsest.data.rr import ROOT, load_pairs
from palimpsest.evaluation.image_pairs import image_metrics
from experiments.origin_detection.platform_statistics.rr.protocol import normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    index = json.loads(args.index.read_text(encoding="utf-8"))
    if (
        index["evaluation_sha256"]
        != hashlib.sha256(args.evaluation.read_bytes()).hexdigest()
    ):
        raise ValueError("index/evaluation fingerprint mismatch")
    expected = {
        row["source"]: row
        for row in evaluation["per_source_features"]
        if row["condition"] == "transfer"
    }
    if len(expected) != len(index["rows"]) or len(expected) != 1000:
        raise ValueError("source count mismatch")
    sources = load_pairs()
    max_difference = 0.0
    for row in index["rows"]:
        source = row["source"]
        original_row = sources[source]["original"]
        with Image.open(ROOT / original_row["filename"]) as opened:
            original = ImageOps.exif_transpose(opened).convert("RGB")
        with Image.open(args.output_root / row["filename"]) as opened:
            simulated = opened.convert("RGB")
        features = image_metrics(normalized(original, 128), normalized(simulated, 128))
        features["width_ratio"] = simulated.width / original.width
        features["height_ratio"] = simulated.height / original.height
        for feature, value in features.items():
            difference = abs(value - expected[source]["simulated"][feature])
            max_difference = max(max_difference, difference)
            if difference > 1e-6:
                raise ValueError(
                    f"materialized feature differs: {source}/{feature}={difference}"
                )
        if (simulated.width, simulated.height) != (row["width"], row["height"]):
            raise ValueError(f"materialized dimensions differ: {source}")
        with Image.open(args.output_root / row["filename"]) as opened:
            qtable = opened.quantization[0]
        qtable_fingerprint = hashlib.sha256(
            np.asarray(qtable, dtype=np.uint16).tobytes()
        ).hexdigest()[:16]
        if qtable_fingerprint != row["jpeg_luma_fingerprint"]:
            raise ValueError(f"materialized JPEG table differs: {source}")
    result = {
        "verified_sources": len(expected),
        "max_feature_absolute_difference": max_difference,
        "evaluation_sha256": index["evaluation_sha256"],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
