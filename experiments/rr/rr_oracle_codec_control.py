"""Test how much of RR processing one resize/JPEG pass can explain.

This diagnostic deliberately uses each processed image's true dimensions, JPEG
quantization tables, and subsampling. It is an oracle control, not a training
simulator or a fair held-out detection benchmark.
"""

from __future__ import annotations

from palimpsest.data.images import resize_unit_float as downsample

from palimpsest.paths import REPO_ROOT

import argparse
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, JpegImagePlugin

from experiments.rr.analyze_rr_simulation_inputs import (
    ROOT,
    load_pairs,
    image_metrics,
    quantiles,
)


BASE = REPO_ROOT


def select_sources(sources, count_per_class, max_pixels):
    chosen = []
    for label in ("ai", "real"):
        candidates = []
        for source, group in sources.items():
            if not source.startswith(label + "/") or "redigital" not in group:
                continue
            if any(
                int(row["width"]) * int(row["height"]) > max_pixels
                for row in group.values()
            ):
                continue
            rank = hashlib.sha256(
                f"rr-oracle-codec-control-20260924/{source}".encode()
            ).digest()
            candidates.append((rank, source))
        selected = [source for _, source in sorted(candidates)[:count_per_class]]
        if len(selected) != count_per_class:
            raise ValueError(f"not enough eligible sources for {label}")
        chosen.extend(selected)
    return chosen


def evaluate_pair(group, condition, side):
    original_row = group["original"]
    target_row = group[condition]
    with Image.open(ROOT / original_row["filename"]) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")
    with Image.open(ROOT / target_row["filename"]) as opened:
        qtables = [opened.quantization[index] for index in sorted(opened.quantization)]
        sampling = JpegImagePlugin.get_sampling(opened)
        actual = downsample(ImageOps.exif_transpose(opened).convert("RGB"), side)
    width, height = int(target_row["width"]), int(target_row["height"])
    resized = original.resize((width, height), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    resized.save(stream, format="JPEG", qtables=qtables, subsampling=sampling)
    stream.seek(0)
    with Image.open(stream) as encoded:
        if encoded.quantization != {
            index: table for index, table in enumerate(qtables)
        }:
            raise ValueError("JPEG quantization was not reproduced")
        simulated = downsample(encoded.convert("RGB"), side)
    reference = downsample(original, side)
    real_metrics = image_metrics(reference, actual)
    simulated_metrics = image_metrics(reference, simulated)
    real_simulated = image_metrics(actual, simulated)
    return {
        "actual_vs_simulated_rgb_mae": float(np.abs(actual - simulated).mean()),
        "actual_vs_simulated_luminance_correlation": real_simulated[
            "coarse_luminance_correlation"
        ],
        "absolute_difference_in_original_correlation": abs(
            real_metrics["coarse_luminance_correlation"]
            - simulated_metrics["coarse_luminance_correlation"]
        ),
        "absolute_difference_in_color_shift": abs(
            real_metrics["mean_rgb_absolute_change"]
            - simulated_metrics["mean_rgb_absolute_change"]
        ),
        "absolute_difference_in_gradient_ratio": abs(
            real_metrics["coarse_gradient_energy_ratio"]
            - simulated_metrics["coarse_gradient_energy_ratio"]
        ),
        "actual_original_correlation": real_metrics["coarse_luminance_correlation"],
        "simulated_original_correlation": simulated_metrics[
            "coarse_luminance_correlation"
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-per-class", type=int, default=60)
    parser.add_argument("--max-pixels", type=int, default=8_000_000)
    parser.add_argument("--normalized-side", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=BASE / "work" / "rr_oracle_codec_control.json"
    )
    args = parser.parse_args()
    sources = load_pairs()
    selected = select_sources(sources, args.sources_per_class, args.max_pixels)
    measures = defaultdict(lambda: defaultdict(list))
    class_measures = defaultdict(lambda: defaultdict(list))
    failures = []
    for source in selected:
        group = sources[source]
        for condition in ("transfer", "redigital"):
            try:
                metrics = evaluate_pair(group, condition, args.normalized_side)
                for name, value in metrics.items():
                    measures[condition][name].append(value)
                class_measures[condition][source.split("/", 1)[0]].append(
                    metrics["actual_vs_simulated_rgb_mae"]
                )
            except (OSError, ValueError) as error:
                failures.append(
                    {"source": source, "condition": condition, "error": str(error)}
                )
    result = {
        "scope": "oracle-size-and-JPEG-codec diagnostic only; no RR detector fitting",
        "selected_sources_per_class": args.sources_per_class,
        "max_pixels": args.max_pixels,
        "normalized_side": args.normalized_side,
        "selected_source_ids_sha256": hashlib.sha256(
            "\n".join(selected).encode()
        ).hexdigest(),
        "conditions": {
            condition: {
                name: quantiles(values) for name, values in values_by_name.items()
            }
            for condition, values_by_name in measures.items()
        },
        "match_counts_at_normalized_resolution": {
            condition: {
                "equal_rgb_arrays": sum(
                    value == 0
                    for value in values_by_name["actual_vs_simulated_rgb_mae"]
                ),
                "rgb_mae_above_0_05": sum(
                    value > 0.05
                    for value in values_by_name["actual_vs_simulated_rgb_mae"]
                ),
                "total": len(values_by_name["actual_vs_simulated_rgb_mae"]),
            }
            for condition, values_by_name in measures.items()
        },
        "class_stratified_rgb_mae": {
            condition: {
                label: {
                    **quantiles(values),
                    "equal_rgb_arrays": sum(value == 0 for value in values),
                    "rgb_mae_above_0_05": sum(value > 0.05 for value in values),
                }
                for label, values in values_by_label.items()
            }
            for condition, values_by_label in class_measures.items()
        },
        "failures": failures,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")
    print(
        {
            condition: next(iter(values.values()))["count"]
            for condition, values in result["conditions"].items()
        }
    )


if __name__ == "__main__":
    main()
