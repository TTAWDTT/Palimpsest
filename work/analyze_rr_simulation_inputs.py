"""Characterize paired RR transformations without assigning unknown process labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(r"E:\ai_image_origin_research\data\derived\rr_test")
MANIFEST = Path(r"E:\ai_image_origin_research\data\manifests\rr_test_files.csv")
BASE = Path(__file__).resolve().parents[1]
OVERLAP_AUDIT = BASE / "work" / "rr_bfree_evaluation.json"
CONDITIONS = ("transfer", "redigital")


def quantiles(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        **({key: float(np.quantile(array, probability))
            for key, probability in (("p05", 0.05), ("p25", 0.25), ("p50", 0.5),
                                     ("p75", 0.75), ("p95", 0.95))} if len(array) else {}),
    }


def load_pairs() -> dict[str, dict[str, dict[str, str]]]:
    excluded = set(json.loads(OVERLAP_AUDIT.read_text(encoding="utf-8"))
                   ["trainval_overlap_audit"]["excluded_source_ids"])
    sources: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            source = f"{row['label']}/{row['source_id']}"
            if source in excluded:
                continue
            condition = row["condition"]
            if condition in sources[source]:
                raise ValueError(f"duplicate {source} {condition}")
            sources[source][condition] = row
    if len(sources) != 16986:
        raise ValueError(f"unexpected source count: {len(sources)}")
    if sum("redigital" in rows for rows in sources.values()) != 16985:
        raise ValueError("unexpected redigital pair count")
    return sources


def describe_manifest(sources: dict[str, dict[str, dict[str, str]]]) -> dict:
    results = {}
    for condition in CONDITIONS:
        ratios: dict[str, list[float]] = defaultdict(list)
        source_count = 0
        equal_dimensions = 0
        output_formats = Counter()
        size_pairs = Counter()
        by_label = Counter()
        for source, group in sources.items():
            if condition not in group:
                continue
            original = group["original"]
            transformed = group[condition]
            ow, oh = (int(original[key]) for key in ("width", "height"))
            tw, th = (int(transformed[key]) for key in ("width", "height"))
            source_count += 1
            by_label[source.split("/", 1)[0]] += 1
            equal_dimensions += (ow, oh) == (tw, th)
            output_formats[transformed["format"]] += 1
            size_pairs[(tw, th)] += 1
            ratios["width_ratio"].append(tw / ow)
            ratios["height_ratio"].append(th / oh)
            ratios["pixel_area_ratio"].append(tw * th / (ow * oh))
            ratios["byte_ratio"].append(int(transformed["bytes"]) / int(original["bytes"]))
            ratios["output_width"].append(tw)
            ratios["output_height"].append(th)
        results[condition] = {
            "paired_sources": source_count,
            "class_counts": dict(by_label),
            "equal_dimensions_fraction": equal_dimensions / source_count,
            "output_formats": dict(output_formats),
            "common_output_sizes": [{"width": w, "height": h, "count": count}
                                    for (w, h), count in size_pairs.most_common(12)],
            "distributions": {name: quantiles(values) for name, values in ratios.items()},
        }
    return results


def load_small_image(row: dict[str, str], side: int) -> tuple[np.ndarray, str | None]:
    with Image.open(ROOT / row["filename"]) as opened:
        jpeg_table = None
        if opened.format == "JPEG" and opened.quantization:
            table = opened.quantization.get(0)
            if table:
                jpeg_table = hashlib.sha256(np.asarray(table, dtype=np.uint16).tobytes()).hexdigest()[:16]
        image = ImageOps.exif_transpose(opened).convert("RGB")
        array = np.asarray(image.resize((side, side), Image.Resampling.BICUBIC), dtype=np.float32) / 255
    return array, jpeg_table


def image_metrics(original: np.ndarray, processed: np.ndarray) -> dict[str, float]:
    reference = original.mean(axis=2)
    observed = processed.mean(axis=2)
    reference_centered = reference - reference.mean()
    observed_centered = observed - observed.mean()
    denominator = float(np.linalg.norm(reference_centered) * np.linalg.norm(observed_centered))
    correlation = float(np.sum(reference_centered * observed_centered) / denominator) if denominator else 0.0
    correlation = min(1.0, max(-1.0, correlation))

    def gradient_energy(gray: np.ndarray) -> float:
        return float((np.abs(np.diff(gray, axis=0)).mean() +
                      np.abs(np.diff(gray, axis=1)).mean()) / 2)

    original_gradient = gradient_energy(reference)
    return {
        "coarse_luminance_correlation": correlation,
        "mean_rgb_absolute_change": float(np.mean(np.abs(original.mean(axis=(0, 1)) -
                                                   processed.mean(axis=(0, 1))))),
        "coarse_gradient_energy_ratio": gradient_energy(observed) / max(original_gradient, 1e-8),
    }


def describe_sample(sources: dict[str, dict[str, dict[str, str]]], sample_per_class: int,
                    max_pixels: int, side: int) -> dict:
    selected = {}
    skipped_large = Counter()
    for label in ("ai", "real"):
        candidates = []
        for source, group in sources.items():
            if not source.startswith(label + "/"):
                continue
            if any(int(row["width"]) * int(row["height"]) > max_pixels
                   for row in group.values()):
                skipped_large[label] += 1
                continue
            rank = hashlib.sha256(f"rr-simulation-diagnostic-20260924/{source}".encode()).digest()
            candidates.append((rank, source))
        selected[label] = [source for _, source in sorted(candidates)[:sample_per_class]]
    if any(len(names) != sample_per_class for names in selected.values()):
        raise ValueError("not enough eligible sources")

    metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    jpeg_tables: dict[str, Counter] = defaultdict(Counter)
    errors = []
    for label, names in selected.items():
        for source in names:
            group = sources[source]
            try:
                original, _ = load_small_image(group["original"], side)
                for condition in CONDITIONS:
                    if condition not in group:
                        continue
                    processed, table = load_small_image(group[condition], side)
                    for name, value in image_metrics(original, processed).items():
                        metrics[condition][name].append(value)
                    if table:
                        jpeg_tables[condition][table] += 1
            except (OSError, ValueError) as error:
                errors.append({"source": source, "error": str(error)})
    return {
        "sampling": {
            "per_class": sample_per_class,
            "max_pixels_per_image": max_pixels,
            "normalized_side": side,
            "eligible_sources_excluded_for_large_image": dict(skipped_large),
            "selected_source_ids_sha256": hashlib.sha256(
                "\n".join(selected["ai"] + selected["real"]).encode()
            ).hexdigest(),
        },
        "pixel_metrics": {condition: {name: quantiles(values) for name, values in measures.items()}
                          for condition, measures in metrics.items()},
        "jpeg_quantization_fingerprints": {
            condition: [{"fingerprint": name, "count": count}
                        for name, count in counter.most_common(12)]
            for condition, counter in jpeg_tables.items()
        },
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-per-class", type=int, default=400)
    parser.add_argument("--max-pixels", type=int, default=8_000_000)
    parser.add_argument("--normalized-side", type=int, default=128)
    parser.add_argument("--output", type=Path, default=BASE / "work" / "rr_simulation_diagnostic.json")
    args = parser.parse_args()
    sources = load_pairs()
    result = {
        "scope": "exploratory paired transformation diagnosis, not simulator validation",
        "exact_overlap_sources_excluded": 14,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "manifest_descriptors": describe_manifest(sources),
        "sample_descriptors": describe_sample(sources, args.sample_per_class,
                                               args.max_pixels, args.normalized_side),
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(json.dumps({condition: data["paired_sources"]
                      for condition, data in result["manifest_descriptors"].items()}))


if __name__ == "__main__":
    main()
