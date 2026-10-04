"""Characterize paired RR transformations without assigning unknown process labels."""

from __future__ import annotations
from palimpsest.data.rr import MANIFEST, load_pairs, load_small_image
from palimpsest.evaluation.image_pairs import image_metrics
from palimpsest.evaluation.distribution import quantiles


from palimpsest.paths import REPO_ROOT

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


BASE = REPO_ROOT
CONDITIONS = ("transfer", "redigital")


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
            ratios["byte_ratio"].append(
                int(transformed["bytes"]) / int(original["bytes"])
            )
            ratios["output_width"].append(tw)
            ratios["output_height"].append(th)
        results[condition] = {
            "paired_sources": source_count,
            "class_counts": dict(by_label),
            "equal_dimensions_fraction": equal_dimensions / source_count,
            "output_formats": dict(output_formats),
            "common_output_sizes": [
                {"width": w, "height": h, "count": count}
                for (w, h), count in size_pairs.most_common(12)
            ],
            "distributions": {
                name: quantiles(values) for name, values in ratios.items()
            },
        }
    return results


def describe_sample(
    sources: dict[str, dict[str, dict[str, str]]],
    sample_per_class: int,
    max_pixels: int,
    side: int,
) -> dict:
    selected = {}
    skipped_large = Counter()
    for label in ("ai", "real"):
        candidates = []
        for source, group in sources.items():
            if not source.startswith(label + "/"):
                continue
            if any(
                int(row["width"]) * int(row["height"]) > max_pixels
                for row in group.values()
            ):
                skipped_large[label] += 1
                continue
            rank = hashlib.sha256(
                f"rr-simulation-diagnostic-20260924/{source}".encode()
            ).digest()
            candidates.append((rank, source))
        selected[label] = [
            source for _, source in sorted(candidates)[:sample_per_class]
        ]
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
        "pixel_metrics": {
            condition: {name: quantiles(values) for name, values in measures.items()}
            for condition, measures in metrics.items()
        },
        "jpeg_quantization_fingerprints": {
            condition: [
                {"fingerprint": name, "count": count}
                for name, count in counter.most_common(12)
            ]
            for condition, counter in jpeg_tables.items()
        },
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-per-class", type=int, default=400)
    parser.add_argument("--max-pixels", type=int, default=8_000_000)
    parser.add_argument("--normalized-side", type=int, default=128)
    parser.add_argument(
        "--output", type=Path, default=BASE / "work" / "rr_simulation_diagnostic.json"
    )
    args = parser.parse_args()
    sources = load_pairs()
    result = {
        "scope": "exploratory paired transformation diagnosis, not simulator validation",
        "exact_overlap_sources_excluded": 14,
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "manifest_descriptors": describe_manifest(sources),
        "sample_descriptors": describe_sample(
            sources, args.sample_per_class, args.max_pixels, args.normalized_side
        ),
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")
    print(
        json.dumps(
            {
                condition: data["paired_sources"]
                for condition, data in result["manifest_descriptors"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
