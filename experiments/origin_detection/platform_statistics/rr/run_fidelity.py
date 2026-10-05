"""Run the historical RR resize/JPEG fidelity pilot; not device validation."""

from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from palimpsest.data.rr import load_pairs

from experiments.origin_detection.platform_statistics.rr.protocol import (
    BASE,
    split_sources,
    fit_donors,
    evaluate,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donors-per-condition", type=int, default=1000)
    parser.add_argument("--validation-per-class", type=int, default=150)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--max-pixels", type=int, default=8_000_000)
    parser.add_argument(
        "--geometry", choices=("resize", "crop-aware"), default="resize"
    )
    parser.add_argument(
        "--jpeg-profile", choices=("coupled", "independent"), default="coupled"
    )
    parser.add_argument(
        "--geometry-donor-sampling",
        choices=("size-neighbor", "full-pool"),
        default="size-neighbor",
    )
    parser.add_argument("--balanced-donors", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=BASE / "work" / "rr_simulator_v0_evaluation.json"
    )
    args = parser.parse_args()
    sources = load_pairs()
    calibration, validation = split_sources(sources)
    donors = fit_donors(
        calibration, args.donors_per_condition, args.geometry, args.balanced_donors
    )
    if args.neighbors > min(len(pool) for pool in donors.values()):
        raise ValueError("neighbors exceeds donor pool")
    conditions, selected, per_source_features = evaluate(
        validation,
        donors,
        args.validation_per_class,
        args.max_pixels,
        args.neighbors,
        args.jpeg_profile,
        args.geometry_donor_sampling,
        args.balanced_donors,
    )
    result = {
        "scope": "source-disjoint RR internal pilot; test-derived calibration, not blind RR detector evidence",
        "calibration_sources": len(calibration),
        "validation_sources_available": len(validation),
        "donor_count_per_condition": {
            condition: len(pool) for condition, pool in donors.items()
        },
        "donor_source_ids_sha256": {
            condition: hashlib.sha256(
                "\n".join(donor["source"] for donor in pool).encode()
            ).hexdigest()
            for condition, pool in donors.items()
        },
        "donor_parameters_sha256": {
            condition: hashlib.sha256(
                json.dumps(pool, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for condition, pool in donors.items()
        },
        "validation_source_ids_sha256": hashlib.sha256(
            "\n".join(selected).encode()
        ).hexdigest(),
        "selected_validation_sources_per_class": args.validation_per_class,
        "max_pixels": args.max_pixels,
        "neighbors": args.neighbors,
        "geometry": args.geometry,
        "jpeg_profile": args.jpeg_profile,
        "geometry_donor_sampling": args.geometry_donor_sampling,
        "balanced_donors": args.balanced_donors,
        "donor_geometry_mode_counts": {
            condition: dict(Counter(donor["geometry_mode"] for donor in pool))
            for condition, pool in donors.items()
        },
        "conditions": conditions,
        "per_source_features": per_source_features,
    }
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output}")
    print({condition: values["images"] for condition, values in conditions.items()})


if __name__ == "__main__":
    main()
