"""Test a class-blind photometric simulator on RR's observable 4:2:0 stratum."""

from __future__ import annotations

from palimpsest.paths import REPO_ROOT

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, JpegImagePlugin

from experiments.rr.analyze_rr_simulation_inputs import ROOT, image_metrics, load_pairs
from experiments.rr.rr_simulator_v0 import (
    normalized,
    source_rank,
    split_sources,
    wasserstein_1d,
)


BASE = REPO_ROOT
DEVELOPMENT = BASE / "work" / "rr_simulator_crop_independent_1000_evaluation.json"
OUTPUT = BASE / "work" / "rr_redigital_420_color_pilot.json"
MAX_PIXELS = 8_000_000
DONORS = 500
NEIGHBORS = 30


def is_420_same_size(group) -> bool:
    original, processed = group["original"], group["redigital"]
    if (int(original["width"]), int(original["height"])) != (
        int(processed["width"]),
        int(processed["height"]),
    ):
        return False
    if original["sha256"] == processed["sha256"]:
        return False
    with Image.open(ROOT / processed["filename"]) as image:
        return JpegImagePlugin.get_sampling(image) == 2


def image_array(row) -> np.ndarray:
    with Image.open(ROOT / row["filename"]) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    return normalized(image, 128)


def fit_photometric_donor(source, group) -> dict[str, object]:
    original = image_array(group["original"]).reshape(-1, 3)
    processed = image_array(group["redigital"]).reshape(-1, 3)
    original_mean = original.mean(axis=0)
    processed_mean = processed.mean(axis=0)
    variance = ((original - original_mean) ** 2).mean(axis=0)
    covariance = ((original - original_mean) * (processed - processed_mean)).mean(
        axis=0
    )
    gain = np.clip(covariance / np.maximum(variance, 1e-4), 0.4, 1.6)
    bias = np.clip(processed_mean - gain * original_mean, -0.4, 0.4)
    return {
        "source": source,
        "width": int(group["original"]["width"]),
        "height": int(group["original"]["height"]),
        "gain": gain.tolist(),
        "bias": bias.tolist(),
    }


def choose_donor(source, original_row, donors, selection):
    if selection == "balanced-full-pool":
        source_id = source.split("/", 1)[1]
        seed = int.from_bytes(
            source_rank(source_id, "redigital-420-balanced-draw")[:8], "big"
        )
        return donors[int(np.random.default_rng(seed).integers(len(donors)))]
    width, height = int(original_row["width"]), int(original_row["height"])
    ranked = sorted(
        donors,
        key=lambda donor: (
            abs(math.log(width / donor["width"]))
            + abs(math.log(height / donor["height"]))
        ),
    )[:NEIGHBORS]
    seed = int.from_bytes(source_rank(source, "redigital-420-color-draw")[:8], "big")
    return ranked[int(np.random.default_rng(seed).integers(len(ranked)))]


def encode(image: Image.Image) -> np.ndarray:
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=95, subsampling=2)
    stream.seek(0)
    with Image.open(stream) as encoded:
        return normalized(encoded.convert("RGB"), 128)


def apply_color(original: Image.Image, donor) -> Image.Image:
    gain, bias = donor["gain"], donor["bias"]
    lookup = [
        int(np.clip(round((gain[channel] * value / 255 + bias[channel]) * 255), 0, 255))
        for channel in range(3)
        for value in range(256)
    ]
    return original.point(lookup)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--donor-selection",
        choices=("size-neighbor", "balanced-full-pool"),
        default="size-neighbor",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    development = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    selected = list(
        dict.fromkeys(row["source"] for row in development["per_source_features"])
    )
    if (
        hashlib.sha256("\n".join(selected).encode()).hexdigest()
        != development["validation_source_ids_sha256"]
    ):
        raise ValueError("development source fingerprint mismatch")
    sources = load_pairs()
    calibration, _ = split_sources(sources)
    eligible_donors = [
        source
        for source, group in calibration.items()
        if "redigital" in group
        and is_420_same_size(group)
        and int(group["original"]["width"]) * int(group["original"]["height"])
        <= MAX_PIXELS
    ]
    if args.donor_selection == "balanced-full-pool":
        donor_sources = []
        for label in ("ai", "real"):
            label_pool = [
                source for source in eligible_donors if source.startswith(label + "/")
            ]
            donor_sources.extend(
                sorted(
                    label_pool,
                    key=lambda source: source_rank(
                        source, "redigital-420-balanced-donors"
                    ),
                )[: DONORS // 2]
            )
    else:
        donor_sources = sorted(
            eligible_donors,
            key=lambda source: source_rank(source, "redigital-420-donors"),
        )[:DONORS]
    if len(donor_sources) != DONORS:
        raise ValueError("not enough calibration donors")
    donors = [
        fit_photometric_donor(source, calibration[source]) for source in donor_sources
    ]
    if {donor["source"] for donor in donors} & set(selected):
        raise ValueError("calibration/development overlap")
    rows = []
    for source in selected:
        group = sources[source]
        if not is_420_same_size(group):
            continue
        donor = choose_donor(source, group["original"], donors, args.donor_selection)
        with Image.open(ROOT / group["original"]["filename"]) as opened:
            original = ImageOps.exif_transpose(opened).convert("RGB")
        reference = normalized(original, 128)
        actual = image_array(group["redigital"])
        baseline = encode(original)
        colored = encode(apply_color(original, donor))
        row = {"source": source, "donor": donor["source"]}
        for name, image in (
            ("real", actual),
            ("baseline", baseline),
            ("color", colored),
        ):
            features = image_metrics(reference, image)
            features["width_ratio"] = 1.0
            features["height_ratio"] = 1.0
            row[name] = features
        rows.append(row)
    metrics = {}
    for variant in ("baseline", "color"):
        metrics[variant] = {
            feature: wasserstein_1d(
                [row["real"][feature] for row in rows],
                [row[variant][feature] for row in rows],
            )
            for feature in rows[0]["real"]
        }
    result = {
        "scope": "RR test-derived source-disjoint 4:2:0 same-size development pilot; uses observed stratum, not true method labels",
        "donor_selection": args.donor_selection,
        "development_sha256": hashlib.sha256(DEVELOPMENT.read_bytes()).hexdigest(),
        "donor_sources_sha256": hashlib.sha256(
            "\n".join(donor_sources).encode()
        ).hexdigest(),
        "donors": DONORS,
        "sources": len(rows),
        "metrics": metrics,
        "per_source_features": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "sources": len(rows),
                "donor_selection": args.donor_selection,
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
