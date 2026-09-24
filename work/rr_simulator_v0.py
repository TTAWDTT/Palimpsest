"""Fit a class-blind resize/JPEG simulator and audit its held-out RR fidelity.

This is a development pilot using a source-disjoint split of RR test images.
It must not be presented as an independent detector test or a physical
re-digitization simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, JpegImagePlugin

from analyze_rr_simulation_inputs import ROOT, image_metrics, load_pairs, quantiles


BASE = Path(__file__).resolve().parents[1]
CONDITIONS = ("transfer", "redigital")
SEED = 20260924


def source_rank(source: str, namespace: str) -> bytes:
    return hashlib.sha256(f"rr-simulator-v0/{namespace}/{source}".encode()).digest()


def split_sources(sources):
    calibration, validation = {}, {}
    for source, rows in sources.items():
        bucket = int.from_bytes(source_rank(source, "split")[:8], "big") % 5
        (calibration if bucket == 0 else validation)[source] = rows
    if calibration.keys() & validation.keys():
        raise ValueError("calibration and validation source overlap")
    return calibration, validation


def eligible(group, max_pixels: int) -> bool:
    return all(int(row["width"]) * int(row["height"]) <= max_pixels
               for row in group.values())


def jpeg_settings(row):
    with Image.open(ROOT / row["filename"]) as image:
        if image.format != "JPEG" or not image.quantization:
            raise ValueError(f"missing JPEG quantization: {row['filename']}")
        qtables = [image.quantization[index] for index in sorted(image.quantization)]
        sampling = JpegImagePlugin.get_sampling(image)
    fingerprint = hashlib.sha256(np.asarray(qtables[0], dtype=np.uint16).tobytes()).hexdigest()[:16]
    return qtables, sampling, fingerprint


def infer_geometry(original_row, processed_row):
    width, height = int(processed_row["width"]), int(processed_row["height"])
    with Image.open(ROOT / original_row["filename"]) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")
    if width >= original.width or height >= original.height:
        return "resize"
    with Image.open(ROOT / processed_row["filename"]) as opened:
        processed = normalized(ImageOps.exif_transpose(opened).convert("RGB"), 128)
    full = normalized(original, 128)
    left, top = (original.width - width) // 2, (original.height - height) // 2
    cropped = normalized(original.crop((left, top, left + width, top + height)), 128)
    resize_corr = image_metrics(full, processed)["coarse_luminance_correlation"]
    crop_corr = image_metrics(cropped, processed)["coarse_luminance_correlation"]
    return "center_crop" if crop_corr > resize_corr + 0.05 else "resize"


def fit_donors(calibration, max_donors: int, geometry: str, balance_labels=False):
    donors = {}
    for condition in CONDITIONS:
        if balance_labels:
            if max_donors % 2:
                raise ValueError("balanced donor count must be even")
            chosen = []
            for label in ("ai", "real"):
                candidates = [(source_rank(source, f"balanced-donor/{condition}"), source)
                              for source, group in calibration.items()
                              if source.startswith(label + "/") and condition in group]
                if len(candidates) < max_donors // 2:
                    raise ValueError(f"insufficient {label} donors for {condition}")
                chosen.extend(source for _, source in sorted(candidates)[:max_donors // 2])
        else:
            candidates = [(source_rank(source, f"donor/{condition}"), source)
                          for source, group in calibration.items() if condition in group]
            chosen = [source for _, source in sorted(candidates)[:max_donors]]
        donors[condition] = []
        for source in chosen:
            original = calibration[source]["original"]
            processed = calibration[source][condition]
            qtables, sampling, fingerprint = jpeg_settings(processed)
            geometry_mode = (infer_geometry(original, processed)
                             if geometry == "crop-aware" and condition == "transfer"
                             else "resize")
            donors[condition].append({
                "source": source,
                "input_width": int(original["width"]),
                "input_height": int(original["height"]),
                "width_ratio": int(processed["width"]) / int(original["width"]),
                "height_ratio": int(processed["height"]) / int(original["height"]),
                "qtables": qtables,
                "subsampling": sampling,
                "jpeg_luma_fingerprint": fingerprint,
                "geometry_mode": geometry_mode,
            })
    return donors


def choose_donor(source, original_row, condition, donors, neighbors,
                 sampling="size-neighbor"):
    if sampling == "full-pool":
        source_id = source.split("/", 1)[1]
        random_seed = int.from_bytes(source_rank(source_id, f"geometry-full-pool/{condition}/{SEED}")[:8], "big")
        pool = donors[condition]
        return pool[int(np.random.default_rng(random_seed).integers(len(pool)))]
    width, height = int(original_row["width"]), int(original_row["height"])
    ranked = sorted(
        donors[condition],
        key=lambda donor: abs(math.log(width / donor["input_width"])) +
                          abs(math.log(height / donor["input_height"])),
    )[:neighbors]
    random_seed = int.from_bytes(source_rank(source, f"draw/{condition}/{SEED}")[:8], "big")
    rng = np.random.default_rng(random_seed)
    return ranked[int(rng.integers(len(ranked)))]


def choose_encoding_donor(source, condition, donors, source_id_seed=False):
    """Draw a class-blind JPEG profile independently of image geometry."""
    seed_source = source.split("/", 1)[1] if source_id_seed else source
    random_seed = int.from_bytes(source_rank(seed_source, f"codec/{condition}/{SEED}")[:8], "big")
    rng = np.random.default_rng(random_seed)
    pool = donors[condition]
    return pool[int(rng.integers(len(pool)))]


def normalized(image, side):
    return np.asarray(image.resize((side, side), Image.Resampling.BICUBIC),
                      dtype=np.float32) / 255


def render_simulated_jpeg(original, donor, max_pixels, encoding_donor=None):
    encoding_donor = donor if encoding_donor is None else encoding_donor
    width = max(1, round(original.width * donor["width_ratio"]))
    height = max(1, round(original.height * donor["height_ratio"]))
    if width * height > max_pixels:
        scale = math.sqrt(max_pixels / (width * height))
        width, height = max(1, int(width * scale)), max(1, int(height * scale))
    if donor["geometry_mode"] == "center_crop" and width <= original.width and height <= original.height:
        left, top = (original.width - width) // 2, (original.height - height) // 2
        transformed = original.crop((left, top, left + width, top + height))
    else:
        transformed = original.resize((width, height), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    transformed.save(stream, format="JPEG", qtables=encoding_donor["qtables"],
                 subsampling=encoding_donor["subsampling"])
    stream.seek(0)
    with Image.open(stream) as encoded:
        if encoded.quantization != {index: table for index, table in enumerate(encoding_donor["qtables"])}:
            raise ValueError("simulated JPEG quantization differs from selected donor")
    return stream.getvalue(), width, height, encoding_donor["jpeg_luma_fingerprint"]


def simulate(original, donor, max_pixels, encoding_donor=None):
    payload, width, height, fingerprint = render_simulated_jpeg(
        original, donor, max_pixels, encoding_donor
    )
    with Image.open(io.BytesIO(payload)) as encoded:
        array = normalized(encoded.convert("RGB"), 128)
    return array, width, height, fingerprint


def wasserstein_1d(first, second):
    first_values, second_values = np.sort(first), np.sort(second)
    if len(first_values) != len(second_values):
        raise ValueError("sample lengths differ")
    return float(np.abs(first_values - second_values).mean())


def total_variation(first, second):
    size_first, size_second = sum(first.values()), sum(second.values())
    keys = first.keys() | second.keys()
    return 0.5 * sum(abs(first[key] / size_first - second[key] / size_second)
                     for key in keys)


def evaluate(validation, donors, selected_per_class, max_pixels, neighbors,
             jpeg_profile="coupled", geometry_donor_sampling="size-neighbor",
             source_id_codec_seed=False):
    selected = []
    for label in ("ai", "real"):
        eligible_sources = [source for source, group in validation.items()
                            if source.startswith(label + "/") and
                            all(condition in group for condition in CONDITIONS) and
                            eligible(group, max_pixels)]
        picked = sorted(eligible_sources, key=lambda source: source_rank(source, "validation"))
        if len(picked) < selected_per_class:
            raise ValueError(f"insufficient validation sources: {label}")
        selected.extend(picked[:selected_per_class])

    measurements = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    jpeg_counts = defaultdict(lambda: defaultdict(Counter))
    per_source_features = []
    for source in selected:
        group = validation[source]
        with Image.open(ROOT / group["original"]["filename"]) as opened:
            original = ImageOps.exif_transpose(opened).convert("RGB")
        reference = normalized(original, 128)
        for condition in CONDITIONS:
            real_row = group[condition]
            donor = choose_donor(source, group["original"], condition, donors, neighbors,
                                 geometry_donor_sampling)
            encoding_donor = (choose_encoding_donor(source, condition, donors,
                                                    source_id_codec_seed)
                              if jpeg_profile == "independent" else donor)
            simulated, width, height, simulated_fingerprint = simulate(
                original, donor, max_pixels, encoding_donor
            )
            with Image.open(ROOT / real_row["filename"]) as opened:
                _, _, real_fingerprint = jpeg_settings(real_row)
                actual = normalized(ImageOps.exif_transpose(opened).convert("RGB"), 128)
            jpeg_counts[condition]["real"][real_fingerprint] += 1
            jpeg_counts[condition]["simulated"][simulated_fingerprint] += 1
            paired_features = {"source": source, "condition": condition,
                               "real_jpeg_luma_fingerprint": real_fingerprint,
                               "simulated_jpeg_luma_fingerprint": simulated_fingerprint}
            for mode, transformed, output_width, output_height in (
                ("real", actual, int(real_row["width"]), int(real_row["height"])),
                ("simulated", simulated, width, height),
            ):
                values = image_metrics(reference, transformed)
                values["width_ratio"] = output_width / original.width
                values["height_ratio"] = output_height / original.height
                paired_features[mode] = values
                for name, value in values.items():
                    measurements[condition][mode][name].append(value)
            per_source_features.append(paired_features)

    result = {}
    for condition in CONDITIONS:
        real = measurements[condition]["real"]
        simulated = measurements[condition]["simulated"]
        result[condition] = {
            "images": len(real["width_ratio"]),
            "real": {name: quantiles(values) for name, values in real.items()},
            "simulated": {name: quantiles(values) for name, values in simulated.items()},
            "feature_distribution_wasserstein": {
                name: wasserstein_1d(real[name], simulated[name]) for name in real
            },
            "jpeg_luma_fingerprint_total_variation": total_variation(
                jpeg_counts[condition]["real"], jpeg_counts[condition]["simulated"]
            ),
            "real_jpeg_luma_unique": len(jpeg_counts[condition]["real"]),
            "simulated_jpeg_luma_unique": len(jpeg_counts[condition]["simulated"]),
        }
    return result, selected, per_source_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donors-per-condition", type=int, default=1000)
    parser.add_argument("--validation-per-class", type=int, default=150)
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--max-pixels", type=int, default=8_000_000)
    parser.add_argument("--geometry", choices=("resize", "crop-aware"), default="resize")
    parser.add_argument("--jpeg-profile", choices=("coupled", "independent"),
                        default="coupled")
    parser.add_argument("--geometry-donor-sampling", choices=("size-neighbor", "full-pool"),
                        default="size-neighbor")
    parser.add_argument("--balanced-donors", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=BASE / "work" / "rr_simulator_v0_evaluation.json")
    args = parser.parse_args()
    sources = load_pairs()
    calibration, validation = split_sources(sources)
    donors = fit_donors(calibration, args.donors_per_condition, args.geometry,
                        args.balanced_donors)
    if args.neighbors > min(len(pool) for pool in donors.values()):
        raise ValueError("neighbors exceeds donor pool")
    conditions, selected, per_source_features = evaluate(
        validation, donors, args.validation_per_class, args.max_pixels, args.neighbors,
        args.jpeg_profile, args.geometry_donor_sampling, args.balanced_donors
    )
    result = {
        "scope": "source-disjoint RR internal pilot; test-derived calibration, not blind RR detector evidence",
        "calibration_sources": len(calibration),
        "validation_sources_available": len(validation),
        "donor_count_per_condition": {condition: len(pool) for condition, pool in donors.items()},
        "donor_source_ids_sha256": {condition: hashlib.sha256(
            "\n".join(donor["source"] for donor in pool).encode()).hexdigest()
            for condition, pool in donors.items()},
        "donor_parameters_sha256": {condition: hashlib.sha256(
            json.dumps(pool, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest() for condition, pool in donors.items()},
        "validation_source_ids_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(),
        "selected_validation_sources_per_class": args.validation_per_class,
        "max_pixels": args.max_pixels,
        "neighbors": args.neighbors,
        "geometry": args.geometry,
        "jpeg_profile": args.jpeg_profile,
        "geometry_donor_sampling": args.geometry_donor_sampling,
        "balanced_donors": args.balanced_donors,
        "donor_geometry_mode_counts": {condition: dict(Counter(
            donor["geometry_mode"] for donor in pool)) for condition, pool in donors.items()},
        "conditions": conditions,
        "per_source_features": per_source_features,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print({condition: values["images"] for condition, values in conditions.items()})


if __name__ == "__main__":
    main()
