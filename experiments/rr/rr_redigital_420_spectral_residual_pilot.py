"""Pilot a class-balanced phase-randomized residual for RR 4:2:0 images.

All simulated outputs here are 128-pixel proxies. This tests whether a
calibration-derived artifact spectrum helps; it is not a full-resolution
camera/display simulator or a detector-training result.
"""

from __future__ import annotations

from experiments.paths import REPO_ROOT

import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.rr.analyze_rr_simulation_inputs import image_metrics, load_pairs
from experiments.rr.rr_redigital_420_color_pilot import (
    DONORS,
    MAX_PIXELS,
    image_array,
    is_420_same_size,
)
from experiments.rr.rr_simulator_v0 import source_rank, split_sources, wasserstein_1d


BASE = REPO_ROOT
DEVELOPMENT = BASE / "work" / "rr_simulator_crop_independent_1000_evaluation.json"
OUTPUT = BASE / "work" / "rr_redigital_420_spectral_residual_pilot.json"
SIDE = 128
GRID = np.fft.fftshift(np.fft.fftfreq(SIDE))
FREQUENCY_Y, FREQUENCY_X = np.meshgrid(GRID, GRID, indexing="ij")
RADIUS = np.hypot(FREQUENCY_X, FREQUENCY_Y)
WINDOW = np.outer(np.hanning(SIDE), np.hanning(SIDE))
FREQUENCY_BANDS = {
    "mid_spectrum_log_power_ratio": (RADIUS >= 0.08) & (RADIUS < 0.20),
    "high_spectrum_log_power_ratio": (RADIUS >= 0.20) & (RADIUS < 0.45),
}


def fit_donor(source, group):
    original = image_array(group["original"])
    processed = image_array(group["redigital"])
    first = original.reshape(-1, 3)
    second = processed.reshape(-1, 3)
    original_mean = first.mean(axis=0)
    processed_mean = second.mean(axis=0)
    gain = np.clip(
        ((first - original_mean) * (second - processed_mean)).mean(axis=0)
        / np.maximum(((first - original_mean) ** 2).mean(axis=0), 1e-4),
        0.4,
        1.6,
    )
    bias = np.clip(processed_mean - gain * original_mean, -0.4, 0.4)
    color_prediction = np.clip(original * gain + bias, 0, 1)
    residual = processed - color_prediction
    residual -= residual.mean(axis=(0, 1), keepdims=True)
    magnitude = np.abs(np.fft.rfft2(residual, axes=(0, 1))).astype(np.float32)
    magnitude[0, 0] = 0
    return {
        "source": source,
        "gain": gain,
        "bias": bias,
        "magnitude": magnitude,
        "residual_std": float(residual.std()),
        "residual_channel_std": residual.std(axis=(0, 1)),
    }


def choose_donor(source, donors):
    source_id = source.split("/", 1)[1]
    seed = int.from_bytes(
        source_rank(source_id, "redigital-420-balanced-draw")[:8], "big"
    )
    return donors[int(np.random.default_rng(seed).integers(len(donors)))]


def draw_residual(source, donor):
    source_id = source.split("/", 1)[1]
    seed = int.from_bytes(
        source_rank(source_id, "redigital-420-spectrum-draw")[:8], "big"
    )
    rng = np.random.default_rng(seed)
    white_spectrum = np.fft.rfft2(rng.normal(size=(SIDE, SIDE, 3)), axes=(0, 1))
    random_phase = white_spectrum / np.maximum(np.abs(white_spectrum), 1e-12)
    synthesized = np.fft.irfft2(
        donor["magnitude"] * random_phase, s=(SIDE, SIDE), axes=(0, 1)
    ).real
    return synthesized.astype(np.float32)


def draw_white_noise(source, donor):
    source_id = source.split("/", 1)[1]
    seed = int.from_bytes(source_rank(source_id, "redigital-420-white-draw")[:8], "big")
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(SIDE, SIDE, 3)) * donor["residual_channel_std"]).astype(
        np.float32
    )


def encode_small(array):
    pixels = quantize_small(array)
    image = Image.fromarray(pixels, "RGB")
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=95, subsampling=2)
    stream.seek(0)
    with Image.open(stream) as encoded:
        return np.asarray(encoded.convert("RGB"), dtype=np.float32) / 255


def quantize_small(array):
    return np.clip(np.round(array * 255), 0, 255).astype(np.uint8)


def spectrum_features(reference, observed):
    def power(array):
        gray = array.mean(axis=2)
        centered = (gray - gray.mean()) * WINDOW
        return np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2

    first, second = power(reference), power(observed)
    return {
        name: math.log(
            (float(second[mask].mean()) + 1e-8) / (float(first[mask].mean()) + 1e-8)
        )
        for name, mask in FREQUENCY_BANDS.items()
    }


def main():
    evaluation = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    selected = list(
        dict.fromkeys(row["source"] for row in evaluation["per_source_features"])
    )
    selected_hash = hashlib.sha256("\n".join(selected).encode()).hexdigest()
    if (
        len(selected) != 1000
        or selected_hash != evaluation["validation_source_ids_sha256"]
    ):
        raise ValueError("development source set differs")
    sources = load_pairs()
    calibration, _ = split_sources(sources)
    donor_sources = []
    for label in ("ai", "real"):
        candidates = [
            source
            for source, group in calibration.items()
            if source.startswith(label + "/")
            and "redigital" in group
            and is_420_same_size(group)
            and int(group["original"]["width"]) * int(group["original"]["height"])
            <= MAX_PIXELS
        ]
        chosen = sorted(
            candidates,
            key=lambda source: source_rank(source, "redigital-420-balanced-donors"),
        )[: DONORS // 2]
        if len(chosen) != DONORS // 2:
            raise ValueError(f"insufficient {label} donors")
        donor_sources.extend(chosen)
    donors = [fit_donor(source, calibration[source]) for source in donor_sources]
    if {donor["source"] for donor in donors} & set(selected):
        raise ValueError("calibration/development overlap")
    rows = []
    for source in selected:
        group = sources[source]
        if not is_420_same_size(group):
            continue
        original = image_array(group["original"])
        actual = image_array(group["redigital"])
        donor = choose_donor(source, donors)
        color = np.clip(original * donor["gain"] + donor["bias"], 0, 1)
        white = np.clip(color + draw_white_noise(source, donor), 0, 1)
        spectral = np.clip(color + draw_residual(source, donor), 0, 1)
        variants = {
            "real": actual,
            "color": encode_small(color),
            "white_noise": encode_small(white),
            "spectral_residual": encode_small(spectral),
            "color_no_proxy_jpeg": quantize_small(color).astype(np.float32) / 255,
            "white_noise_no_proxy_jpeg": quantize_small(white).astype(np.float32) / 255,
            "spectral_residual_no_proxy_jpeg": (
                quantize_small(spectral).astype(np.float32) / 255
            ),
        }
        features = {}
        for name, image in variants.items():
            values = image_metrics(original, image) | spectrum_features(original, image)
            values["width_ratio"] = 1.0
            values["height_ratio"] = 1.0
            features[name] = values
        rows.append({"source": source, "donor": donor["source"], **features})
    distances = {
        variant: {
            feature: wasserstein_1d(
                [row["real"][feature] for row in rows],
                [row[variant][feature] for row in rows],
            )
            for feature in rows[0]["real"]
        }
        for variant in rows[0]
        if variant not in ("source", "donor", "real")
    }
    result = {
        "scope": "128-pixel phase-randomized spectral residual pilot on RR test-derived development stratum; not full-resolution physics",
        "selected_sources_sha256": selected_hash,
        "donor_sources_sha256": hashlib.sha256(
            "\n".join(donor_sources).encode()
        ).hexdigest(),
        "donors": len(donors),
        "sources": len(rows),
        "donor_residual_std_p50": float(
            np.median([donor["residual_std"] for donor in donors])
        ),
        "wasserstein": distances,
        "per_source_features": rows,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "sources": len(rows),
                "donor_residual_std_p50": result["donor_residual_std_p50"],
                "wasserstein": distances,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
