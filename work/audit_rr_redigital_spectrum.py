"""Measure paired low/mid/high frequency changes in RR redigital strata."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, JpegImagePlugin

from analyze_rr_simulation_inputs import ROOT, load_pairs


BASE = Path(__file__).resolve().parents[1]
DEVELOPMENT = BASE / "work" / "rr_simulator_crop_independent_1000_evaluation.json"
OUTPUT = BASE / "work" / "rr_redigital_spectrum_development.json"
SIDE = 256
COORDINATES = np.fft.fftshift(np.fft.fftfreq(SIDE))
FREQUENCY_Y, FREQUENCY_X = np.meshgrid(COORDINATES, COORDINATES, indexing="ij")
RADIUS = np.hypot(FREQUENCY_X, FREQUENCY_Y)
WINDOW = np.outer(np.hanning(SIDE), np.hanning(SIDE))
BANDS = {
    "low": (RADIUS >= 0.02) & (RADIUS < 0.08),
    "mid": (RADIUS >= 0.08) & (RADIUS < 0.20),
    "high": (RADIUS >= 0.20) & (RADIUS < 0.45),
}


def read_power(row):
    with Image.open(ROOT / row["filename"]) as opened:
        sampling = JpegImagePlugin.get_sampling(opened) if row["condition"] == "redigital" else None
        image = ImageOps.exif_transpose(opened).convert("RGB")
    gray = np.asarray(image.resize((SIDE, SIDE), Image.Resampling.BICUBIC),
                      dtype=np.float32).mean(axis=2) / 255
    spectrum = np.fft.fftshift(np.fft.fft2((gray - gray.mean()) * WINDOW))
    return np.abs(spectrum) ** 2, sampling


def paired_features(original, processed):
    result = {}
    for band_name, mask in BANDS.items():
        original_energy = float(original[mask].mean())
        processed_energy = float(processed[mask].mean())
        result[f"{band_name}_log_power_ratio"] = math.log(
            (processed_energy + 1e-8) / (original_energy + 1e-8)
        )
    mid = BANDS["mid"]
    axis = mid & ((np.abs(FREQUENCY_X) < 0.025) |
                  (np.abs(FREQUENCY_Y) < 0.025))
    for name, spectrum in (("original", original), ("processed", processed)):
        result[f"{name}_axis_power_fraction"] = float(spectrum[axis].sum() /
                                                     max(spectrum[mid].sum(), 1e-8))
        mid_power = spectrum[mid]
        result[f"{name}_mid_peak_ratio"] = float(
            np.quantile(mid_power, 0.99) / max(np.median(mid_power), 1e-8)
        )
    result["axis_power_fraction_change"] = (result["processed_axis_power_fraction"] -
                                             result["original_axis_power_fraction"])
    result["mid_peak_ratio_log_change"] = math.log(
        max(result["processed_mid_peak_ratio"], 1e-8) /
        max(result["original_mid_peak_ratio"], 1e-8)
    )
    return result


def quantiles(values):
    return {f"p{int(probability*100):02d}": float(np.quantile(values, probability))
            for probability in (0.1, 0.5, 0.9)}


def main() -> None:
    evaluation = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
    selected = list(dict.fromkeys(row["source"] for row in evaluation["per_source_features"]))
    if len(selected) != 1000:
        raise ValueError("expected 1000 development sources")
    sources = load_pairs()
    grouped = defaultdict(list)
    for source in selected:
        group = sources[source]
        original, _ = read_power(group["original"])
        processed, sampling = read_power(group["redigital"])
        grouped[str(sampling)].append({"source": source,
                                       "features": paired_features(original, processed)})
    result = {sampling: {
        "sources": len(rows),
        "feature_quantiles": {feature: quantiles([row["features"][feature] for row in rows])
                              for feature in rows[0]["features"]},
        "per_source": rows,
    } for sampling, rows in grouped.items()}
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({sampling: {"sources": group["sources"],
                                 "selected_medians": {feature: values["p50"]
                                                      for feature, values in group["feature_quantiles"].items()
                                                      if feature.endswith("log_power_ratio") or
                                                      feature.endswith("change")}}
                      for sampling, group in result.items()}, indent=2))


if __name__ == "__main__":
    main()
