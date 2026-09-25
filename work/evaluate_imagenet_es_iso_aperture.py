"""Check aperture/ISO interventions at the JPEG layer on frozen sources."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


MANIFEST = Path("work/imagenet_es_20class_aperture_iso_manifest.json")
OUTPUT = Path("work/imagenet_es_20class_iso_aperture_evaluation.json")
PARAMS = (4, 5, 13, 14, 22, 23)
WEIGHTS = np.array([.2126, .7152, .0722])


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= .0031308, 12.92 * rgb,
                    1.055 * np.power(rgb, 1 / 2.4) - .055)


def image_stats(path: str) -> tuple[np.ndarray, dict]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    stats = {
        "encoded_mean_luma": float((rgb * WEIGHTS).sum(2).mean()),
        "inverse_srgb_mean_luma": float((srgb_to_linear(rgb) * WEIGHTS).sum(2).mean()),
        "near_black_fraction": float((rgb.max(2) <= .02).mean()),
        "near_white_fraction": float((rgb.min(2) >= .98).mean()),
    }
    return rgb, stats


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if tuple(manifest["param_ids"]) != PARAMS or len(manifest["source_classes"]) != 20:
        raise ValueError("expected 20-source, six-condition fixed development manifest")
    groups = defaultdict(dict)
    for item in manifest["rows"]:
        parts = item["member"].split("/")
        if "param_control" not in parts:
            continue
        key = f"{parts[-2]}/{parts[-1]}"
        param_id = int(parts[4].split("_")[1])
        if param_id in groups[key]:
            raise ValueError(f"duplicate pair {key}, {param_id}")
        groups[key][param_id] = item
    if len(groups) != 20 or any(set(g) != set(PARAMS) for g in groups.values()):
        raise ValueError("incomplete source groups")
    rows = []
    for key, group in sorted(groups.items()):
        stats = {}
        images = {}
        for param_id in PARAMS:
            images[param_id], stats[param_id] = image_stats(group[param_id]["local_path"])
        if len({images[i].shape for i in PARAMS}) != 1:
            raise ValueError(f"unequal dimensions {key}")
        # f/16 at ISO 2000 has nominal gain*area = 8*(5/16)^2 ~= 0.781
        # compared with f/5 at ISO 250 at the same shutter and light state.
        exposure_factor = 8 * (5 / 16) ** 2
        predicted = linear_to_srgb(srgb_to_linear(images[4]) * exposure_factor)
        predicted_mean = float((predicted * WEIGHTS).sum(2).mean())
        observed_mean = stats[23]["encoded_mean_luma"]
        rows.append({
            "source_key": key,
            "encoded_mean_luma": {str(i): stats[i]["encoded_mean_luma"] for i in PARAMS},
            "inverse_srgb_mean_luma": {str(i): stats[i]["inverse_srgb_mean_luma"] for i in PARAMS},
            "near_black_fraction": {str(i): stats[i]["near_black_fraction"] for i in PARAMS},
            "near_white_fraction": {str(i): stats[i]["near_white_fraction"] for i in PARAMS},
            "iso_brightness_monotone": {str(a): stats[a+1]["encoded_mean_luma"] > stats[a]["encoded_mean_luma"]
                                        for a in (4, 13, 22)},
            "matched_nominal_exposure_f16_iso2000_over_f5_iso250_encoded_mean": observed_mean / stats[4]["encoded_mean_luma"],
            "nominal_exposure_prediction_over_observed_mean": predicted_mean / observed_mean,
            "nominal_exposure_prediction_rgb_mae": float(np.abs(predicted - images[23]).mean()),
        })
    report = {
        "scope": "20 fixed development sources; official l5 JPEG; 1/60s; f5/f9/f16 x ISO250/2000",
        "source_count": len(rows),
        "condition_count": len(rows) * len(PARAMS),
        "nominal_f16_iso2000_to_f5_iso250_gain_area_ratio": 8 * (5 / 16) ** 2,
        "iso_brightness_monotone_source_counts": {
            str(i): sum(r["iso_brightness_monotone"][str(i)] for r in rows)
            for i in (4, 13, 22)},
        "median_encoded_mean_luma_by_param": {
            str(i): float(np.median([r["encoded_mean_luma"][str(i)] for r in rows]))
            for i in PARAMS},
        "median_near_white_fraction_by_param": {
            str(i): float(np.median([r["near_white_fraction"][str(i)] for r in rows]))
            for i in PARAMS},
        "matched_nominal_exposure_encoded_mean_ratio_median": float(np.median(
            [r["matched_nominal_exposure_f16_iso2000_over_f5_iso250_encoded_mean"] for r in rows])),
        "matched_nominal_exposure_prediction_over_observed_mean_median": float(np.median(
            [r["nominal_exposure_prediction_over_observed_mean"] for r in rows])),
        "matched_nominal_exposure_prediction_rgb_mae_median": float(np.median(
            [r["nominal_exposure_prediction_rgb_mae"] for r in rows])),
        "rows": rows,
        "qualification": "ISO is camera signal gain, not added photons; published JPEG ISP and censoring unknown; nominal gain*area not a JPEG truth model",
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
