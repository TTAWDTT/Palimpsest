"""External JPEG-level aperture intervention check on fixed-source pairs."""

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


MANIFEST = Path("work/imagenet_es_20class_aperture_manifest.json")
OUT = Path("work/imagenet_es_20class_aperture_evaluation.json")
IDS = (4, 13, 22)  # Author test-grid CSV: ISO 250, 1/60 s, f/5/9/16.


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    return np.where(
        rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - 0.055
    )


def image_stats(path: str) -> tuple[dict, np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    y = (rgb * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).sum(2)
    linear = srgb_to_linear(rgb)
    linear_y = (linear * np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)).sum(2)
    stats = {
        "mean_encoded_luma": float(y.mean()),
        "mean_srgb_inverse_luma": float(linear_y.mean()),
        "near_white_fraction": float((rgb.min(2) >= 0.98).mean()),
        "near_black_fraction": float((rgb.max(2) <= 0.02).mean()),
    }
    return stats, linear_y, rgb


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_source = defaultdict(dict)
    for item in manifest["rows"]:
        name = item["member"]
        parts = name.split("/")
        param = 0 if "sampled_tin_no_resize2" in parts else int(parts[4].split("_")[1])
        by_source[parts[-1]][param] = item
    rows = []
    for source, group in sorted(by_source.items()):
        if any(key not in group for key in IDS):
            raise RuntimeError(f"incomplete aperture group {source}")
        if len({(group[key]["width"], group[key]["height"]) for key in IDS}) != 1:
            raise RuntimeError(f"capture dimensions differ within {source}")
        stats, planes, rgb_images = {}, {}, {}
        for key in IDS:
            stats[key], planes[key], rgb_images[key] = image_stats(
                group[key]["local_path"]
            )
        # Spatial low-pass reduces registration and JPEG block differences.
        smooth = {
            key: cv2.GaussianBlur(value, (0, 0), 5) for key, value in planes.items()
        }
        mask = (smooth[4] > 0.05) & (smooth[4] < 0.75) & (smooth[22] > 0.005)
        if mask.sum() < 2000:
            ratios = {"f9_over_f5": None, "f16_over_f5": None}
        else:
            ratios = {
                "f9_over_f5": float(np.median(smooth[13][mask] / smooth[4][mask])),
                "f16_over_f5": float(np.median(smooth[22][mask] / smooth[4][mask])),
            }
        baseline = {}
        for key, n in ((13, 9), (22, 16)):
            predicted = linear_to_srgb(srgb_to_linear(rgb_images[4]) * (5 / n) ** 2)
            predicted_mean = float(
                (predicted * np.array([0.2126, 0.7152, 0.0722])).sum(2).mean()
            )
            observed_mean = stats[key]["mean_encoded_luma"]
            baseline[str(key)] = {
                "predicted_mean_encoded_luma": predicted_mean,
                "predicted_over_observed_mean": predicted_mean / observed_mean,
                "rgb_mae": float(np.abs(predicted - rgb_images[key]).mean()),
            }
        rows.append(
            {
                "source": source,
                "class": group[4]["member"].split("/")[-2],
                "pixels": int(mask.size),
                "common_mask_pixels": int(mask.sum()),
                "mean_encoded_luma": [stats[i]["mean_encoded_luma"] for i in IDS],
                "mean_srgb_inverse_luma": [
                    stats[i]["mean_srgb_inverse_luma"] for i in IDS
                ],
                "near_white_fraction": [stats[i]["near_white_fraction"] for i in IDS],
                "near_black_fraction": [stats[i]["near_black_fraction"] for i in IDS],
                "masked_median_inverse_srgb_ratio": ratios,
                "simple_srgb_aperture_baseline": baseline,
                "brightness_order_pass": bool(
                    stats[4]["mean_encoded_luma"]
                    > stats[13]["mean_encoded_luma"]
                    > stats[22]["mean_encoded_luma"]
                ),
            }
        )
    ratio_9 = np.array(
        [
            r["masked_median_inverse_srgb_ratio"]["f9_over_f5"]
            for r in rows
            if r["masked_median_inverse_srgb_ratio"]["f9_over_f5"] is not None
        ]
    )
    ratio_16 = np.array(
        [
            r["masked_median_inverse_srgb_ratio"]["f16_over_f5"]
            for r in rows
            if r["masked_median_inverse_srgb_ratio"]["f16_over_f5"] is not None
        ]
    )
    rng = np.random.default_rng(190)
    ix9 = rng.integers(0, len(ratio_9), (5000, len(ratio_9)))
    ix16 = rng.integers(0, len(ratio_16), (5000, len(ratio_16)))
    report = {
        "scope": "20 hash-selected distinct source classes; same source, light-off, ISO250, 1/60s, f5/f9/f16; official JPEG release",
        "source_count": len(rows),
        "photometric_mask_valid_sources": len(ratio_9),
        "all_mean_brightness_monotone": sum(r["brightness_order_pass"] for r in rows),
        "ideal_pupil_area_ratios_to_f5": {
            "f9_over_f5": (5 / 9) ** 2,
            "f16_over_f5": (5 / 16) ** 2,
        },
        "source_median_inverse_srgb_ratios": {
            "f9_over_f5": {
                "median": float(np.median(ratio_9)),
                "min": float(ratio_9.min()),
                "max": float(ratio_9.max()),
                "source_bootstrap_95_percent": np.quantile(
                    np.median(ratio_9[ix9], axis=1), [0.025, 0.975]
                ).tolist(),
            },
            "f16_over_f5": {
                "median": float(np.median(ratio_16)),
                "min": float(ratio_16.min()),
                "max": float(ratio_16.max()),
                "source_bootstrap_95_percent": np.quantile(
                    np.median(ratio_16[ix16], axis=1), [0.025, 0.975]
                ).tolist(),
            },
        },
        "mean_near_white_fraction_f5_f9_f16": np.mean(
            [r["near_white_fraction"] for r in rows], axis=0
        ).tolist(),
        "mean_near_black_fraction_f5_f9_f16": np.mean(
            [r["near_black_fraction"] for r in rows], axis=0
        ).tolist(),
        "simple_srgb_aperture_baseline": {
            str(key): {
                "predicted_over_observed_mean_median": float(
                    np.median(
                        [
                            r["simple_srgb_aperture_baseline"][str(key)][
                                "predicted_over_observed_mean"
                            ]
                            for r in rows
                        ]
                    )
                ),
                "predicted_brighter_source_count": sum(
                    r["simple_srgb_aperture_baseline"][str(key)][
                        "predicted_over_observed_mean"
                    ]
                    > 1
                    for r in rows
                ),
                "rgb_mae_median": float(
                    np.median(
                        [
                            r["simple_srgb_aperture_baseline"][str(key)]["rgb_mae"]
                            for r in rows
                        ]
                    )
                ),
            }
            for key in (13, 22)
        },
        "rows": rows,
        "qualification": "inverse sRGB is not inverse camera ISP; JPEG, crop, black level, spectral response and unknown actual exposure prevent calibrating true photon ratio",
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "rows"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
