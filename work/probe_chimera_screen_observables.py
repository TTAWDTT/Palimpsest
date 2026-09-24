"""Measure preregistered output observables on nonreserved Chimera pairs.

All images are 256x256, so these numbers are composite publication-channel
observables; they are not estimates of display/optics/sensor parameters.
"""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import sobel


ROOT = Path("E:/ai_image_origin_research/data/derived/chimera_paired")
CONTROL = Path("E:/ai_image_origin_research/data/derived/chimera_recap256")
SPLIT = Path("E:/ai_image_origin_research/data/manifests/chimera_simulation_source_split.csv")
OUTPUT = Path("work/chimera_screen_observables_nonreserved.json")
CONDITIONS = ("recap_mac", "recap_monitor")


def features(path: Path) -> dict[str, float]:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    if rgb.shape != (256, 256, 3):
        raise RuntimeError(f"expected 256x256 RGB image: {path}")
    luminance = np.tensordot(rgb, np.array([.2126, .7152, .0722], dtype=np.float32), axes=1)
    gradient = np.hypot(sobel(luminance, axis=0) / 8, sobel(luminance, axis=1) / 8)
    window = np.outer(np.hanning(256), np.hanning(256)).astype(np.float32)
    power = np.abs(np.fft.fftshift(np.fft.fft2((luminance - luminance.mean()) * window))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(256))
    fx = fy
    radius = np.hypot(fy[:, None], fx[None, :])
    mid = float(power[(radius >= .05) & (radius < .25)].sum())
    high = float(power[(radius >= .25) & (radius < .45)].sum())
    return {
        "luma_mean": float(luminance.mean()),
        "luma_contrast_p90_p10": float(np.quantile(luminance, .9) - np.quantile(luminance, .1)),
        "chroma_spread_mean": float((rgb.max(axis=2) - rgb.min(axis=2)).mean()),
        "gradient_mean": float(gradient.mean()),
        "high_to_mid_spectral_power": high / max(mid, 1e-12),
    }


def numeric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {"median": float(np.median(array)),
            "p10": float(np.quantile(array, .1)),
            "p90": float(np.quantile(array, .9))}


def main() -> None:
    with SPLIT.open(newline="", encoding="utf-8-sig") as stream:
        sources = list(csv.DictReader(stream))
    if len(sources) != 1200:
        raise RuntimeError("split manifest incomplete")
    selected = [row for row in sources if row["split"] in ("calibration", "development")]
    if len(selected) != 360:
        raise RuntimeError("expected 360 nonreserved sources")
    table = {}
    for split in ("calibration", "development"):
        table[split] = {}
        for condition in CONDITIONS:
            table[split][condition] = {"paired_differences": {}, "source_values": {}, "recapture_values": {}}
    for row in selected:
        source = row["src"]
        original = features(ROOT / "stylegan2_orig" / source)
        for condition in CONDITIONS:
            recapture = features(CONTROL / condition / source)
            target = table[row["split"]][condition]
            for name, value in original.items():
                target["source_values"].setdefault(name, []).append(value)
                target["recapture_values"].setdefault(name, []).append(recapture[name])
                target["paired_differences"].setdefault(name, []).append(recapture[name] - value)
    result = {
        "scope": "Chimera simulation calibration/development split only, 256x256 common output size",
        "warning": "Composite published RGB observables; crop/registration and ISP can confound interpretation",
        "feature_definitions": {
            "luma_mean": "mean Rec.709 weighted gamma-encoded RGB, not physical luminance",
            "luma_contrast_p90_p10": "global 90th minus 10th percentile of luma",
            "chroma_spread_mean": "mean(max RGB - min RGB) per pixel",
            "gradient_mean": "mean central Sobel magnitude in 256x256 image",
            "high_to_mid_spectral_power": "Hann-windowed luma Fourier power .25-.45 / .05-.25 cycles per output pixel",
        },
        "counts": {"calibration": 240, "development": 120},
        "conditions": {
            split: {condition: {category: {feature: numeric_summary(values)
                                          for feature, values in groups.items()}
                               for category, groups in categories.items()}
                    for condition, categories in condition_table.items()}
            for split, condition_table in table.items()
        },
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({split: {condition: result["conditions"][split][condition]["paired_differences"]
                              for condition in CONDITIONS}
                      for split in ("calibration", "development")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
