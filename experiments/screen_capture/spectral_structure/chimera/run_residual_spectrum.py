"""Probe unfitted real-minus-proxy spatial residuals on development sources.

This is exploratory. A residual spectral peak is not automatically moire or
evidence for a specific display subpixel/CFA mechanism.
"""

from palimpsest.paths import DATA_ROOT, WORK_DIR

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr

from experiments.screen_capture.photometry.chimera.protocol import (
    align_rgb,
    design,
    load_rgb,
)
from experiments.screen_capture.photometry.chimera.evaluate_composite_photometry import (
    CONTROL,
    ROOT,
)


GEOMETRY = WORK_DIR / "chimera_fixed_publication_geometry.json"
PHOTOMETRY = WORK_DIR / "chimera_composite_photometry.json"
BLUR = WORK_DIR / "chimera_effective_blur_proxy.json"
SPLIT = DATA_ROOT / "manifests/chimera_simulation_source_split.csv"
OUTPUT = WORK_DIR / "chimera_residual_spectrum_development.json"


def luminance(rgb: np.ndarray) -> np.ndarray:
    return np.tensordot(
        rgb, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axes=1
    )


def features(
    source: str, condition: str, warp: np.ndarray, coefficient: np.ndarray, sigma: float
) -> dict:
    original = load_rgb(ROOT / "stylegan2_orig" / source)
    real = align_rgb(load_rgb(CONTROL / condition / source), warp)
    predicted = np.clip((design(original) @ coefficient).reshape(256, 256, 3), 0, 1)
    if sigma:
        predicted = cv2.GaussianBlur(
            predicted, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
        )
    residual = luminance(real - predicted)
    src_gray = luminance(original)
    gx = cv2.Sobel(src_gray, cv2.CV_32F, 1, 0) / 8
    gy = cv2.Sobel(src_gray, cv2.CV_32F, 0, 1) / 8
    gradient = np.hypot(gx, gy)
    inner = (slice(16, 240), slice(16, 240))
    low = np.quantile(gradient[inner], 0.25)
    flat_mask = gradient[inner] <= low
    highpass = residual - cv2.GaussianBlur(
        residual, (0, 0), sigmaX=1.5, borderType=cv2.BORDER_REFLECT
    )
    flat_highpass_rms = float(np.sqrt(np.mean(np.square(highpass[inner][flat_mask]))))

    candidates = [
        (float(gradient[y : y + 64, x : x + 64].mean()), y, x)
        for y in range(16, 177, 16)
        for x in range(16, 177, 16)
    ]
    _, py, px = min(candidates)
    patch = residual[py : py + 64, px : px + 64]
    window = np.outer(np.hanning(64), np.hanning(64)).astype(np.float32)
    power = np.abs(np.fft.fftshift(np.fft.fft2((patch - patch.mean()) * window))) ** 2
    frequencies = np.fft.fftshift(np.fft.fftfreq(64))
    radius = np.hypot(frequencies[:, None], frequencies[None, :])
    annulus = power[(radius >= 0.1) & (radius <= 0.45)]
    peak_ratio = float(annulus.max() / max(float(np.median(annulus)), 1e-12))
    peak_index = np.unravel_index(
        np.argmax(power * ((radius >= 0.1) & (radius <= 0.45))), power.shape
    )
    return {
        "residual_luma_mae": float(np.abs(residual[inner]).mean()),
        "flat_highpass_rms": flat_highpass_rms,
        "flat_patch_peak_to_median_power": peak_ratio,
        "flat_patch_xy": [px, py],
        "peak_frequency_xy_cycles_per_output_pixel": [
            float(frequencies[peak_index[1]]),
            float(frequencies[peak_index[0]]),
        ],
    }


def main() -> None:
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    photometry = json.loads(PHOTOMETRY.read_text(encoding="utf-8"))
    blur = json.loads(BLUR.read_text(encoding="utf-8"))
    with SPLIT.open(newline="", encoding="utf-8-sig") as stream:
        development = [
            row for row in csv.DictReader(stream) if row["split"] == "development"
        ]
    if len(development) != 120:
        raise RuntimeError("expected 120 development sources")

    # The inference CSV does not have a condition column; infer it from path.
    def score_by_condition(path: Path) -> dict[tuple[str, str], float]:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            return {
                (row["filename"].split("/")[0], row["src"]): float(row["score"])
                for row in csv.DictReader(stream)
            }

    real_scores = score_by_condition(WORK_DIR / "chimera_bfree_recap256.csv")
    sim_scores = score_by_condition(WORK_DIR / "chimera_composite_sim_bfree_dev.csv")
    result = {
        "scope": "development-only real minus frozen proxy residual, at common 256px size",
        "warning": "frequency peaks are descriptive; neither moire nor physical noise labels",
        "conditions": {},
    }
    for condition in ("recap_mac", "recap_monitor"):
        warp = np.asarray(
            geometry["conditions"][condition]["median_calibration_affine"],
            dtype=np.float32,
        )
        coefficient = np.asarray(
            photometry["conditions"][condition]["rgb3_plus_bias_coefficient"],
            dtype=np.float64,
        )
        sigma = float(blur["conditions"][condition]["selected_sigma_output_pixels"])
        records = []
        for row in development:
            source = row["src"]
            observed = features(source, condition, warp, coefficient, sigma)
            observed["src"] = source
            observed["label"] = row["label"]
            observed["absolute_bfree_score_error"] = abs(
                real_scores[(condition, source)] - sim_scores[(condition, source)]
            )
            records.append(observed)
        summary = {}
        score_error = np.array([item["absolute_bfree_score_error"] for item in records])
        for key in (
            "residual_luma_mae",
            "flat_highpass_rms",
            "flat_patch_peak_to_median_power",
        ):
            values = np.array([item[key] for item in records])
            correlation = spearmanr(values, score_error)
            summary[key] = {
                "median": float(np.median(values)),
                "p05": float(np.quantile(values, 0.05)),
                "p95": float(np.quantile(values, 0.95)),
                "spearman_with_absolute_bfree_score_error": float(
                    correlation.statistic
                ),
                "spearman_pvalue_exploratory": float(correlation.pvalue),
            }
        result["conditions"][condition] = {
            "count": len(records),
            "summary": summary,
            "records": records,
        }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value["summary"] for key, value in result["conditions"].items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
