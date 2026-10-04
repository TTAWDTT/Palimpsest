"""Adapt Dzanic et al.'s spectral features and 5-NN to RRDataset.

This is an inspired cross-dataset adaptation, not a numerical reproduction of
the author's MATLAB code or the paper's reported scores. Sparse radial bins
are interpolated, and arbitrary-sized inputs use fixed center crops.
The author code is at github.com/tarikdzanic/FourierSpectrumDiscrepancies,
commit 7bfdda1793e8d61a355f4d3b0899fd1153cc50c7.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from experiments.baselines.evaluate_rr_bfree import paired_change
from experiments.baselines.score_published_logits import evaluate


TRAIN_ROOT = Path(r"E:\ai_image_origin_research\data\derived\rr_trainval")
TEST_ROOT = Path(r"E:\ai_image_origin_research\data\derived\rr_test")
TRAIN_MANIFEST = Path(
    r"E:\ai_image_origin_research\data\manifests\rr_trainval_files.csv"
)
TEST_MANIFEST = Path(r"E:\ai_image_origin_research\data\manifests\rr_test_files.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fixed_size_grayscale(path: Path) -> tuple[np.ndarray, bool]:
    with Image.open(path) as loaded:
        width, height = loaded.size
        short_side = min(width, height)
        if short_side >= 256:
            left = (width - 256) // 2
            top = (height - 256) // 2
            image = loaded.crop((left, top, left + 256, top + 256))
            resized = False
        else:
            left = (width - short_side) // 2
            top = (height - short_side) // 2
            image = loaded.crop((left, top, left + short_side, top + short_side))
            image = image.resize((256, 256), Image.Resampling.BICUBIC)
            resized = True
        image = ImageOps.exif_transpose(image).convert("L")
        return np.asarray(image, dtype=np.float32), resized


def spectral_features(gray: np.ndarray) -> np.ndarray:
    """Return slope, starting energy, ending energy of radial FFT magnitude.

    Uses author's 200 bins from radius 0.5 to 1, five-bin smoothing, and
    fitting from bin 141 (kT=0.85 in released MATLAB code). Sparse empty bins
    are interpolated. The slope minimizes L2 error with the starting magnitude
    fixed, using bounded golden search instead of MATLAB fminsearch.
    """
    if gray.shape != (256, 256):
        raise ValueError(f"Expected 256x256 gray image, got {gray.shape}")
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    # MATLAB's one-based loop uses i-nx, with nx=128 for a 256-pixel image.
    coordinates = np.arange(256, dtype=np.float64) - 127
    radius = np.hypot(coordinates[:, None], coordinates[None, :])
    maximum_radius = math.hypot(128, 128)
    selected = radius > 0.5 * maximum_radius
    selected_radius = radius[selected]
    selected_magnitude = spectrum[selected]
    low = selected_radius.min()
    high = selected_radius.max()
    edges = np.linspace(low, high, 201)
    bin_index = np.minimum(
        np.searchsorted(edges, selected_radius, side="right") - 1, 199
    )
    counts = np.bincount(bin_index, minlength=200)
    magnitudes = np.bincount(bin_index, weights=selected_magnitude, minlength=200)
    radial = magnitudes / np.maximum(counts, 1)
    # Sparse corners of a 256px FFT leave a few of the author's 200 bins empty.
    # Interpolate them explicitly; the released MATLAB code produces NaN there.
    populated = counts > 0
    radial[~populated] = np.interp(
        np.flatnonzero(~populated), np.flatnonzero(populated), radial[populated]
    )
    # The author's smooth(y,5) uses a shorter average at both endpoints.
    smooth = np.convolve(radial, np.ones(5), mode="same") / np.convolve(
        np.ones(200), np.ones(5), mode="same"
    )
    x = (edges[:-1] + edges[1:]) / 2
    x = x[140:]
    y = smooth[140:]
    start = float(y[0])
    end = float(y[-1])
    if start <= 0 and np.all(y == 0):
        return np.zeros(3, dtype=np.float64)
    if start <= 0 or not np.all(np.isfinite(y)):
        raise ValueError("Nonpositive or nonfinite radial spectrum")
    log_ratio = np.log(x / x[0])

    def error(slope: float) -> float:
        fitted = start * np.exp(slope * log_ratio)
        return float(np.sum((y - fitted) ** 2))

    # Bounded one-dimensional minimization replaces MATLAB fminsearch.
    left, right = -32.0, 32.0
    fraction = (math.sqrt(5) - 1) / 2
    c = right - fraction * (right - left)
    d = left + fraction * (right - left)
    fc, fd = error(c), error(d)
    for _ in range(64):
        if fc < fd:
            right, d, fd = d, c, fc
            c = right - fraction * (right - left)
            fc = error(c)
        else:
            left, c, fc = c, d, fd
            d = left + fraction * (right - left)
            fd = error(d)
    return np.array([(left + right) / 2, start, end], dtype=np.float64)


def extract(path: Path) -> tuple[np.ndarray, bool, float]:
    start = time.perf_counter()
    gray, resized = fixed_size_grayscale(path)
    try:
        features = spectral_features(gray)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    return features, resized, (time.perf_counter() - start) * 1000


def knn_score(query: np.ndarray, reference: np.ndarray, labels: np.ndarray) -> float:
    distances = np.sum((reference - query) ** 2, axis=1)
    neighbors = np.argpartition(distances, 5)[:5]
    return float(np.mean(labels[neighbors]) - 0.5)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)
    result: dict[str, object] = {}
    for condition, group in by_condition.items():
        timing = sorted(float(row["end_to_end_ms"]) for row in group)
        metric_rows = [
            {"src": row["src"], "label": row["label"], "Fourier-5NN": row["score"]}
            for row in group
        ]
        result[condition] = {
            "metrics": evaluate(metric_rows, "Fourier-5NN"),
            "latency_ms_p50": float(np.median(timing)),
            "latency_ms_p95": float(np.percentile(timing, 95)),
            "resized_short_images": sum(bool(row["resized"]) for row in group),
        }
    sources = {
        condition: {str(row["src"]): row for row in group}
        for condition, group in by_condition.items()
    }
    if "original" in sources:
        result["paired_changes"] = {
            condition: paired_change(sources["original"], sources[condition])
            for condition in ("transfer", "redigital")
            if condition in sources
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-sources-per-class",
        type=int,
        default=0,
        help="Zero means all test sources; positive means a deterministic pilot",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    trainval = read_csv(TRAIN_MANIFEST)
    test = read_csv(TEST_MANIFEST)
    print(f"trainval={len(trainval)} test={len(test)}", flush=True)

    split_data: dict[str, list[tuple[np.ndarray, int]]] = defaultdict(list)
    for index, row in enumerate(trainval, 1):
        features, _, _ = extract(TRAIN_ROOT / row["filename"])
        split_data[row["split"]].append((features, int(row["label"] == "ai")))
        if index % 500 == 0:
            print(f"trainval extracted {index}/{len(trainval)}", flush=True)
    training = split_data["train"]
    mean = np.mean([entry[0] for entry in training], axis=0)
    scale = np.std([entry[0] for entry in training], axis=0)
    scale = np.where(scale > 0, scale, 1)
    reference = (np.array([entry[0] for entry in training]) - mean) / scale
    labels = np.array([entry[1] for entry in training])
    validation = [
        {
            "src": f"val/{index}",
            "label": "FAKE" if label else "REAL",
            "Fourier-5NN": knn_score((features - mean) / scale, reference, labels),
        }
        for index, (features, label) in enumerate(split_data["val"])
    ]
    val_metrics = evaluate(validation, "Fourier-5NN")
    print("validation", json.dumps(val_metrics), flush=True)

    trainval_hashes = {row["sha256"] for row in trainval}
    excluded = {
        f"{row['label']}/{row['source_id']}"
        for row in test
        if row["condition"] == "original" and row["sha256"] in trainval_hashes
    }
    if len(excluded) != 14:
        raise ValueError(f"Expected 14 exact-overlap sources, found {len(excluded)}")
    eligible = [
        row for row in test if f"{row['label']}/{row['source_id']}" not in excluded
    ]
    if args.test_sources_per_class:
        selected = {
            label: set(
                sorted(
                    {row["source_id"] for row in eligible if row["label"] == label},
                    key=lambda source: hashlib.sha256(
                        f"rr-fourier-pilot-20260924/{label}/{source}".encode()
                    ).digest(),
                )[: args.test_sources_per_class]
            )
            for label in ("ai", "real")
        }
        eligible = [
            row for row in eligible if row["source_id"] in selected[row["label"]]
        ]
    results: list[dict[str, object]] = []
    for index, row in enumerate(eligible, 1):
        inference_start = time.perf_counter()
        features, resized, _ = extract(TEST_ROOT / row["filename"])
        score = knn_score((features - mean) / scale, reference, labels)
        latency = (time.perf_counter() - inference_start) * 1000
        results.append(
            {
                "filename": row["filename"],
                "condition": row["condition"],
                "src": f"{row['label']}/{row['source_id']}",
                "label": "FAKE" if row["label"] == "ai" else "REAL",
                "score": score,
                "resized": resized,
                "end_to_end_ms": latency,
                "slope": float(features[0]),
                "start_energy": float(features[1]),
                "end_energy": float(features[2]),
            }
        )
        if index % 1000 == 0:
            print(f"test extracted {index}/{len(eligible)}", flush=True)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with args.output_prefix.with_suffix(".csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "method": "Dzanic 2020 inspired three-spectrum-feature adaptation + 5NN, RR train only",
        "crop": "native 256 center crop; bicubic enlargement if shorter than 256",
        "excluded_exact_overlap_sources": len(excluded),
        "training_images": len(training),
        "validation_images": len(validation),
        "test_sources_per_class": args.test_sources_per_class,
        "validation_metrics": val_metrics,
        "test": summarize(results),
    }
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({"output": str(args.output_prefix), "test_rows": len(results)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
