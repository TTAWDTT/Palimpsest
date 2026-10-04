"""DCT first-digit statistics plus random forest on RRDataset.

Inspired by Bonettini et al. (ICPR 2020). This implementation uses a fixed
Benford reference distribution instead of their fitted generalized curve, and
computes block DCT in memory instead of reading JPEG entropy coefficients.
Its scores therefore do not numerically reproduce the published method.
"""

from __future__ import annotations

from palimpsest.paths import DATA_ROOT

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn
from sklearn.ensemble import RandomForestClassifier

from palimpsest.evaluation.pairing import paired_change
from palimpsest.evaluation.classification import evaluate


TRAIN_ROOT = DATA_ROOT / "derived/rr_trainval"
TEST_ROOT = DATA_ROOT / "derived/rr_test"
TRAIN_MANIFEST = DATA_ROOT / "manifests/rr_trainval_files.csv"
TEST_MANIFEST = DATA_ROOT / "manifests/rr_test_files.csv"
BASE_QUANTIZATION = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float64,
)
QUANTIZATION_95 = np.maximum(1, np.floor((BASE_QUANTIZATION * 10 + 50) / 100))
ZIGZAG_FIRST_NINE = (
    (0, 1),
    (1, 0),
    (2, 0),
    (1, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (2, 1),
    (3, 0),
)
BENFORD = np.log10(1 + 1 / np.arange(1, 10))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def crop_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as loaded:
        width, height = loaded.size
        short_side = min(width, height)
        left = (width - short_side) // 2
        top = (height - short_side) // 2
        square = loaded.crop((left, top, left + short_side, top + short_side))
        # A single normalized field of view across source and processed images.
        gray = square.convert("L").resize((256, 256), Image.Resampling.BICUBIC)
        return np.asarray(gray, dtype=np.float32)


def first_digit_features(gray: np.ndarray) -> np.ndarray:
    blocks = gray.reshape(32, 8, 32, 8).transpose(0, 2, 1, 3) - 128
    coefficients = dctn(blocks, axes=(-2, -1), norm="ortho")
    quantized = np.rint(coefficients / QUANTIZATION_95)
    features = []
    for row, column in ZIGZAG_FIRST_NINE:
        values = np.abs(quantized[..., row, column].ravel())
        nonzero = values[values > 0]
        if len(nonzero):
            magnitude = np.floor(np.log10(nonzero))
            digits = np.floor(nonzero / np.power(10.0, magnitude)).astype(np.int64)
            digits = np.clip(digits, 1, 9)
            distribution = np.bincount(digits, minlength=10)[1:10] / len(nonzero)
        else:
            distribution = np.zeros(9)
        midpoint = (distribution + BENFORD) / 2
        js = 0.5 * np.sum(
            np.where(
                distribution > 0,
                distribution * np.log2(np.maximum(distribution, 1e-12) / midpoint),
                0,
            )
        )
        js += 0.5 * np.sum(BENFORD * np.log2(BENFORD / midpoint))
        features.extend((float(js), len(nonzero) / len(values)))
    result = np.asarray(features, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("Nonfinite DCT feature")
    return result


def extract(path: Path) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    features = first_digit_features(crop_grayscale(path))
    return features, (time.perf_counter() - start) * 1000


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)
    summary: dict[str, object] = {}
    for condition, group in by_condition.items():
        metric_rows = [
            {"src": row["src"], "label": row["label"], "Benford-RF": row["score"]}
            for row in group
        ]
        times = [float(row["end_to_end_ms"]) for row in group]
        summary[condition] = {
            "metrics": evaluate(metric_rows, "Benford-RF"),
            "latency_ms_p50": float(np.percentile(times, 50)),
            "latency_ms_p95": float(np.percentile(times, 95)),
        }
    sources = {
        condition: {str(row["src"]): row for row in group}
        for condition, group in by_condition.items()
    }
    summary["paired_changes"] = {
        condition: paired_change(sources["original"], sources[condition])
        for condition in ("transfer", "redigital")
        if condition in sources
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-sources-per-class", type=int, default=0)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    trainval = read_csv(TRAIN_MANIFEST)
    test = read_csv(TEST_MANIFEST)
    split_features: dict[str, list[np.ndarray]] = defaultdict(list)
    split_labels: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(trainval, 1):
        features, _ = extract(TRAIN_ROOT / row["filename"])
        split_features[row["split"]].append(features)
        split_labels[row["split"]].append(int(row["label"] == "ai"))
        if index % 500 == 0:
            print(f"trainval {index}/{len(trainval)}", flush=True)
    forest = RandomForestClassifier(
        n_estimators=100, random_state=20260924, n_jobs=1, min_samples_leaf=1
    )
    forest.fit(np.array(split_features["train"]), np.array(split_labels["train"]))
    val_score = forest.predict_proba(np.array(split_features["val"]))[:, 1] - 0.5
    val_rows = [
        {
            "src": f"val/{index}",
            "label": "FAKE" if label else "REAL",
            "Benford-RF": float(score),
        }
        for index, (label, score) in enumerate(zip(split_labels["val"], val_score))
    ]
    val_metrics = evaluate(val_rows, "Benford-RF")
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

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_prefix.with_suffix(".csv")
    if output_csv.exists() and not args.resume:
        raise FileExistsError(f"Use --resume to continue {output_csv}")
    results: list[dict[str, object]] = (
        read_csv(output_csv) if output_csv.exists() else []
    )
    if any(
        row["filename"] != expected["filename"]
        or row["src"] != f"{expected['label']}/{expected['source_id']}"
        or row["label"] != ("FAKE" if expected["label"] == "ai" else "REAL")
        for row, expected in zip(results, eligible)
    ) or len(results) > len(eligible):
        raise ValueError("Existing CSV is not a prefix of the selected test manifest")
    fields = ("filename", "condition", "src", "label", "score", "end_to_end_ms")
    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not results:
            writer.writeheader()
        for index, row in enumerate(eligible[len(results) :], len(results) + 1):
            start = time.perf_counter()
            features, _ = extract(TEST_ROOT / row["filename"])
            score = float(forest.predict_proba(features[None, :])[0, 1] - 0.5)
            result = {
                "filename": row["filename"],
                "condition": row["condition"],
                "src": f"{row['label']}/{row['source_id']}",
                "label": "FAKE" if row["label"] == "ai" else "REAL",
                "score": score,
                "end_to_end_ms": (time.perf_counter() - start) * 1000,
            }
            writer.writerow(result)
            results.append(result)
            if index % 1000 == 0:
                handle.flush()
                print(f"test {index}/{len(eligible)}", flush=True)
    summary = {
        "method": "Bonettini 2020 inspired fixed-Benford DCT first digits + RF100",
        "crop": "center square resize 256 bicubic; block DCT; fixed JPEG-Q95 luminance table",
        "training_images": len(split_labels["train"]),
        "validation_images": len(split_labels["val"]),
        "excluded_exact_overlap_sources": len(excluded),
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
