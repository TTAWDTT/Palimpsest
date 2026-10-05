"""Exploratory RAW prediction change from replacing unstable per-Tag extrapolation.

Uses frozen old ten-calibration color weights; ten already-inspected phase
confirmation sources are diagnostic only, not a fresh confirmatory holdout.
"""

from palimpsest.paths import DATA_ROOT, WORK_DIR

import csv
import json

import numpy as np

from palimpsest.data.cifar10 import load_originals
from experiments.screen_capture.cfa_phase.raw2event.evaluate_phase_confirmation import (
    metrics,
    predict_phase,
)
from experiments.screen_capture.spectral_response.raw2event.protocol import render_bands
from experiments.screen_capture.source_to_raw.raw2event.protocol import prepare
from palimpsest.data.raw2event import WIDTH, HEIGHT


MANIFEST = DATA_ROOT / "manifests/raw2event_phase_confirmation_v1.csv"
GEOM = WORK_DIR / "raw2event_sensor_geometry_audit.json"
OLD = WORK_DIR / "raw2event_phase_confirmation_evaluation.json"
WEIGHTS = WORK_DIR / "raw2event_cfa_phase_probe.json"
OUT = WORK_DIR / "raw2event_geometry_transfer_evaluation.json"
METHOD = "simple_rgb_sample_control"
PHASE = "BGGR"


def main() -> None:
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")))
    if len(rows) != 10:
        raise RuntimeError("expected fixed ten confirmation prefixes")
    originals = load_originals(rows)
    old = {
        r["prefix"]: r
        for r in json.loads(OLD.read_text(encoding="utf-8"))["per_source"]
    }
    matrix = np.asarray(
        json.loads(GEOM.read_text(encoding="utf-8"))["fit_matrices"]["global_affine"],
        np.float64,
    )
    weights = np.asarray(
        json.loads(WEIGHTS.read_text(encoding="utf-8"))["conditions"][METHOD][PHASE][
            "weights_black_and_gain_per_color"
        ],
        np.float64,
    )
    results = []
    for row in rows:
        prefix = row["prefix"]
        corners = np.asarray(
            old[prefix]["geometry"]["source_rgb_refined_corners"], np.float32
        )
        variants = {}
        full_predictions = {}
        full_masks = {}
        for name, override in (
            ("per_recording_tag_homography", None),
            ("ten_calibration_global_affine", matrix),
        ):
            prepared = prepare(prefix, None, originals[prefix], corners, override)
            bands, seconds = render_bands(prepared, METHOD)
            predicted = predict_phase(bands, PHASE, weights)
            stats = metrics(predicted, prepared)
            left, top, right, bottom = [int(v) for v in prepared["roi"]]
            canvas = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
            canvas[top:bottom, left:right] = predicted
            mask_canvas = np.zeros((HEIGHT, WIDTH), dtype=bool)
            mask_canvas[top:bottom, left:right] = prepared["mask"]
            full_predictions[name] = canvas
            full_masks[name] = mask_canvas
            variants[name] = {
                **stats,
                "render_seconds": seconds,
                "raw_corners": prepared["raw_corners"].tolist(),
                "roi": [int(v) for v in prepared["roi"]],
            }
        tag_name, global_name = (
            "per_recording_tag_homography",
            "ten_calibration_global_affine",
        )
        shared = full_masks[tag_name] & full_masks[global_name]
        if shared.sum() < 1000:
            raise RuntimeError(f"insufficient shared pixels: {prefix}")
        truth = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
        for name, override in ((tag_name, None), (global_name, matrix)):
            prepared = prepare(prefix, None, originals[prefix], corners, override)
            left, top, right, bottom = [int(v) for v in prepared["roi"]]
            truth[top:bottom, left:right] = prepared["actual"]
        shared_truth = truth[shared]
        shared_metrics = {
            name: {
                "n_pixels": int(shared.sum()),
                "mae_counts": float(
                    np.mean(np.abs(full_predictions[name][shared] - shared_truth))
                ),
                "pearson": float(
                    np.corrcoef(full_predictions[name][shared], shared_truth)[0, 1]
                ),
            }
            for name in (tag_name, global_name)
        }
        results.append(
            {
                "prefix": prefix,
                "class_name": row["class_name"],
                "variants": variants,
                "shared_pixel_metrics": shared_metrics,
            }
        )
        print(f"geometry comparison {len(results)}/10 {row['class_name']}", flush=True)
    aggregate = {}
    for name in ("per_recording_tag_homography", "ten_calibration_global_affine"):
        aggregate[name] = {
            "mean_source_mae_counts": float(
                np.mean([r["variants"][name]["mae_counts"] for r in results])
            ),
            "median_source_mae_counts": float(
                np.median([r["variants"][name]["mae_counts"] for r in results])
            ),
            "mean_source_pearson": float(
                np.mean([r["variants"][name]["pearson"] for r in results])
            ),
        }
    differences = np.asarray(
        [
            r["variants"]["ten_calibration_global_affine"]["mae_counts"]
            - r["variants"]["per_recording_tag_homography"]["mae_counts"]
            for r in results
        ]
    )
    shared_aggregate = {
        name: {
            "mean_source_mae_counts": float(
                np.mean(
                    [r["shared_pixel_metrics"][name]["mae_counts"] for r in results]
                )
            ),
            "mean_source_pearson": float(
                np.mean([r["shared_pixel_metrics"][name]["pearson"] for r in results])
            ),
        }
        for name in ("per_recording_tag_homography", "ten_calibration_global_affine")
    }
    shared_differences = np.asarray(
        [
            r["shared_pixel_metrics"]["ten_calibration_global_affine"]["mae_counts"]
            - r["shared_pixel_metrics"]["per_recording_tag_homography"]["mae_counts"]
            for r in results
        ]
    )
    report = {
        "scope": "ten already-opened phase-confirmation sources; exploratory geometry diagnostic",
        "method": METHOD,
        "phase": PHASE,
        "virtual_display_settings_unchanged": True,
        "old_color_weights_unchanged": True,
        "source_rgb_corners": "per-source known-original-to-ISP-RGB refinement; never optimized on RAW",
        "global_affine_fit": "forty Tag corner pairs from prior ten calibration sources only",
        "comparison_warning": "full-ROI scores have different masks; shared-pixel scores use intersection in original RAW coordinates",
        "aggregate": aggregate,
        "global_minus_tag_source_mae_mean": float(differences.mean()),
        "global_better_n": int(np.count_nonzero(differences < 0)),
        "shared_pixel_aggregate": shared_aggregate,
        "shared_global_minus_tag_source_mae_mean": float(shared_differences.mean()),
        "shared_global_better_n": int(np.count_nonzero(shared_differences < 0)),
        "results": results,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "aggregate": aggregate,
                "global_minus_tag_source_mae_mean": float(differences.mean()),
                "global_better_n": report["global_better_n"],
                "shared_pixel_aggregate": shared_aggregate,
                "shared_global_better_n": report["shared_global_better_n"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
