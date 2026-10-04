"""Recalibrate effective RAW counts after replacing RAW/RGB Tag extrapolation.

Geometry and two renderer conditions are fixed before fitting. Ten old
calibration sources fit count response; ten development and ten previously
opened confirmation sources are diagnostic. No reserved sources are opened.
"""

from palimpsest.paths import DATA_ROOT, WORK_DIR

import csv
import hashlib
import json

import numpy as np

from experiments.raw2event.audit_raw2event_split_first_frames import load_originals
from experiments.raw2event.evaluate_raw2event_spectral_mix import (
    cached_bands,
    code_hash,
)
from experiments.raw2event.probe_raw2event_cfa_phase import mask_for
from experiments.raw2event.probe_raw2event_source_to_raw import prepare


SPLIT = DATA_ROOT / "manifests/raw2event_process_split_v1.csv"
NEW = DATA_ROOT / "manifests/raw2event_phase_confirmation_v1.csv"
GEOM = WORK_DIR / "raw2event_sensor_geometry_audit.json"
OLD_CORNERS = WORK_DIR / "raw2event_content_registered_geometry.json"
NEW_CORNERS = WORK_DIR / "raw2event_phase_confirmation_evaluation.json"
OUT = WORK_DIR / "raw2event_global_geometry_process_evaluation.json"
METHODS = ("vertical_rgb", "co_spatial_rgb_control", "simple_rgb_sample_control")
PHASE = "BGGR"


def fit_diagonal(entries: list[tuple[dict, dict, np.ndarray]]) -> np.ndarray:
    weights = np.empty((3, 2), dtype=np.float64)
    for color in range(3):
        xs, ys = [], []
        for row, prepared, bands in entries:
            if row["role"] != "calibration":
                continue
            select = prepared["mask"] & mask_for(prepared["actual"].shape, PHASE, color)
            xs.append(bands[..., color][select].astype(np.float64))
            ys.append(prepared["actual"][select].astype(np.float64))
        x, y = np.concatenate(xs), np.concatenate(ys)
        slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
        slope = max(0, slope)
        weights[color] = [float(y.mean() - slope * x.mean()), slope]
    return weights


def evaluate(entries: list[tuple[dict, dict, np.ndarray]], weights: np.ndarray) -> dict:
    results = []
    for row, prepared, bands in entries:
        pred = np.empty(prepared["actual"].shape, np.float32)
        for color in range(3):
            channel = mask_for(pred.shape, PHASE, color)
            pred[channel] = (
                weights[color, 0] + weights[color, 1] * bands[..., color][channel]
            )
        select = prepared["mask"]
        x, y = pred[select], prepared["actual"][select]
        results.append(
            {
                "prefix": row["prefix"],
                "role": row["role"],
                "class_name": row["class_name"],
                "n_pixels": int(select.sum()),
                "mae_counts": float(np.mean(np.abs(x - y))),
                "pearson": float(np.corrcoef(x, y)[0, 1]),
            }
        )
    aggregates = {
        role: {
            "n_sources": len(subset),
            "mean_source_mae_counts": float(np.mean([r["mae_counts"] for r in subset])),
            "mean_source_pearson": float(np.mean([r["pearson"] for r in subset])),
        }
        for role in ("calibration", "development", "phase_confirmation")
        if (subset := [r for r in results if r["role"] == role])
    }
    return {
        "weights_intercept_gain_by_display_primary": weights.tolist(),
        "aggregate": aggregates,
        "per_source": results,
    }


def main() -> None:
    split = [
        r
        for r in csv.DictReader(SPLIT.open(encoding="utf-8", newline=""))
        if r["role"] in ("calibration", "development")
    ]
    newer = list(csv.DictReader(NEW.open(encoding="utf-8", newline="")))
    rows = split + newer
    if len(rows) != 30 or len({r["prefix"] for r in rows}) != 30:
        raise RuntimeError("expected fixed 30 permitted prefixes")
    originals = load_originals(rows)
    geometry = json.loads(GEOM.read_text(encoding="utf-8"))
    matrix = np.asarray(geometry["fit_matrices"]["global_affine"], np.float64)
    corners = {
        r["prefix"]: r["refined_corners"]
        for r in json.loads(OLD_CORNERS.read_text(encoding="utf-8"))["records"]
    }
    corners.update(
        {
            r["prefix"]: r["geometry"]["source_rgb_refined_corners"]
            for r in json.loads(NEW_CORNERS.read_text(encoding="utf-8"))["per_source"]
        }
    )
    if set(corners) != {r["prefix"] for r in rows}:
        raise RuntimeError("missing content geometry")
    entries = {name: [] for name in METHODS}
    fingerprint = code_hash()
    for i, row in enumerate(rows, 1):
        prepared = prepare(
            row["prefix"],
            None,
            originals[row["prefix"]],
            np.asarray(corners[row["prefix"]], np.float32),
            matrix,
        )
        for method in METHODS:
            bands, _ = cached_bands(row, prepared, method, fingerprint)
            entries[method].append((row, prepared, bands))
        print(f"rendered {i}/30 {row['role']} {row['class_name']}", flush=True)
    results = {
        method: evaluate(entries[method], fit_diagonal(entries[method]))
        for method in METHODS
    }
    report = {
        "manifest_sha256": {
            "old": hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
            "new": hashlib.sha256(NEW.read_bytes()).hexdigest(),
        },
        "global_geometry_audit_sha256": hashlib.sha256(GEOM.read_bytes()).hexdigest(),
        "radiance_code_fingerprint": fingerprint,
        "phase": PHASE,
        "virtual_display_settings": "same assumed 192 lattice, encoded-sRGB Lanczos, gamma2.2, blur0.8, fill0.85",
        "fit_scope": "effective intercept and nonnegative gain fitted on old ten calibration sources only",
        "test_scope": "development 10 and previously inspected confirmation 10, exploratory diagnostics",
        "qualification": "Raw2Event screen geometry and display pixels are unverified; old RGB content corners optimized with known source, not RAW; original reserved untouched",
        "methods": results,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {name: value["aggregate"] for name, value in results.items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
