"""Four CFA phase hypotheses on the same cached screen irradiance fields."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from work.audit_raw2event_split_first_frames import load_originals
from work.evaluate_raw2event_spectral_mix import cached_bands, code_hash
from work.probe_raw2event_source_to_raw import prepare


SPLIT = Path("E:/ai_image_origin_research/data/manifests/raw2event_process_split_v1.csv")
SPLIT_SHA = "471fbff8020f88c1b73664e5794285df28da4792e5fb77bdfe682b5ee9bb7a43"
GEOMETRY = Path("work/raw2event_content_registered_geometry.json")
OUT = Path("work/raw2event_cfa_phase_probe.json")
PHASES = {
    "RGGB": ((0, 1), (1, 2)),
    "BGGR": ((2, 1), (1, 0)),
    "GRBG": ((1, 0), (2, 1)),
    "GBRG": ((1, 2), (0, 1)),
}


def mask_for(shape: tuple[int, int], phase: str, band: int) -> np.ndarray:
    y, x = np.indices(shape)
    layout = np.asarray(PHASES[phase])
    return layout[y % 2, x % 2] == band


def fit(entries: list[tuple[dict, dict, np.ndarray, float]], phase: str) -> np.ndarray:
    weights = np.zeros((3, 2), dtype=np.float64)
    for band in range(3):
        gram = np.zeros((2, 2), dtype=np.float64)
        target = np.zeros(2, dtype=np.float64)
        for row, prepared, radiance, _ in entries:
            if row["role"] != "calibration":
                continue
            select = prepared["mask"] & mask_for(prepared["actual"].shape, phase, band)
            signal = radiance[..., band][select].astype(np.float64)
            actual = prepared["actual"][select].astype(np.float64)
            features = np.column_stack((np.ones_like(signal), signal))
            gram += features.T @ features
            target += features.T @ actual
        lower = np.linalg.cholesky(gram)
        weights[band], _ = nnls(lower.T, np.linalg.solve(lower, target))
    return weights


def evaluate(entries: list[tuple[dict, dict, np.ndarray, float]], phase: str,
             weights: np.ndarray) -> dict:
    records = []
    for row, prepared, radiance, _ in entries:
        actual = prepared["actual"]
        predicted = np.empty_like(actual)
        for band in range(3):
            mask = mask_for(actual.shape, phase, band)
            predicted[mask] = weights[band, 0] + weights[band, 1] * radiance[..., band][mask]
        select = prepared["mask"]
        truth = actual[select]
        estimate = predicted[select]
        records.append({"prefix": row["prefix"], "role": row["role"], "class_name": row["class_name"],
                        "mae_counts": float(np.abs(truth - estimate).mean()),
                        "pearson": float(np.corrcoef(truth, estimate)[0, 1])})
    aggregate = {}
    for role in ("calibration", "development"):
        subset = [r for r in records if r["role"] == role]
        aggregate[role] = {"n": len(subset),
                           "mean_source_mae_counts": float(np.mean([r["mae_counts"] for r in subset])),
                           "mean_source_pearson": float(np.mean([r["pearson"] for r in subset]))}
    return {"weights_black_and_gain_per_color": weights.tolist(), **aggregate, "per_source": records}


def main() -> None:
    if hashlib.sha256(SPLIT.read_bytes()).hexdigest() != SPLIT_SHA:
        raise RuntimeError("frozen split changed")
    rows = [r for r in csv.DictReader(SPLIT.open(encoding="utf-8", newline=""))
            if r["role"] in ("calibration", "development")]
    geometry = {r["prefix"]: r for r in json.loads(GEOMETRY.read_text(encoding="utf-8"))["records"]}
    if len(rows) != 20 or set(geometry) != {r["prefix"] for r in rows}:
        raise RuntimeError("complete split required")
    originals = load_originals(rows)
    fingerprint = code_hash()
    conditions = {}
    for method in ("vertical_rgb", "co_spatial_rgb_control", "simple_rgb_sample_control"):
        entries = []
        for row in rows:
            prepared = prepare(row["prefix"], None, originals[row["prefix"]],
                               np.asarray(geometry[row["prefix"]]["refined_corners"], dtype=np.float32))
            bands, seconds = cached_bands(row, prepared, method, fingerprint)
            entries.append((row, prepared, bands, seconds))
        conditions[method] = {}
        for phase in PHASES:
            weights = fit(entries, phase)
            conditions[method][phase] = evaluate(entries, phase, weights)
        print(method, {phase: round(value["development"]["mean_source_mae_counts"], 3)
                       for phase, value in conditions[method].items()}, flush=True)
    report = {"scope": "four assumed CFA phases on unchanged RGB-registered screen irradiance and frozen 10/10 split",
              "phase_status": "hypotheses; actual MKV Bayer phase unverified and possibly altered by crop/encode",
              "selection_warning": "phase comparison initiated after full-mix coefficients suggested red/blue swap; exploratory development analysis",
              "conditions": conditions}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
