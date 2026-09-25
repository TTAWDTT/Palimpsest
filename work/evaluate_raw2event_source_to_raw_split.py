"""Frozen 10/10 cross-content RAW test for the screen forward renderer.

Calibration/development membership, display assumptions and the three
mechanisms were fixed before downloading these twenty recordings. Per-prefix
render caches include a code/data fingerprint and can be resumed safely.
"""

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from work.audit_raw2event_split_first_frames import load_originals
from work.probe_raw2event_source_to_raw import (
    cfa_design, direct_sample_control, evaluate, prepare, render,
)


SPLIT = Path("E:/ai_image_origin_research/data/manifests/raw2event_process_split_v1.csv")
SPLIT_SHA = "471fbff8020f88c1b73664e5794285df28da4792e5fb77bdfe682b5ee9bb7a43"
FIRST_FRAME_AUDIT = Path("work/raw2event_process_split_first_frame_audit.json")
CACHE = Path("E:/ai_image_origin_research/data/derived/raw2event_source_to_raw_split_v1")
OUT = Path("work/raw2event_source_to_raw_split_v1.json")
LAYOUTS = ("vertical_rgb", "co_spatial_rgb_control", "simple_rgb_sample_control")
CODE = (
    Path("origin_simulation/screen_capture.py"),
    Path("origin_simulation/screen_pipeline.py"),
    Path("work/probe_raw2event_source_to_raw.py"),
    Path("work/evaluate_raw2event_source_to_raw_split.py"),
)


def fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(SPLIT_SHA.encode("ascii"))
    for path in CODE:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def cached_simulation(row: dict, prepared: dict, layout: str, code_hash: str) -> tuple[np.ndarray, float]:
    path = CACHE / layout / f"{row['prefix']}.npz"
    source_hash = hashlib.sha256(prepared["drive"].tobytes()).hexdigest()
    key = hashlib.sha256(f"{code_hash}|{row['prefix']}|{layout}|{source_hash}".encode("utf-8")).hexdigest()
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if data["key"].item() != key or data["sim"].shape != prepared["actual"].shape:
                raise RuntimeError(f"stale simulation cache: {path}")
            return data["sim"].copy(), float(data["render_seconds"])
    sim, seconds = (direct_sample_control(prepared) if layout == "simple_rgb_sample_control"
                    else render(prepared, layout))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(partial, sim=sim.astype(np.float32), render_seconds=np.asarray(seconds),
                        key=np.asarray(key))
    partial.replace(path)
    return sim, seconds


def pooled_fit(entries: list[tuple[dict, dict, np.ndarray, float]]) -> np.ndarray:
    gram = np.zeros((4, 4), dtype=np.float64)
    target = np.zeros(4, dtype=np.float64)
    for row, prepared, sim, _ in entries:
        if row["role"] != "calibration":
            continue
        design = cfa_design(sim)[prepared["mask"]].astype(np.float64)
        actual = prepared["actual"][prepared["mask"]].astype(np.float64)
        gram += design.T @ design
        target += design.T @ actual
    return np.linalg.solve(gram, target)


def aggregate(rows: list[dict], role: str) -> dict:
    subset = [row for row in rows if row["role"] == role]
    maes = np.array([row["mae_counts"] for row in subset])
    corrs = np.array([row["pearson"] for row in subset])
    return {"n_sources": len(subset), "mean_source_mae_counts": float(maes.mean()),
            "median_source_mae_counts": float(np.median(maes)),
            "mean_source_pearson": float(corrs.mean()),
            "pooled_pixel_mae_counts": float(sum(row["mae_counts"] * row["n"] for row in subset)
                                             / sum(row["n"] for row in subset))}


def main() -> None:
    if hashlib.sha256(SPLIT.read_bytes()).hexdigest() != SPLIT_SHA:
        raise RuntimeError("frozen split changed")
    frame_audit = json.loads(FIRST_FRAME_AUDIT.read_text(encoding="utf-8"))
    if frame_audit["count"] != 20 or not frame_audit["complete_split"]:
        raise RuntimeError("twenty first-frame geometry/association audits not complete")
    split = [row for row in csv.DictReader(SPLIT.open(encoding="utf-8", newline=""))
             if row["role"] in ("calibration", "development")]
    if len(split) != 20:
        raise RuntimeError("expected 10 calibration and 10 development sources")
    originals = load_originals(split)
    code_hash = fingerprint()
    by_layout = defaultdict(list)
    for number, row in enumerate(split, start=1):
        prepared = prepare(row["prefix"], None, originals[row["prefix"]])
        for layout in LAYOUTS:
            sim, seconds = cached_simulation(row, prepared, layout, code_hash)
            by_layout[layout].append((row, prepared, sim, seconds))
        print(f"rendered {number}/20 {row['role']} {row['class_name']} {row['prefix']}", flush=True)
    conditions = {}
    for layout, entries in by_layout.items():
        weights = pooled_fit(entries)
        metrics = []
        for row, prepared, sim, seconds in entries:
            item = {"prefix": row["prefix"], "role": row["role"], "class_name": row["class_name"],
                    "capture_day": row["capture_day"], "render_seconds": seconds}
            item.update(evaluate(sim, prepared["actual"], prepared["mask"], weights))
            metrics.append(item)
        conditions[layout] = {
            "counts_fit": {"intercept": float(weights[0]), "r_gain": float(weights[1]),
                           "g_gain": float(weights[2]), "b_gain": float(weights[3])},
            "calibration": aggregate(metrics, "calibration"),
            "development": aggregate(metrics, "development"),
            "median_render_seconds": float(np.median([row["render_seconds"] for row in metrics])),
            "per_source": metrics,
        }
    report = {"split_sha256": SPLIT_SHA, "code_fingerprint": code_hash,
              "source_count": 20, "roles": {"calibration": 10, "development": 10},
              "process_settings": "frozen as in work/probe_raw2event_source_to_raw.py; 192 display pixels and 0.8 sensor-pixel Gaussian are virtual",
              "conditions": conditions,
              "qualification": "raw mosaics; approximate fixed content corners, unknown actual screen raster and CFA phase; class/date confounded"}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({layout: {"development": value["development"],
                              "median_render_seconds": value["median_render_seconds"]}
                      for layout, value in conditions.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
