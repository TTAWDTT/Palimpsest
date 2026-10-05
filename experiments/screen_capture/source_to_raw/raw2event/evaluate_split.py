"""Frozen 10/10 cross-content RAW test for the screen forward renderer.

Calibration/development membership, display assumptions and the three
mechanisms were fixed before downloading these twenty recordings. Per-prefix
render caches include a code/data fingerprint and can be resumed safely.
"""

from palimpsest.paths import DATA_ROOT, WORK_DIR

from collections import defaultdict
import argparse
import csv
import hashlib
import json
from pathlib import Path

from palimpsest.io.provenance import MODEL_ROOT, simulation_code_files

import numpy as np

from palimpsest.data.cifar10 import load_originals
from experiments.screen_capture.source_to_raw.raw2event.protocol import (
    cfa_design,
    direct_sample_control,
    evaluate,
    prepare,
    render,
)


SPLIT = DATA_ROOT / "manifests/raw2event_process_split_v1.csv"
SPLIT_SHA = "471fbff8020f88c1b73664e5794285df28da4792e5fb77bdfe682b5ee9bb7a43"
FIRST_FRAME_AUDIT = WORK_DIR / "raw2event_process_split_first_frame_audit.json"
REFINED_GEOMETRY = WORK_DIR / "raw2event_content_registered_geometry.json"
CACHE = DATA_ROOT / "derived/raw2event_source_to_raw_split_v1"
OUT = WORK_DIR / "raw2event_source_to_raw_split_v1.json"
LAYOUTS = ("vertical_rgb", "co_spatial_rgb_control", "simple_rgb_sample_control")
CODE = (
    *simulation_code_files(),
    MODEL_ROOT.parent / "data/video.py",
    MODEL_ROOT.parent / "data/raw2event.py",
    MODEL_ROOT.parent / "data/cifar10.py",
    Path("experiments/screen_capture/source_to_raw/raw2event/protocol.py"),
    Path("experiments/screen_capture/source_to_raw/raw2event/run_two_sources.py"),
    Path(
        "experiments/screen_capture/raw_geometry/raw2event/audit_split_first_frames.py"
    ),
    Path("experiments/screen_capture/source_to_raw/raw2event/evaluate_split.py"),
)


def fingerprint() -> str:
    digest = hashlib.sha256()
    digest.update(SPLIT_SHA.encode("ascii"))
    for path in CODE:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def cached_simulation(
    row: dict, prepared: dict, layout: str, code_hash: str, cache_root: Path
) -> tuple[np.ndarray, float]:
    path = cache_root / code_hash[:16] / layout / f"{row['prefix']}.npz"
    source_hash = hashlib.sha256(prepared["drive"].tobytes()).hexdigest()
    key_data = (
        f"{code_hash}|{row['prefix']}|{layout}|{source_hash}".encode("utf-8")
        + prepared["H"].tobytes()
        + np.asarray(prepared["roi"], dtype=np.int64).tobytes()
    )
    key = hashlib.sha256(key_data).hexdigest()
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if (
                data["key"].item() != key
                or data["sim"].shape != prepared["actual"].shape
            ):
                raise RuntimeError(f"stale simulation cache: {path}")
            return data["sim"].copy(), float(data["render_seconds"])
    sim, seconds = (
        direct_sample_control(prepared)
        if layout == "simple_rgb_sample_control"
        else render(prepared, layout)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    np.savez_compressed(
        partial,
        sim=sim.astype(np.float32),
        render_seconds=np.asarray(seconds),
        key=np.asarray(key),
    )
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
    return {
        "n_sources": len(subset),
        "mean_source_mae_counts": float(maes.mean()),
        "median_source_mae_counts": float(np.median(maes)),
        "mean_source_pearson": float(corrs.mean()),
        "pooled_pixel_mae_counts": float(
            sum(row["mae_counts"] * row["n"] for row in subset)
            / sum(row["n"] for row in subset)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--geometry",
        choices=("tag_similarity", "source_rgb_refined"),
        default="tag_similarity",
    )
    args = parser.parse_args()
    if hashlib.sha256(SPLIT.read_bytes()).hexdigest() != SPLIT_SHA:
        raise RuntimeError("frozen split changed")
    frame_audit = json.loads(FIRST_FRAME_AUDIT.read_text(encoding="utf-8"))
    if frame_audit["count"] != 20 or not frame_audit["complete_split"]:
        raise RuntimeError(
            "twenty first-frame geometry/association audits not complete"
        )
    split = [
        row
        for row in csv.DictReader(SPLIT.open(encoding="utf-8", newline=""))
        if row["role"] in ("calibration", "development")
    ]
    if len(split) != 20:
        raise RuntimeError("expected 10 calibration and 10 development sources")
    originals = load_originals(split)
    first_frames = {row["prefix"]: row for row in frame_audit["rows"]}
    if set(first_frames) != {row["prefix"] for row in split}:
        raise RuntimeError("first-frame audit does not cover exact frozen split")
    refined = None
    if args.geometry == "source_rgb_refined":
        ref_audit = json.loads(REFINED_GEOMETRY.read_text(encoding="utf-8"))
        refined = {row["prefix"]: row for row in ref_audit["records"]}
        if ref_audit["n"] != 20 or set(refined) != set(first_frames):
            raise RuntimeError("refined geometry does not cover exact frozen split")
    cache_root = (
        CACHE
        if refined is None
        else CACHE.with_name(CACHE.name + "_source_rgb_refined")
    )
    output_path = (
        OUT if refined is None else OUT.with_name(OUT.stem + "_source_rgb_refined.json")
    )
    code_hash = fingerprint()
    by_layout = defaultdict(list)
    for number, row in enumerate(split, start=1):
        corners = np.asarray(
            first_frames[row["prefix"]]["tag_similarity_content_corners"]
            if refined is None
            else refined[row["prefix"]]["refined_corners"],
            dtype=np.float32,
        )
        prepared = prepare(row["prefix"], None, originals[row["prefix"]], corners)
        for layout in LAYOUTS:
            sim, seconds = cached_simulation(
                row, prepared, layout, code_hash, cache_root
            )
            by_layout[layout].append((row, prepared, sim, seconds))
        print(
            f"rendered {number}/20 {row['role']} {row['class_name']} {row['prefix']}",
            flush=True,
        )
    conditions = {}
    for layout, entries in by_layout.items():
        weights = pooled_fit(entries)
        metrics = []
        for row, prepared, sim, seconds in entries:
            item = {
                "prefix": row["prefix"],
                "role": row["role"],
                "class_name": row["class_name"],
                "capture_day": row["capture_day"],
                "render_seconds": seconds,
            }
            item.update(evaluate(sim, prepared["actual"], prepared["mask"], weights))
            metrics.append(item)
        conditions[layout] = {
            "counts_fit": {
                "intercept": float(weights[0]),
                "r_gain": float(weights[1]),
                "g_gain": float(weights[2]),
                "b_gain": float(weights[3]),
            },
            "calibration": aggregate(metrics, "calibration"),
            "development": aggregate(metrics, "development"),
            "median_render_seconds": float(
                np.median([row["render_seconds"] for row in metrics])
            ),
            "per_source": metrics,
        }
    report = {
        "split_sha256": SPLIT_SHA,
        "code_fingerprint": code_hash,
        "source_count": 20,
        "roles": {"calibration": 10, "development": 10},
        "geometry": args.geometry,
        "process_settings": "192 display pixels and 0.8 sensor-pixel Gaussian are virtual; Tag similarity geometry selected after partial development audit; optional source-RGB refinement optimized known-source ZNCC after first RAW results",
        "conditions": conditions,
        "qualification": "raw mosaics; geometry selected after inspecting development subset, and optional RGB-source refinement is post-hoc exploratory but uses no RAW target; unknown actual screen raster and CFA phase; class/date confounded; reserved untouched",
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                layout: {
                    "development": value["development"],
                    "median_render_seconds": value["median_render_seconds"],
                }
                for layout, value in conditions.items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
