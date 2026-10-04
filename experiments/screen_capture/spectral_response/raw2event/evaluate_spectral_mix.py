"""Fit effective display-primary to RAW-CFA coupling on frozen calibration sources.

The source-to-screen and spatial assumptions remain virtual. The learned
nonnegative 3x3 coefficients combine unknown display spectra, optics and sensor
responses; they must not be described as an identified sensor spectrum."""

import csv
import hashlib
import json
import numpy as np
from palimpsest.data.cifar10 import load_originals
from experiments.screen_capture.source_to_raw.raw2event.protocol import prepare

from experiments.screen_capture.spectral_response.raw2event.protocol import (
    SPLIT,
    SPLIT_SHA,
    GEOMETRY,
    OUT,
    METHODS,
    code_hash,
    cached_bands,
    fit_nnls,
    evaluate,
)


def main() -> None:
    if hashlib.sha256(SPLIT.read_bytes()).hexdigest() != SPLIT_SHA:
        raise RuntimeError("frozen split changed")
    rows = [
        row
        for row in csv.DictReader(SPLIT.open(encoding="utf-8", newline=""))
        if row["role"] in ("calibration", "development")
    ]
    geometry = {
        row["prefix"]: row
        for row in json.loads(GEOMETRY.read_text(encoding="utf-8"))["records"]
    }
    if len(rows) != 20 or set(geometry) != {row["prefix"] for row in rows}:
        raise RuntimeError("complete twenty-source RGB-only registration required")
    originals = load_originals(rows)
    fingerprint = code_hash()
    entries = {method: [] for method in METHODS}
    for number, row in enumerate(rows, start=1):
        corners = np.asarray(
            geometry[row["prefix"]]["refined_corners"], dtype=np.float32
        )
        prepared = prepare(row["prefix"], None, originals[row["prefix"]], corners)
        for method in METHODS:
            bands, seconds = cached_bands(row, prepared, method, fingerprint)
            entries[method].append((row, prepared, bands, seconds))
        print(
            f"rendered bands {number}/20 {row['role']} {row['class_name']}", flush=True
        )
    conditions = {}
    for method in METHODS:
        for coupling in ("diagonal", "nonnegative_full"):
            weights = fit_nnls(entries[method], diagonal=(coupling == "diagonal"))
            conditions[f"{method}__{coupling}"] = evaluate(entries[method], weights)
    report = {
        "split_sha256": SPLIT_SHA,
        "code_fingerprint": fingerprint,
        "n_sources": 20,
        "n_calibration": 10,
        "n_development": 10,
        "geometry": "known-source RGB-only local optimization, selected after initial development RAW results",
        "fit": "nonnegative least squares, separate intercept per CFA color; calibration only",
        "coupling_interpretation": "effective display-primary-to-RAW response; cannot identify physical sensor spectral sensitivity separately",
        "conditions": conditions,
        "qualification": "exploratory post-hoc physical diagnosis, not a preregistered heldout validation; display raster, gamma, CFA phase and blur are virtual, reserved untouched",
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value["development"] for key, value in conditions.items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
