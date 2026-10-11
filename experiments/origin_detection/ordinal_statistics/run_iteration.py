"""Signed ordinal cache, two-rule development screening and single-file timing."""

import argparse
import csv
from datetime import datetime, timezone
import json
import platform
import random
from time import perf_counter
import tomllib

import cv2
import numpy as np
from PIL import Image

from palimpsest.detection.algorithms.ordinal_statistics.detector import OrdinalDetector
from palimpsest.detection.algorithms.ordinal_statistics.features import FEATURE_NAMES, extract_features
from palimpsest.detection.files import predict_file
from palimpsest.evaluation.timing import percentile
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import DATA_ROOT, REPO_ROOT, WORK_DIR
from experiments.origin_detection.robust_statistics.rr.protocol import verified_inventory, MANIFEST, TRAINVAL
from .audit_sources import chimera_inventory
from .fit_rules import run_screen, validate_rows

CONFIG = REPO_ROOT / "configs/evaluation/joint_ordinal_statistics.toml"
OUTPUT = WORK_DIR / "robust_statistics/joint_ordinal_statistics"
CHIMERA_MANIFEST = DATA_ROOT / "manifests/chimera_bfree_manifest.csv"
CHIMERA_AUDIT = DATA_ROOT / "manifests/chimera_paired_image_audit.csv"


def write_json(path, value):
    # Serialize fully before mutation; numpy scalar leakage/NaN is a hard failure.
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.write_text(encoded, encoding="utf-8")


def pins():
    paths = [CONFIG, REPO_ROOT / "src/palimpsest/data/source_groups.py",
             REPO_ROOT / "src/palimpsest/detection/algorithms/local_statistics/features.py",
             REPO_ROOT / "experiments/origin_detection/robust_statistics/rr/evaluate_features.py"]
    for directory in (REPO_ROOT / "src/palimpsest/detection/algorithms/ordinal_statistics",
                      REPO_ROOT / "experiments/origin_detection/ordinal_statistics"):
        paths.extend(directory.glob("*.py"))
        paths.append(directory / "README.md")
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)}


def inventory(config):
    rr, audit = verified_inventory()
    for path, sha in ((CHIMERA_MANIFEST, "39da9e9eac3bd2d133c53769f9c5a158c1769039c67560c0bbfc4660d09a7d24"),
                      (CHIMERA_AUDIT, "1cb41a8ee2b33b187b80d6fb294216c46798faffd7495f1883c480a0a2f0cfc8")):
        if file_sha256(path) != sha:
            raise ValueError("Historical Chimera inventory changed")
    external = {r["sha256"] for p in (MANIFEST, TRAINVAL) for r in read_rows(p)}
    chimera, summary = chimera_inventory(read_rows(CHIMERA_MANIFEST), read_rows(CHIMERA_AUDIT), external, config)
    normalized = [{"filename": r["filename"], "src": "rr/" + r["source_group"], "condition": r["condition"],
                   "label": "FAKE" if r["label"] == "ai" else "REAL", "scene": "all", "domain": "rr",
                   "sha256": r["sha256"], "width": r["width"], "height": r["height"], "role": r["algorithm_role"]} for r in rr]
    combined = normalized + chimera
    # Dataset filenames need globally unique namespace in the cache.
    for r in combined:
        r["relative_path"] = r["filename"]
        r["filename"] = r["domain"] + "/" + r["filename"]
    return combined, {"rr_audit_sha256": file_sha256(WORK_DIR / "robust_statistics/rr_first_iteration/source_audit.json"),
                      "rr_development_manifest_sha256": audit["development_manifest_sha256"], "chimera": summary,
                      "inputs": {str(p): file_sha256(p) for p in (CHIMERA_MANIFEST, CHIMERA_AUDIT, MANIFEST, TRAINVAL)}}


def image_path(row):
    root = DATA_ROOT / "derived" / ("rr_test" if row["domain"] == "rr" else "chimera_paired")
    relative = row["relative_path"]
    path = root / relative
    if not path.is_relative_to(root) or ".." in path.parts:
        raise ValueError("Image path escapes data root")
    return path


def extract_inventory(rows, output):
    features, start = [], perf_counter()
    for index, row in enumerate(rows, 1):
        path = image_path(row)
        if file_sha256(path) != row["sha256"]:
            raise ValueError(f"Image digest changed: {path}")
        with Image.open(path) as im:
            if im.size != (int(row["width"]), int(row["height"])):
                raise ValueError("Image dimensions changed")
            rgb = np.asarray(im.convert("RGB"))
        result = extract_features(rgb)
        features.append({**row, **dict(zip(FEATURE_NAMES, result.values.tolist())),
                         "preprocess_ms": result.preprocess_ms, "statistics_ms": result.statistics_ms})
        if index % 60 == 0 or index == len(rows):
            print(json.dumps({"completed": index, "total": len(rows), "elapsed_s": round(perf_counter() - start, 3)}), flush=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(features[0]))
        writer.writeheader()
        writer.writerows(features)
    return perf_counter() - start


def load_cache(config):
    receipt = json.loads((OUTPUT / "features.json").read_text(encoding="utf-8"))
    if receipt["code_pins"] != pins():
        raise ValueError("Frozen ordinal code/config changed")
    for name, sha in receipt["outputs"].items():
        if file_sha256(OUTPUT / name) != sha:
            raise ValueError("Ordinal cache signature changed")
    current, audit = inventory(config)
    saved = read_rows(OUTPUT / "inventory.csv")
    if saved != [{k: str(v) for k, v in r.items()} for r in current] or audit != receipt["audit"]:
        raise ValueError("Source inventory audit changed")
    return validate_rows(read_rows(OUTPUT / "features.csv"), saved), saved, receipt


def benchmark(rows, rules):
    # Reuse the signed historical 60-image selection, without resampling it.
    prior = WORK_DIR / "robust_statistics/rr_first_iteration"
    prior_receipt = json.loads((prior / "benchmark.json").read_text(encoding="utf-8"))
    if file_sha256(prior / "benchmark.csv") != prior_receipt["csv_sha256"]:
        raise ValueError("Historical benchmark image selection changed")
    filenames = sorted({"rr/" + r["filename"] for r in read_rows(prior / "benchmark.csv")})
    by_name = {r["filename"]: r for r in rows}
    selected = [by_name[n] for n in filenames]
    if len(selected) != 60:
        raise ValueError("Historical benchmark denominator changed")
    for row in selected:
        if file_sha256(image_path(row)) != row["sha256"]:
            raise ValueError("Benchmark image SHA changed")
    rng = random.Random(20261005)
    measurements, output = [], {}
    for name, rule in rules.items():
        detector = OrdinalDetector(rule)
        detector.predict(np.full((256, 256, 3), 128, np.uint8))
        for repeat in range(3):
            shuffled = selected[:]
            rng.shuffle(shuffled)
            for row in shuffled:
                result = predict_file(detector, image_path(row))
                expected = float(rule.score([float(row[f]) for f in FEATURE_NAMES]))
                if abs(expected - result.prediction.score) > 1e-12:
                    raise ValueError("Cached/live ordinal prediction mismatch")
                measurements.append({"mode": name, "filename": row["filename"], "repeat": repeat,
                                     "decode_ms": result.decode_ms, "end_to_end_ms": result.end_to_end_ms,
                                     "predict_ms": result.prediction.timing_ms["predict_ms"]})
        current = [r for r in measurements if r["mode"] == name]
        output[name] = {"images": len(selected), "runs": len(current),
                        "core": {f"p{int(q * 100)}": percentile([r["predict_ms"] for r in current], q) for q in (0.5, 0.95)},
                        "end_to_end": {f"p{int(q * 100)}": percentile([r["end_to_end_ms"] for r in current], q) for q in (0.5, 0.95)}}
    return {"methods": output, "measurements": measurements,
            "processor": platform.processor(), "python_version": platform.python_version(),
            "opencv_version": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
            "prior_selection_csv_sha256": prior_receipt["csv_sha256"],
            "scope": "CPU batch1 OpenCV1thread warm filesystem, 60 RR development images x3; not startup latency"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    with CONFIG.open("rb") as stream:
        config = tomllib.load(stream)
    if args.prepare or args.pilot:
        directory = OUTPUT if args.prepare else OUTPUT.with_name("joint_ordinal_pilot")
        if directory.exists():
            raise FileExistsError("Preserve prior results; output already exists")
        rows, audit = inventory(config)
        if args.pilot:
            random.Random(config["seed"]).shuffle(rows)
            rows = rows[:args.pilot]
        directory.mkdir(parents=True)
        with (directory / "inventory.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        elapsed = extract_inventory(rows, directory / "features.csv")
        if args.prepare:
            validate_rows(read_rows(directory / "features.csv"), read_rows(directory / "inventory.csv"))
        write_json(directory / "features.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                                                  "images": len(rows), "elapsed_s": elapsed, "audit": audit,
                                                  "code_pins": pins(), "outputs": {n: file_sha256(directory / n)
                                                      for n in ("inventory.csv", "features.csv")}})
        return
    rows, _, receipt = load_cache(config)
    if args.benchmark:
        from palimpsest.detection.algorithms.ordinal_statistics.detector import OrdinalRule
        path = OUTPUT / "benchmark.json"
        if path.exists():
            raise FileExistsError("Benchmark already exists")
        iteration = json.loads((OUTPUT / "iteration.json").read_text(encoding="utf-8"))
        if iteration["features_receipt_sha256"] != file_sha256(OUTPUT / "features.json"):
            raise ValueError("Iteration receipt/cache changed")
        for mode, sha in iteration["rule_files"].items():
            if file_sha256(OUTPUT / f"{mode}_rule.json") != sha:
                raise ValueError("Frozen ordinal rule changed")
        result = benchmark(rows, {m: OrdinalRule.load(OUTPUT / f"{m}_rule.json") for m in config["candidate_modes"]})
        result["features_receipt_sha256"] = file_sha256(OUTPUT / "features.json")
        write_json(path, result)
        return
    if (OUTPUT / "iteration.json").exists():
        raise FileExistsError("Preserve registered two-rule screening")
    result, rules = run_screen(rows, config, receipt["outputs"]["inventory.csv"])
    result["features_receipt_sha256"] = file_sha256(OUTPUT / "features.json")
    result["code_pins"] = pins()
    # Serialize first, then write the rules and receipt.
    json.dumps(result, allow_nan=False)
    for mode, rule in rules.items():
        rule.save(OUTPUT / f"{mode}_rule.json")
    result["rule_files"] = {m: file_sha256(OUTPUT / f"{m}_rule.json") for m in rules}
    write_json(OUTPUT / "iteration.json", result)
    print(json.dumps({"chosen": result["chosen"], "kept": {m: len(v["kept_features"]) for m, v in result["candidates"].items()}}))


if __name__ == "__main__":
    main()
