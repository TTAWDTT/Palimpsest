"""Fourth iteration from an inherited signed inventory, never RR reserved."""

import argparse
import csv
from datetime import datetime, timezone
from io import BytesIO
import json
import platform
import random
from time import perf_counter
import tomllib

import cv2
import numpy as np
from PIL import Image

from palimpsest.detection.algorithms.phase_statistics.detector import PhaseDetector, PhaseRule
from palimpsest.detection.algorithms.phase_statistics.features import FEATURE_NAMES, extract_features
from palimpsest.detection.algorithms.local_statistics.features import bounded_rgb
from palimpsest.evaluation.features import validate_feature_cache
from palimpsest.evaluation.file_benchmark import benchmark_files
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.ordinal_statistics.run_iteration import (
    CONFIG as PARENT_CONFIG, OUTPUT as PARENT, load_cache as parent_cache, image_path, write_json,
)
from .fit_rules import run_screen

CONFIG = REPO_ROOT / "configs/evaluation/joint_phase_statistics.toml"
OUTPUT = WORK_DIR / "robust_statistics/joint_phase_statistics"
VARIANTS = ("raw", "jpeg90_444_after_resize")
PARENT_SHA = "b23f79dffe79e0e4cfd2bae2b4a51b82d883d03e1b9d478ddf1285a33c8d06b4"


def inherited_inventory():
    if file_sha256(PARENT / "features.json") != PARENT_SHA:
        raise ValueError("Inherited source receipt changed")
    with PARENT_CONFIG.open("rb") as stream:
        _, rows, receipt = parent_cache(tomllib.load(stream))
    return rows, receipt["outputs"]["inventory.csv"]


def code_pins():
    files = [CONFIG, REPO_ROOT / "src/palimpsest/evaluation/features.py", REPO_ROOT / "src/palimpsest/evaluation/file_benchmark.py"]
    for directory in (REPO_ROOT / "src/palimpsest/detection/algorithms/phase_statistics",
                      REPO_ROOT / "experiments/origin_detection/phase_statistics"):
        files.extend(directory.glob("*.py"))
        files.append(directory / "README.md")
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(files)}


def extract_inventory(rows, output):
    records, start = [], perf_counter()
    for index, row in enumerate(rows, 1):
        path = image_path(row)
        if file_sha256(path) != row["sha256"]:
            raise ValueError("Phase input file SHA changed")
        with Image.open(path) as im:
            if im.size != (int(row["width"]), int(row["height"])):
                raise ValueError("Phase input dimensions changed")
            rgb = np.asarray(im.convert("RGB"))
        for variant in VARIANTS:
            if variant == "raw":
                image = rgb
            else:
                stream = BytesIO()
                Image.fromarray(bounded_rgb(rgb, 512)).save(stream, format="JPEG", quality=90, subsampling=0)
                stream.seek(0)
                with Image.open(stream) as decoded:
                    image = np.asarray(decoded.convert("RGB"))
            result = extract_features(image)
            records.append({**row, "variant": variant, **dict(zip(FEATURE_NAMES, result.values.tolist())),
                            "patches": result.patches, "preprocess_ms": result.preprocess_ms, "statistics_ms": result.statistics_ms})
        if index % 60 == 0 or index == len(rows):
            print(json.dumps({"images": index, "total": len(rows), "elapsed_s": round(perf_counter() - start, 3)}), flush=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return perf_counter() - start


def signed_features():
    path = OUTPUT / "features.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt["code_pins"] != code_pins() or receipt["parent_receipt_sha256"] != PARENT_SHA:
        raise ValueError("Frozen phase code/parent lineage changed")
    if file_sha256(OUTPUT / "features.csv") != receipt["csv_sha256"]:
        raise ValueError("Frozen phase cache changed")
    inventory, sha = inherited_inventory()
    if sha != receipt["inventory_sha256"]:
        raise ValueError("Inherited phase inventory changed")
    rows = validate_feature_cache(read_rows(OUTPUT / "features.csv"), inventory, FEATURE_NAMES, variants=VARIANTS)
    return rows, receipt


def benchmark(rows, rules):
    prior = WORK_DIR / "robust_statistics/rr_first_iteration"
    receipt = json.loads((prior / "benchmark.json").read_text(encoding="utf-8"))
    if file_sha256(prior / "benchmark.csv") != receipt["csv_sha256"]:
        raise ValueError("Prior benchmark filenames changed")
    names = sorted({"rr/" + r["filename"] for r in read_rows(prior / "benchmark.csv")})
    by_name = {r["filename"]: r for r in rows if r["variant"] == "raw"}
    selected = [by_name[n] for n in names]
    if len(selected) != 60:
        raise ValueError("Phase benchmark denominator changed")
    output = {}
    for mode, rule in rules.items():
        detector = PhaseDetector(rule)
        detector.predict(np.full((256, 256, 3), 128, np.uint8))
        items = [{"filename": r["filename"], "path": image_path(r), "sha256": r["sha256"],
                  "expected_score": float(rule.score([float(r[n]) for n in FEATURE_NAMES]))} for r in selected]
        output[mode] = benchmark_files(detector, items)
    return {"methods": output, "processor": platform.processor(), "python_version": platform.python_version(),
            "opencv_threads": cv2.getNumThreads(), "opencv_version": cv2.__version__,
            "prior_selection_csv_sha256": receipt["csv_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--pilot", type=int, default=0)
    group.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.pilot < 0:
        parser.error("pilot must be positive")
    cv2.setNumThreads(1)
    with CONFIG.open("rb") as stream:
        config = tomllib.load(stream)
    if args.prepare or args.pilot:
        rows, inventory_sha = inherited_inventory()
        directory = OUTPUT if args.prepare else OUTPUT.with_name("joint_phase_pilot")
        if directory.exists():
            raise FileExistsError("Preserve phase results, output exists")
        if args.pilot:
            random.Random(config["seed"]).shuffle(rows)
            rows = rows[:args.pilot]
        directory.mkdir(parents=True)
        elapsed = extract_inventory(rows, directory / "features.csv")
        validate_feature_cache(read_rows(directory / "features.csv"), rows, FEATURE_NAMES, variants=VARIANTS)
        write_json(directory / "features.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "images": len(rows),
                                                  "records": len(rows) * 2, "elapsed_s": elapsed, "code_pins": code_pins(),
                                                  "parent_receipt_sha256": PARENT_SHA, "inventory_sha256": inventory_sha,
                                                  "csv_sha256": file_sha256(directory / "features.csv")})
        return
    rows, receipt = signed_features()
    if args.benchmark:
        path = OUTPUT / "benchmark.json"
        if path.exists():
            raise FileExistsError("Phase benchmark already exists")
        iteration = json.loads((OUTPUT / "iteration.json").read_text(encoding="utf-8"))
        if iteration["features_receipt_sha256"] != file_sha256(OUTPUT / "features.json"):
            raise ValueError("Phase iteration cache link changed")
        rules = {}
        for mode, sha in iteration["rule_files"].items():
            rule_path = OUTPUT / f"{mode}_rule.json"
            if file_sha256(rule_path) != sha:
                raise ValueError("Frozen phase rule changed")
            rules[mode] = PhaseRule.load(rule_path)
        result = benchmark(rows, rules)
    else:
        path = OUTPUT / "iteration.json"
        result, rules = run_screen(rows, config, receipt["inventory_sha256"])
        json.dumps(result, allow_nan=False)
        if path.exists():
            raise FileExistsError("Registered phase screening exists")
        for mode, rule in rules.items():
            rule.save(OUTPUT / f"{mode}_rule.json")
        result["rule_files"] = {m: file_sha256(OUTPUT / f"{m}_rule.json") for m in rules}
    if path.exists():
        raise FileExistsError("Preserve phase evaluation")
    result["features_receipt_sha256"] = file_sha256(OUTPUT / "features.json")
    result["code_pins"] = code_pins()
    write_json(path, result)
    print(json.dumps({"output": path.name, "chosen": result.get("chosen")}))


if __name__ == "__main__":
    main()
