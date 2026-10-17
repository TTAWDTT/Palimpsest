"""Evaluate frozen M on a manifest or existing, source-identified scores.

No fitting or candidate selection occurs. Invalid/QC-degenerate images retain
their forced decisions in every accuracy denominator. Bootstrap draws sources
within category x label strata, carrying every view of a source together.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from palimpsest.evaluation.classification import auc

IDENTITY = ("source_id", "category", "label", "condition")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: Path, *, scores: bool) -> list[dict]:
    """Reject ambiguous identities and incomplete records before inference."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(
            set(reader.fieldnames)
        ):
            raise ValueError("Missing or duplicate CSV columns")
        required = set(IDENTITY) | (
            {"M", "image_sha256", "valid"} if scores else {"path"}
        )
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(
                f"Missing columns: {sorted(required - set(reader.fieldnames or ()))}"
            )
        records = list(reader)
    if not records:
        raise ValueError("Input table is empty")
    seen, sources, output = set(), {}, []
    for line, raw in enumerate(records, 2):
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError(f"Malformed CSV row {line}")
        row = {key: value.strip() for key, value in raw.items()}
        if (
            any(not row[key] for key in IDENTITY)
            or row["label"] not in {"0", "1"}
            or row["condition"] == "all_views"
        ):
            raise ValueError(f"Invalid identity/label at row {line}")
        row["label"] = int(row["label"])
        key = row["source_id"], row["condition"]
        if key in seen:
            raise ValueError(f"Duplicate source/condition: {key}")
        seen.add(key)
        identity = row["category"], row["label"]
        if row["source_id"] in sources and sources[row["source_id"]] != identity:
            raise ValueError(
                f"Conflicting category/label for source {row['source_id']}"
            )
        sources[row["source_id"]] = identity
        fingerprint = row.get("image_sha256", "").lower()
        if (scores or fingerprint) and (
            len(fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in fingerprint)
        ):
            raise ValueError(f"Invalid image_sha256 at row {line}")
        row["image_sha256"] = fingerprint
        if scores:
            if row["valid"] not in {"0", "1"}:
                raise ValueError(f"valid must be 0 or 1 at row {line}")
            row["valid"] = bool(int(row["valid"]))
        elif not row["path"]:
            raise ValueError(f"Empty image path at row {line}")
        for name in ("M", "BFree") if scores else ("BFree",):
            if name in row:
                row[name] = float(row[name])
                if not math.isfinite(row[name]):
                    raise ValueError(f"Nonfinite {name} at row {line}")
        output.append(row)
    return output


def infer(rows: list[dict], image_root: Path) -> list[dict]:
    """Validate every input file/hash first; then run the frozen pixel detector."""
    from PIL import Image
    from palimpsest.detection.algorithms.marginal_intermittency import (
        MIntermittencyDetector,
    )

    inputs = []
    for row in rows:
        path = Path(row["path"])
        path = path if path.is_absolute() else image_root / path
        if not path.is_file():
            raise ValueError(f"Image not found: {path}")
        fingerprint = file_sha256(path)
        if row["image_sha256"] and fingerprint != row["image_sha256"]:
            raise ValueError(f"Image hash mismatch: {path}")
        inputs.append((path, fingerprint))
    detector, output = MIntermittencyDetector(), []
    for row, (path, fingerprint) in zip(rows, inputs):
        with Image.open(path) as loaded:
            result = detector.analyze(np.asarray(loaded.convert("RGB")))
        output.append(
            {
                **row,
                "image_sha256": fingerprint,
                "M": result["score"],
                "valid": not result["qc"]["degenerate"],
            }
        )
    return output


def metrics(rows: list[dict], score: str = "M") -> dict:
    """Ordinary row-count metrics; M ties follow the frozen >=0 convention."""
    correct, counts = [0, 0], [0, 0]
    by_category = defaultdict(list)
    for row in rows:
        label = row["label"]
        decision = row[score] >= 0 if score == "M" else row[score] > 0
        correct[label] += int(decision == bool(label))
        counts[label] += 1
        by_category[row["category"]].append((row[score], label))
    class_accuracy = [correct[i] / counts[i] if counts[i] else None for i in (0, 1)]
    category_auc = {
        cat: auc(pairs) if {label for _, label in pairs} == {0, 1} else None
        for cat, pairs in sorted(by_category.items())
    }
    return {
        "images": len(rows),
        "sources": len({r["source_id"] for r in rows}),
        "real_images": counts[0],
        "ai_images": counts[1],
        "correct": sum(correct),
        "accuracy": sum(correct) / len(rows),
        "balanced_accuracy": sum(class_accuracy) / 2 if all(counts) else None,
        "real_specificity": class_accuracy[0],
        "ai_recall": class_accuracy[1],
        "pooled_auc": auc([(r[score], r["label"]) for r in rows])
        if all(counts)
        else None,
        "category_macro_auc": float(np.mean(list(category_auc.values())))
        if all(v is not None for v in category_auc.values())
        else None,
        "category_auc": category_auc,
        "invalid_images": sum(not r["valid"] for r in rows),
        "qc_valid_fraction": sum(r["valid"] for r in rows) / len(rows),
    }


def source_groups(rows: list[dict]) -> dict[tuple[str, int], list[list[dict]]]:
    """A resampling unit contains all conditions from one original source."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_id"]].append(row)
    strata = defaultdict(list)
    for group in grouped.values():
        strata[group[0]["category"], group[0]["label"]].append(group)
    return dict(strata)


def evaluate(rows: list[dict], *, bootstrap: int = 1000, seed: int = 20261011) -> dict:
    if bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    methods = ("M", "BFree") if "BFree" in rows[0] else ("M",)
    groups = {"all_views": rows}
    for condition in sorted({r["condition"] for r in rows}):
        groups[condition] = [r for r in rows if r["condition"] == condition]
    point = {
        method: {name: metrics(group, method) for name, group in groups.items()}
        for method in methods
    }
    keys = ("accuracy", "balanced_accuracy", "category_macro_auc")
    draws = {method: {key: [] for key in keys} for method in methods}
    differences = {key: [] for key in keys}
    rng, strata = np.random.default_rng(seed), source_groups(rows)
    for _ in range(bootstrap):
        sample = []
        for stratum in sorted(strata):
            units = strata[stratum]
            for index in rng.integers(len(units), size=len(units)):
                sample.extend(units[int(index)])
        values = {method: metrics(sample, method) for method in methods}
        for key in keys:
            for method in methods:
                if values[method][key] is not None:
                    draws[method][key].append(values[method][key])
            if "BFree" in methods and values["M"][key] is not None:
                differences[key].append(values["M"][key] - values["BFree"][key])

    def interval(values):
        return (
            [float(v) for v in np.quantile(values, (0.025, 0.975))] if values else None
        )

    return {
        "metrics": point,
        "all_views_ci95": {
            method: {key: interval(v) for key, v in values.items()}
            for method, values in draws.items()
        },
        "paired_M_minus_BFree": {
            key: {
                "estimate": point["M"]["all_views"][key]
                - point["BFree"]["all_views"][key]
                if point["M"]["all_views"][key] is not None
                else None,
                "ci95": interval(differences[key]),
            }
            for key in keys
        }
        if "BFree" in methods
        else None,
        "bootstrap": {
            "iterations": bootstrap,
            "seed": seed,
            "unit": "source; all views together; category x label strata",
        },
        "decision_rule": {"M": "score >= 0", "BFree": "score > 0"},
        "denominator": "all rows, including QC-degenerate forced predictions",
        "scope": "fixed-formula evaluation; not evidence of independent data or physical-channel coverage",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--scores", type=Path)
    parser.add_argument(
        "--image-root", type=Path, help="Relative image paths; default manifest parent"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New output directory; refuses overwrite",
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.bootstrap < 0 or (args.scores and args.image_root):
        parser.error(
            "bootstrap must be nonnegative; image-root applies only to manifest inference"
        )
    input_path = args.scores or args.manifest
    rows = read_table(input_path, scores=bool(args.scores))
    if args.manifest:
        rows = infer(rows, args.image_root or args.manifest.parent)
    report = evaluate(rows, bootstrap=args.bootstrap, seed=args.seed)
    report["input"] = {
        "mode": "scores" if args.scores else "fresh_inference",
        "sha256": file_sha256(input_path),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    fields = [*IDENTITY, "image_sha256", "M", "valid"]
    if "BFree" in rows[0]:
        fields.append("BFree")
    with (args.output / "scores.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**row, "valid": int(row["valid"])} for row in rows)
    with (args.output / "metrics.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
