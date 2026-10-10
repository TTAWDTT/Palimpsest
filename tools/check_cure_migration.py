"""Verify direct relocation and optionally replay signed development features.

This is a software migration check, not new scientific or external validation.
No image decoding, neural inference, fitting or threshold selection is performed.
"""

import argparse
import ast
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from palimpsest.detection.models.frozen_features.cure.quantile_score_consistency import (
    QuantileScoreRule,
)
from palimpsest.io.hashing import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def implementation(tree):
    return ast.dump(
        ast.Module(
            body=[
                node
                for node in tree.body
                if not isinstance(node, (ast.Import, ast.ImportFrom))
                and not (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                )
            ],
            type_ignores=[],
        ),
        include_attributes=False,
    )


def audit_layout():
    registry = json.loads(
        (ROOT / "docs/maintenance/cure_method_migration.json").read_text(
            encoding="utf-8"
        )
    )
    commit = registry["source_commit"]
    checked = []
    for old, new in registry["moves"].items():
        if (ROOT / old).exists() or not (ROOT / new).is_file():
            raise ValueError(f"Old forwarding path or missing target: {old}")
        if not old.startswith("src/") or not old.endswith(".py"):
            continue
        before = subprocess.check_output(
            ["git", "show", f"{commit}:{old}"], cwd=ROOT
        ).decode("utf-8")
        after = (ROOT / new).read_text(encoding="utf-8")
        if implementation(ast.parse(before)) != implementation(ast.parse(after)):
            raise ValueError(f"Moved numerical implementation changed: {new}")
        checked.append(new)
    old = "src/palimpsest/detection/algorithms/paired_stability.py"
    original = ast.parse(
        subprocess.check_output(["git", "show", f"{commit}:{old}"], cwd=ROOT).decode(
            "utf-8"
        )
    )
    destinations = registry["extraction"][old]
    for name, destination in (
        ("StableRule", destinations[0]),
        ("fit_stable_rule", destinations[0]),
        ("StableDetector", destinations[1]),
    ):
        a = next(
            n
            for n in original.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name
        )
        current = ast.parse((ROOT / destination).read_text(encoding="utf-8"))
        b = next(
            n
            for n in current.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name
        )
        if ast.dump(a, include_attributes=False) != ast.dump(
            b, include_attributes=False
        ):
            raise ValueError(f"Extracted implementation changed: {name}")
    return {
        "numerical_modules_identical_except_imports": len(checked),
        "extracted_definitions_identical": 3,
        "direct_moves": len(registry["moves"]),
        "old_forwarding_files": 0,
        "historical_commit": commit,
    }


def replay_cached(cache, rules):
    receipt = json.loads((cache / "features.json").read_text(encoding="utf-8"))
    if (
        not receipt["passed"]
        or receipt["records"] != 20160
        or receipt["dimensions"] != 3920
    ):
        raise ValueError("Unexpected signed descriptor receipt")
    expected_path = rules / "selection_scores.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))["selected"]
    key = lambda r: tuple(r[n] for n in ("domain", "src", "condition", "variant"))
    lookup = {key(row): row for row in expected}
    if len(expected) != 5040 or len(lookup) != 5040:
        raise ValueError("Saved selected-score identity coverage differs")
    rule_path = rules / "selected_rule.json"
    rule = QuantileScoreRule.load(rule_path)
    restored = QuantileScoreRule.from_payload(rule.to_payload())
    seen = set()
    maximum_error = 0.0
    for part in ("base", "q60"):
        folder = cache / part
        for filename, field in (
            ("vectors.npy", "vectors_sha256"),
            ("metadata.csv", "metadata_sha256"),
        ):
            if file_sha256(folder / filename) != receipt["parts"][part][field]:
                raise ValueError(f"Signed cache changed: {part}/{filename}")
        with (folder / "metadata.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        vectors = np.load(folder / "vectors.npy", mmap_mode="r")
        if vectors.shape != (len(rows), 3920):
            raise ValueError("Cache matrix dimensions differ")
        indices = [i for i, row in enumerate(rows) if row["role"] == "selection"]
        values = np.asarray(vectors[indices], dtype=float)
        scores = rule.score(values) - rule.threshold
        if not np.array_equal(scores, restored.score(values) - restored.threshold):
            raise ValueError("Rule JSON round trip changes scores")
        for i, score in zip(indices, scores):
            row = rows[i]
            identity = key(row)
            if identity in seen or identity not in lookup:
                raise ValueError("Duplicate or unrecognized selection query")
            before = lookup[identity]
            if row["label"] != before["label"] or before["role"] != "selection":
                raise ValueError("Selection identity or label differs")
            error = abs(float(score) - before["score"])
            maximum_error = max(maximum_error, error)
            if error != 0:
                raise ValueError(
                    "Migrated selected margin differs from historical score"
                )
            seen.add(identity)
    if seen != set(lookup):
        raise ValueError("Missing selection query")
    return {
        "queries": len(seen),
        "maximum_absolute_margin_difference": maximum_error,
        "changed_decisions": 0,
        "rule_json_roundtrip_exact": True,
        "saved_scores_sha256": file_sha256(expected_path),
        "rule_sha256": file_sha256(rule_path),
        "scope": "Cached development software replay; no new independent data or image inference",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--rules", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.cache) != bool(args.rules):
        parser.error("--cache and --rules must be provided together")
    result = {"passed": True, "layout": audit_layout()}
    if args.cache:
        result["replay"] = replay_cached(args.cache, args.rules)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        if args.output.exists():
            raise ValueError("Refuse to overwrite a migration receipt")
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
