"""Compare four baselines on one prespecified 600-source RRDataset pilot."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from evaluate_rr_bfree import paired_change
from score_published_logits import evaluate


PILOT_FILES = {
    "Fourier-5NN": Path("work/fourier_knn_rr_pilot.csv"),
    "Benford-RF": Path("work/benford_rr_pilot.csv"),
    "D3": Path("work/d3_rr_pilot.csv"),
}
BFREE_FILE = Path("work/rr_bfree_complete.csv")
OUTPUT = Path("work/matched_rr_pilot_evaluation.json")


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]], method: str) -> dict[str, object]:
    conditions: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        conditions[row["filename"].split("/", 1)[0]].append(row)
    if {condition: len(group) for condition, group in conditions.items()} != {
        "original": 600, "transfer": 600, "redigital": 600
    }:
        raise ValueError(f"Missing paired pilot observations for {method}")
    result = {
        condition: evaluate([
            {"src": row["src"], "label": row["label"], method: row["score"]}
            for row in group
        ], method)
        for condition, group in conditions.items()
    }
    source_rows = {
        condition: {row["src"]: row for row in group}
        for condition, group in conditions.items()
    }
    result["paired_changes"] = {
        condition: paired_change(source_rows["original"], source_rows[condition])
        for condition in ("transfer", "redigital")
    }
    return result


def paired_method_difference(
    first: list[dict[str, str]], second: list[dict[str, str]], condition: str
) -> dict[str, object]:
    first_rows = {row["src"]: row for row in first if row["filename"].startswith(condition + "/")}
    second_rows = {row["src"]: row for row in second if row["filename"].startswith(condition + "/")}
    if first_rows.keys() != second_rows.keys() or len(first_rows) != 600:
        raise ValueError("Method comparison must use 600 matched sources")
    by_class: dict[str, list[int]] = {"REAL": [], "FAKE": []}
    for source, a in first_rows.items():
        b = second_rows[source]
        if (a["label"], a["filename"]) != (b["label"], b["filename"]):
            raise ValueError("Mismatched paired method metadata")
        truth = a["label"] == "FAKE"
        by_class[a["label"]].append(
            int((float(a["score"]) > 0) == truth)
            - int((float(b["score"]) > 0) == truth)
        )
    difference = sum(sum(values) / len(values) for values in by_class.values()) / 2
    rng = random.Random(20260924)
    bootstrap = sorted(
        sum(sum(rng.choices(values, k=len(values))) / len(values)
            for values in by_class.values()) / 2
        for _ in range(2000)
    )
    return {"balanced_accuracy_difference": difference,
            "source_bootstrap_ci95": [bootstrap[49], bootstrap[1950]]}


def main() -> None:
    pilot = {name: read(path) for name, path in PILOT_FILES.items()}
    filenames = {row["filename"] for row in pilot["Fourier-5NN"]}
    if len(filenames) != 1800:
        raise ValueError("Pilot file has duplicate or missing filenames")
    for name, rows in pilot.items():
        if len(rows) != len(filenames) or {row["filename"] for row in rows} != filenames:
            raise ValueError(f"{name} was not evaluated on the matched pilot")
    bfree_rows = [row for row in read(BFREE_FILE) if row["filename"] in filenames]
    if len(bfree_rows) != len(filenames):
        raise ValueError("B-Free full run does not cover the pilot")
    pilot["B-Free"] = bfree_rows
    for name, rows in pilot.items():
        labels = {row["filename"]: (row["src"], row["label"]) for row in rows}
        reference = {row["filename"]: (row["src"], row["label"]) for row in bfree_rows}
        if labels != reference:
            raise ValueError(f"{name} has mismatched label or source ID")
    result = {
        "sample": "SHA-256 prespecified 300 AI + 300 real source IDs, all 3 RR conditions",
        "scope": "pilot, not full test",
        "methods": {name: summarize(rows, name) for name, rows in pilot.items()},
        "paired_method_comparisons": {
            f"{first}_minus_{second}": {
                condition: paired_method_difference(pilot[first], pilot[second], condition)
                for condition in ("original", "transfer", "redigital")
            }
            for first, second in (("D3", "B-Free"), ("Benford-RF", "Fourier-5NN"))
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, conditions in result["methods"].items():
        print(name, {condition: round(conditions[condition]["balanced_accuracy_at_zero"], 4)
                     for condition in ("original", "transfer", "redigital")})


if __name__ == "__main__":
    main()
