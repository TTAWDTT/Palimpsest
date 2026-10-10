"""Source identities, forced denominators and paired resampling semantics."""

import csv
import importlib.util
from pathlib import Path

import pytest

_path = (
    Path(__file__).resolve().parents[2]
    / "experiments/origin_detection/marginal_intermittency/evaluate_manifest.py"
)
_spec = importlib.util.spec_from_file_location("m_manifest_evaluator", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
evaluate, metrics = _module.evaluate, _module.metrics
read_table, source_groups = _module.read_table, _module.source_groups


def record(source, label, score, *, condition="original", category="cat", valid=True):
    return {
        "source_id": source,
        "label": label,
        "category": category,
        "condition": condition,
        "M": score,
        "BFree": score,
        "valid": valid,
        "image_sha256": "a" * 64,
    }


def write_scores(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({**row, "valid": int(row["valid"])} for row in rows)


@pytest.mark.parametrize(
    "bad",
    [
        [record("a", 0, -1), record("a", 0, 1)],
        [record("a", 0, -1), record("a", 1, 1, condition="screen")],
        [record("a", 0, -1), record("a", 0, 1, condition="screen", category="horse")],
    ],
)
def test_manifest_rejects_duplicate_or_conflicting_source_identity(tmp_path, bad):
    path = tmp_path / "scores.csv"
    write_scores(path, bad)
    with pytest.raises(ValueError, match="Duplicate|Conflicting"):
        read_table(path, scores=True)


def test_zero_score_policy_and_invalid_rows_keep_full_denominator():
    rows = [record("real", 0, 0, valid=False), record("ai", 1, 1)]
    actual, baseline = metrics(rows), metrics(rows, "BFree")
    assert actual["accuracy"] == 0.5
    assert baseline["accuracy"] == 1
    assert actual["invalid_images"] == 1
    assert actual["qc_valid_fraction"] == 0.5
    assert actual["images"] == 2


def test_macro_auc_and_bootstrap_carry_all_source_views_together():
    rows = [
        record("r", 0, -1),
        record("a", 1, 1),
        record("r", 0, 1, condition="screen"),
        record("a", 1, -1, condition="screen"),
    ]
    strata = source_groups(rows)
    assert sorted(len(unit) for units in strata.values() for unit in units) == [2, 2]
    result = evaluate(rows, bootstrap=20, seed=7)
    overall = result["metrics"]["M"]["all_views"]
    assert overall["sources"] == 2 and overall["images"] == 4
    assert overall["pooled_auc"] == overall["category_macro_auc"] == 0.5
    assert result["metrics"]["M"]["original"]["pooled_auc"] == 1
    assert result["metrics"]["M"]["screen"]["pooled_auc"] == 0
    assert result["all_views_ci95"]["M"]["accuracy"] == [0.5, 0.5]
    assert result["paired_M_minus_BFree"]["accuracy"]["ci95"] == [0, 0]


def test_nonfinite_scores_and_invalid_label_rejected(tmp_path):
    path = tmp_path / "scores.csv"
    for row, error in [
        (record("a", 1, float("nan")), "Nonfinite"),
        (record("a", 2, 1), "identity/label"),
    ]:
        write_scores(path, [row])
        with pytest.raises(ValueError, match=error):
            read_table(path, scores=True)
