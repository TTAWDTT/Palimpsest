import csv

import pytest

from palimpsest.contracts import Origin, Prediction
from palimpsest.evaluation.cached import (
    CachedRun,
    ExpectedImage,
    evaluate_cached_method,
)
from palimpsest.evaluation.channel_fidelity import paired_score_error_gain


def fixture_inputs(tmp_path):
    path = tmp_path / "scores.csv"
    rows, expected = [], []
    for condition in ("original", "transfer"):
        for source, label, score in (("r", Origin.NATURAL, 0.5), ("a", Origin.AI, 0.9)):
            filename = f"{condition}/{source}.jpg"
            expected.append(ExpectedImage(source, condition, label, filename))
            rows.append(
                {
                    "filename": filename,
                    "src": source,
                    "label": "REAL" if label == Origin.NATURAL else "FAKE",
                    "score": score,
                    "end_to_end_ms": 10,
                    "error": "",
                }
            )

    def write():
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    write()
    return path, rows, expected, write


def test_cached_scores_use_explicit_threshold_and_independent_inventory(tmp_path):
    path, _, expected, _ = fixture_inputs(tmp_path)
    report, predictions = evaluate_cached_method(
        expected,
        [
            CachedRun(
                path, condition_map={"original": "original", "transfer": "transfer"}
            )
        ],
        method="fixed",
        threshold=0.5,
    )
    assert report["conditions"]["original"]["balanced_accuracy_at_zero"] == 1
    assert predictions["original"]["r"].origin == Origin.NATURAL
    assert report["paired_changes"]["transfer"]["balanced_accuracy_change"] == 0
    assert report["timing_ms"]["all"]["end_to_end_ms"]["p95"] == 10


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate",
        "missing",
        "filename",
        "label",
        "nan",
        "timing",
        "error",
        "condition",
    ],
)
def test_invalid_cache_never_silently_passes(tmp_path, defect):
    path, rows, expected, write = fixture_inputs(tmp_path)
    if defect == "duplicate":
        rows.append(dict(rows[0]))
    elif defect == "missing":
        rows.pop()
    elif defect == "filename":
        rows[0]["filename"] = "original/wrong.jpg"
    elif defect == "label":
        rows[0]["label"] = "FAKE"
    elif defect == "nan":
        rows[0]["score"] = "nan"
    elif defect == "timing":
        rows[0]["end_to_end_ms"] = -1
    elif defect == "error":
        rows[0]["error"] = "decode failed"
    else:
        rows[0]["filename"] = "unexpected/r.jpg"
    write()
    with pytest.raises(ValueError):
        evaluate_cached_method(
            expected,
            [
                CachedRun(
                    path, condition_map={"original": "original", "transfer": "transfer"}
                )
            ],
            method="fixed",
        )


def test_fingerprint_and_overlapping_runs_fail(tmp_path):
    path, _, expected, _ = fixture_inputs(tmp_path)
    with pytest.raises(ValueError, match="fingerprint"):
        evaluate_cached_method(
            expected,
            [CachedRun(path, fixed_condition="original", expected_sha256="wrong")],
            method="fixed",
        )
    run = CachedRun(
        path, condition_map={"original": "original", "transfer": "transfer"}
    )
    with pytest.raises(ValueError, match="Multiple runs"):
        evaluate_cached_method(expected, [run, run], method="fixed")


def test_bootstrap_preserves_pairing_and_error_gain_direction():
    labels = {"r": Origin.NATURAL, "a": Origin.AI}
    real = {"r": Prediction("fixed", -2), "a": Prediction("fixed", 2)}
    comparator = {"r": Prediction("fixed", -1), "a": Prediction("fixed", 1)}
    result = paired_score_error_gain(
        real, real, comparator, labels, seed=7, replicates=100
    )
    assert result["class_balanced_mae_gain"] == 1
    assert result["ci95"] == [1, 1]
    with pytest.raises(ValueError, match="identical"):
        paired_score_error_gain(real, real, {"r": comparator["r"]}, labels, seed=7)
