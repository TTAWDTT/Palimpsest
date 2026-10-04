import csv
import json

import pytest

from palimpsest.data.inference import validated_inference


def files(tmp_path, predictions):
    manifest = tmp_path / "manifest.csv"
    inference = tmp_path / "predictions.csv"
    summary = tmp_path / "summary.json"
    expected = [
        {"filename": "a.png", "src": "a", "label": "REAL", "condition": "original"},
        {"filename": "b.png", "src": "b", "label": "FAKE", "condition": "transfer"},
    ]
    for path, rows in ((manifest, expected), (inference, predictions)):
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary.write_text(
        json.dumps(
            {
                "completed_images": 2,
                "remaining_images": 0,
                "errors": 0,
                "stopping_error": None,
            }
        ),
        encoding="utf-8",
    )
    return manifest, inference, summary


def predictions():
    return [
        {"filename": "a.png", "src": "a", "label": "REAL", "score": "-1", "error": ""},
        {"filename": "b.png", "src": "b", "label": "FAKE", "score": "1", "error": ""},
    ]


def test_manifest_groups_valid_predictions(tmp_path):
    grouped = validated_inference(*files(tmp_path, predictions()), 2)
    assert grouped["transfer"][0]["src"] == "b"


def test_matching_row_count_cannot_hide_reordered_or_missing_sources(tmp_path):
    with pytest.raises(RuntimeError, match="identity/order"):
        validated_inference(*files(tmp_path, list(reversed(predictions()))), 2)


def test_finite_score_gate_is_preserved(tmp_path):
    rows = predictions()
    rows[1]["score"] = "nan"
    with pytest.raises(RuntimeError, match="nonfinite"):
        validated_inference(*files(tmp_path, rows), 2)
