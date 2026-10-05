"""Pre-registered independent axis plants and destructive controls."""

from dataclasses import replace
import csv

import numpy as np
from PIL import Image
import pytest

from palimpsest.detection.algorithms.local_statistics.features import FEATURE_NAMES
from palimpsest.detection.algorithms.local_statistics.projection import (
    PairedProjectionDetector, ProjectionRule, fit_projection,
)
from palimpsest.detection.files import predict_file
from palimpsest.io.tables import read_rows
from experiments.origin_detection.paired_projection.rr.run_iteration import paired_matrices, screen_iteration
from experiments.origin_detection.robust_statistics.rr.evaluate_features import validate_feature_rows


def axis_fixture(tmp_path, *, overlap=False, null=False, confounded=False):
    # By construction: class separation on axis 0; fixed processing on axis 1.
    # Overlap counterexample puts both on axis 0. No image extractor is used
    # to author expected answers, since this iteration changes only the rule.
    inventory, rows = [], []
    for role in ("fit", "selection", "threshold"):
        for label in (0, 1):
            for index in range(4):
                source = f"{role}/{label}/{index}"
                for condition, shift in (("original", 0), ("transfer", 4), ("redigital", 7)):
                    values = np.zeros(17)
                    if not null:
                        values[0] = label * 2 - 1
                        if confounded:
                            values[1] = 4 * (label * 2 - 1) if role == "fit" else (20 if index % 2 else -20)
                        values[0 if overlap else 1] += shift
                    filename = f"{condition}/{source}.png"
                    inventory.append({"filename": filename, "source_group": source, "condition": condition,
                                      "algorithm_role": role, "label": "ai" if label else "real"})
                    for variant in ("raw", "jpeg90_444_after_resize"):
                        rows.append({"filename": filename, "src": source, "condition": condition,
                                     "role": role, "label": "FAKE" if label else "REAL",
                                     "long_edge": 512, "variant": variant,
                                     **dict(zip(FEATURE_NAMES, values))})
    path = tmp_path / "features.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    config = {"conditions": ["original", "transfer", "redigital"], "long_edges": [512],
              "long_edge": 512, "removed_ranks": [0, 1, 3], "z_clip": 5.0, "seed": 42,
              "bootstrap_repetitions": 20, "selection_auc_lower_gate": 0.5,
              "minimum_worst_auc_gain": 0.02, "minimum_processed_class_accuracy": 0.55}
    return inventory, read_rows(path), config


def test_gate_recovers_planted_nuisance_confound_and_refuses_bad_gain(tmp_path):
    inventory, rows, config = axis_fixture(tmp_path, confounded=True)
    table = validate_feature_rows(rows, inventory, config)
    result, _ = screen_iteration(table, config, "constructed-manifest", 0.5)
    assert result["chosen"] == "rank1"
    assert result["candidates"]["rank0"]["worst_processed_auc"] < 1
    assert result["candidates"]["rank1"]["selection_gate"]
    blocked, _ = screen_iteration(table, {**config, "minimum_worst_auc_gain": 1.0}, "constructed-manifest", 0.5)
    assert blocked["chosen"] is None  # deliberately impossible gain must fire


def test_full_csv_path_recovers_independent_axes(tmp_path):
    inventory, rows, config = axis_fixture(tmp_path)
    table = validate_feature_rows(rows, inventory, config)
    result, rules = screen_iteration(table, config, "constructed-manifest", 0.5)
    projected = result["candidates"]["rank1"]
    assert projected["fit"]["effective_removed_rank"] == 1
    assert projected["fit"]["removed_difference_energy_fraction"] == pytest.approx(1.0)
    assert projected["fit"]["retained_contrast_norm_fraction"] == pytest.approx(1.0)
    for condition in ("transfer", "redigital"):
        assert projected["selection_auc"][condition]["auc"] == 1
        assert projected["selection_classification"]["conditions"][condition]["balanced_accuracy_at_zero"] == 1
    np.testing.assert_allclose(rules["rank1"].weights, [1] + [0] * 16, atol=1e-12)
    assert all(projected["selection_score_drift"][c][l]["mean_score_change"] == 0
               for c in ("transfer", "redigital") for l in ("natural", "ai"))
    arrays, labels = paired_matrices(table, "fit", config)
    # Deliberate break: processing on the class axis must remove the class signal.
    broken = arrays.copy()
    broken[1:, :, 0] += broken[1:, :, 1]
    broken[:, :, 1] = 0
    broken_rule, diagnostics = fit_projection(broken, labels, 1)
    assert diagnostics["zero_rule"] and np.all(broken_rule.score(broken) == 0)
    # The unprojected control has not removed its paired score shift when its
    # class axis overlaps processing; this establishes the control can fail.
    unprojected, _ = fit_projection(broken, labels, 0)
    assert np.any(unprojected.score(broken[1]) != unprojected.score(broken[0]))


@pytest.mark.parametrize("null", [False, True])
def test_overlapping_signal_and_zero_null_are_failures(tmp_path, null):
    inventory, rows, config = axis_fixture(tmp_path, overlap=True, null=null)
    table = validate_feature_rows(rows, inventory, config)
    result, _ = screen_iteration(table, config, "constructed-manifest", 0.5)
    assert result["chosen"] is None
    for rank in (1, 3):
        candidate = result["candidates"][f"rank{rank}"]
        assert candidate["fit"]["zero_rule"]
        assert candidate["worst_processed_auc"] == 0.5


@pytest.mark.parametrize("corruption", ["pair_source", "pair_label", "nan"])
def test_alignment_and_nonfinite_refusal(tmp_path, corruption):
    inventory, rows, config = axis_fixture(tmp_path)
    table = validate_feature_rows(rows, inventory, config)
    paired_matrices(table, "fit", config)  # clean twin
    if corruption == "nan":
        arrays, labels = paired_matrices(table, "fit", config)
        arrays[0, 0, 0] = np.nan
        with pytest.raises(ValueError):
            fit_projection(arrays, labels, 1)
    else:
        row = table[512, "raw", "fit", "transfer"][0][0]
        row["src" if corruption == "pair_source" else "label"] = "wrong" if corruption == "pair_source" else "FAKE"
        # First fit row belongs to class 0 by construction.
        with pytest.raises(ValueError):
            paired_matrices(table, "fit", config)


def test_rule_roundtrip_and_file_prediction(tmp_path):
    weights = (1.0,) + (0.0,) * 16
    rule = ProjectionRule((0.0,) * 17, (1.0,) * 17, weights, (), 0)
    path = tmp_path / "rule.json"
    rule.save(path)
    restored = ProjectionRule.load(path)
    assert rule.fingerprint == restored.fingerprint
    image_path = tmp_path / "image.png"
    Image.fromarray(np.full((32, 32, 3), 128, np.uint8)).save(image_path)
    prediction = predict_file(PairedProjectionDetector(restored), image_path)
    assert prediction.prediction.score == 0
    assert prediction.prediction.origin.value == "natural"
    with pytest.raises(ValueError):
        replace(restored, scale=(0.0,) * 17)
    with pytest.raises(ValueError):
        replace(restored, removed_rank=1, basis=(weights,))
    with pytest.raises(ValueError):
        replace(restored, weights=(2.0,) + (0.0,) * 16)
    with pytest.raises(ValueError):
        replace(restored, removed_rank=1, basis=((0.0,) * 34,))
