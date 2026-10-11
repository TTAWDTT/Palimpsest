"""Known-answer controls registered before RR candidate statistics are viewed."""

import json

import numpy as np
from PIL import Image
import pytest

from palimpsest.detection.algorithms.local_statistics.detector import LocalStatisticsDetector, StatisticsRule
from palimpsest.detection.algorithms.local_statistics.features import extract_features, FEATURE_NAMES
from palimpsest.detection.files import predict_file
from palimpsest.evaluation.classification import auc
from experiments.origin_detection.robust_statistics.rr.audit_sources import assign_roles
from experiments.origin_detection.robust_statistics.rr.evaluate_features import (
    auc_intervals, fit_family, selection_null_control, threshold_for_rule, validate_feature_rows,
)


def fixture_images():
    images, registry = [], []
    for label in ("ai", "real"):
        for index in range(3):
            source_id = f"{label}_{index}"
            registry.append({"source": f"{label}/{source_id}", "label": label, "status": "development"})
            for condition in ("original", "transfer", "redigital"):
                images.append({"filename": f"{condition}/{label}/{source_id}.png",
                               "condition": condition, "label": label, "source_id": source_id,
                               "sha256": f"{condition}_{label}_{index}"})
    config = {"conditions": ["original", "transfer", "redigital"], "seed": 42,
              "development_status": "development", "sources_per_class": 2,
              "fit_per_class": 1, "selection_per_class": 1, "threshold_per_class": 0}
    return images, registry, config


def test_audit_clean_duplicate_and_overlap():
    images, registry, config = fixture_images()
    # Independent by construction: real_0 and real_1 have the same original.
    for row in images:
        if row["condition"] == "original" and row["source_id"] == "real_1":
            row["sha256"] = "original_real_0"
    passports, selected, summary = assign_roles(images, registry, [], config)
    assert len(selected) == 12 and summary["development_sources"] == 4
    real_duplicates = [r for r in passports if r["source_group"] in ("real/real_0", "real/real_1")]
    assert real_duplicates[0]["component"] == real_duplicates[1]["component"]
    assert any(r["algorithm_role"] == "excluded_duplicate_of_algorithm_development" for r in real_duplicates)
    reduced = {**config, "sources_per_class": 1, "selection_per_class": 0}
    _, _, overlapped = assign_roles(images, registry, [{"sha256": "original_real_0"}], reduced)
    assert overlapped["direct_trainval_overlaps"] == ["real/real_0", "real/real_1"]
    omitted = [r for r in registry if r["source"] not in ("real/real_0", "real/real_1")]
    _, _, inherited = assign_roles(images, omitted, [{"sha256": "original_real_0"}], reduced)
    assert inherited["registry_omissions_verified_as_overlap"] == overlapped["direct_trainval_overlaps"]


@pytest.mark.parametrize("corruption", ["duplicate", "label", "missing"])
def test_audit_catches_corruptions(corruption):
    images, registry, config = fixture_images()
    if corruption == "duplicate":
        images.append(dict(images[0]))
    elif corruption == "label":
        images[0]["label"] = "real"
    else:
        images = [r for r in images if r["label"] == "real"]
    with pytest.raises(ValueError):
        assign_roles(images, registry, [], config)


@pytest.mark.parametrize("shape", [(1, 1, 3), (2, 3, 3), (64, 64, 3), (512, 512, 3)])
def test_constant_null(shape):
    result = extract_features(np.full(shape, 128, np.uint8))
    np.testing.assert_allclose(result.values, np.zeros(len(FEATURE_NAMES)), atol=1e-10)


def test_stripe_direction_known_answer():
    stripe = ((np.arange(128) % 8 < 4) * 255).astype(np.uint8)
    image = np.repeat(np.repeat(stripe[None, :, None], 128, axis=0), 3, axis=2)
    original = extract_features(image).values
    rotated = extract_features(np.rot90(image)).values
    assert original[4] > 0.99 and original[3] < original[4] - 0.1
    np.testing.assert_allclose(original[3], rotated[4], atol=1e-7)
    np.testing.assert_allclose(original[4], rotated[3], atol=1e-7)


def test_bad_dtype_and_feature_table_rejected():
    with pytest.raises(ValueError):
        extract_features(np.zeros((64, 64, 3), np.float32))
    rule = StatisticsRule("local", 512, (0,), (0,), (1,), (1,))
    with pytest.raises(ValueError):
        rule.score(np.full(17, np.nan))
    with pytest.raises(ValueError):
        StatisticsRule("local", 512, (0,), (0,), (0,), (1,))


def test_complete_file_rule_path_and_corruption(tmp_path):
    rule = StatisticsRule("local", 512, (0,), (0,), (1,), (1,), threshold=0.0)
    path = tmp_path / "parameters.json"
    rule.save(path)
    restored = StatisticsRule.load(path)
    assert restored.fingerprint == rule.fingerprint
    detector = LocalStatisticsDetector(restored)
    for filename in ("AI_name.png", "natural_name.png"):
        Image.fromarray(np.full((64, 64, 3), 128, np.uint8)).save(tmp_path / filename)
    left, right = [predict_file(detector, tmp_path / p) for p in ("AI_name.png", "natural_name.png")]
    assert left.prediction.score == right.prediction.score == 0.0
    assert left.prediction.origin.value == "natural"  # strictly greater than threshold
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["feature_names"][0] = "wrong_feature"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        StatisticsRule.load(path)


def test_auc_known_order_reversal_null():
    perfect = [(-2, 0), (-1, 0), (1, 1), (2, 1)]
    assert auc(perfect) == 1
    assert auc([(-s, label) for s, label in perfect]) == 0
    assert auc([(0, label) for _, label in perfect]) == 0.5


def planted_feature_table():
    # Answer is written by construction, not generated by image statistics:
    # feature 0 separates both classes perfectly; all other features are null.
    inventory, rows = [], []
    for role in ("fit", "selection", "threshold"):
        for label in ("ai", "real"):
            for index in range(2):
                source = f"{label}/{role}_{index}"
                for condition in ("original", "transfer", "redigital"):
                    filename = f"{condition}/{source}.png"
                    inventory.append({"filename": filename, "source_group": source, "condition": condition,
                                      "algorithm_role": role, "label": label})
                    for variant in ("raw", "jpeg90_444_after_resize"):
                        rows.append({"filename": filename, "src": source, "condition": condition,
                                     "role": role, "label": "FAKE" if label == "ai" else "REAL",
                                     "long_edge": "512", "variant": variant,
                                     **{name: (1 if label == "ai" else -1) if i == 0 else 0
                                        for i, name in enumerate(FEATURE_NAMES)}})
    config = {"long_edges": [512], "minimum_fit_auc": 0.55, "z_clip": 5.0,
              "seed": 42, "bootstrap_repetitions": 40}
    return inventory, rows, config


def test_rule_pipeline_recovers_plant_and_null():
    inventory, rows, config = planted_feature_table()
    table = validate_feature_rows(rows, inventory, config)
    rule, diagnostics = fit_family(table, "local", 512, config, "synthetic_inventory")
    assert rule.indices == (0,) and diagnostics[0]["transfer_auc"] == 1
    assert json.loads(json.dumps(diagnostics))[0]["kept"] is True
    selected = threshold_for_rule(table, rule, config)
    planted = np.zeros((2, 17))
    planted[:, 0] = [-1, 1]
    assert (selected.score(planted) > selected.threshold).tolist() == [False, True]
    labels = np.array([0, 0, 1, 1])
    interval = auc_intervals({"transfer": np.array([-1, -1, 1, 1]),
                              "redigital": np.array([-1, -1, 1, 1])}, labels, config)
    assert interval == {"transfer": [1.0, 1.0], "redigital": [1.0, 1.0]}
    for row in rows:
        row[FEATURE_NAMES[0]] = 0
    null = validate_feature_rows(rows, inventory, config)
    empty_rule, _ = fit_family(null, "local", 512, config, "synthetic_inventory")
    assert empty_rule.indices == () and np.all(empty_rule.score(planted) == 0)


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "label", "nonfinite"])
def test_feature_inventory_refuses_corruption(corruption):
    inventory, rows, config = planted_feature_table()
    validate_feature_rows(rows, inventory, config)  # the clean twin must pass
    if corruption == "duplicate":
        rows.append(dict(rows[0]))
    elif corruption == "missing":
        rows.pop()
    elif corruption == "label":
        rows[0]["label"] = "wrong_label"
    else:
        rows[0][FEATURE_NAMES[0]] = "nan"
    with pytest.raises(ValueError):
        validate_feature_rows(rows, inventory, config)


def test_source_label_shuffle_null_and_planted_signal():
    config = {"seed": 42, "bootstrap_repetitions": 40}
    labels = np.repeat([0, 1], 20)
    null = {"constant": {"transfer": np.zeros(40), "redigital": np.zeros(40)}}
    assert selection_null_control(null, labels, config)["permutation_p"] == 1.0
    signal = {"plant": {"transfer": labels * 2 - 1, "redigital": labels * 2 - 1}}
    control = selection_null_control(signal, labels, config)
    assert control["observed_max_worst_auc"] == 1.0
    assert control["permutation_p"] == 1 / 41
