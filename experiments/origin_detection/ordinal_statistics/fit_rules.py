"""Two registered frequency rules using source roles, no per-condition inference."""

from dataclasses import replace

import numpy as np

from palimpsest.contracts import Origin, Prediction
from palimpsest.detection.algorithms.ordinal_statistics.detector import OrdinalRule
from palimpsest.detection.algorithms.ordinal_statistics.features import FEATURE_NAMES
from palimpsest.evaluation.classification import auc, evaluate
from palimpsest.evaluation.detection import DetectionObservation, evaluate_detection
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals


def validate_rows(rows, inventory):
    expected = {r["filename"]: r for r in inventory}
    if not expected or len(expected) != len(inventory):
        raise ValueError("Empty/duplicate inventory")
    seen = set()
    for row in rows:
        name = row["filename"]
        if name not in expected or name in seen:
            raise ValueError("Unexpected/duplicate ordinal record")
        seen.add(name)
        if any(row[k] != expected[name][k] for k in ("src", "label", "role", "domain", "scene", "condition", "sha256")):
            raise ValueError("Ordinal source/label/role identity changed")
        values = np.asarray([float(row[f]) for f in FEATURE_NAMES])
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("Nonfinite/invalid frequency")
        for index in range(0, len(values), 7):
            if not np.isclose(values[index:index + 6].sum(), 1, atol=1e-12, rtol=0):
                raise ValueError("Orbit histogram not normalized")
    if seen != set(expected):
        raise ValueError("Missing ordinal records")
    # Complete paired role groups, no source divided between scenes/domains/labels/roles.
    groups = {}
    for r in rows:
        metadata = (r["label"], r["role"], r["domain"], r["scene"])
        prior, conditions = groups.setdefault(r["src"], (metadata, set()))
        if prior != metadata or r["condition"] in conditions:
            raise ValueError("Conflicting/duplicate paired source")
        conditions.add(r["condition"])
    for (metadata, conditions) in groups.values():
        expected_conditions = {"original", "transfer", "redigital"} if metadata[2] == "rr" else {"original", "mac_iphone", "lg_blackfly"}
        if conditions != expected_conditions:
            raise ValueError("Incomplete paired source")
    return rows


def views(rows, role, kind):
    groups = {}
    for row in rows:
        if row["role"] != role or (kind == "original") != (row["condition"] == "original"):
            continue
        key = f"{row['domain']}/{row['scene']}/{row['condition']}"
        groups.setdefault(key, []).append(row)
    output = {}
    for key, values in sorted(groups.items()):
        values.sort(key=lambda r: r["src"])
        labels = np.array([r["label"] == "FAKE" for r in values], dtype=int)
        if set(labels) != {0, 1}:
            raise ValueError("Each view needs both classes")
        output[key] = (values, np.asarray([[float(r[f]) for f in FEATURE_NAMES] for r in values]), labels)
    if len(output) != (4 if kind == "original" else 8):
        raise ValueError("Registered view count changed")
    return output


def choose_threshold(rule, view_data):
    scored = [(rule.score(values), labels) for _, values, labels in view_data.values()]
    unique = np.unique(np.concatenate([scores for scores, _ in scored]))
    thresholds = np.r_[np.nextafter(unique[0], -np.inf), unique]
    ranked = []
    for threshold in thresholds:
        accuracies = []
        for scores, labels in scored:
            correct = (scores > threshold) == labels.astype(bool)
            accuracies.append(float((correct[labels == 0].mean() + correct[labels == 1].mean()) / 2))
        ranked.append((min(accuracies), -abs(float(threshold)), -float(threshold), float(threshold)))
    return replace(rule, threshold=max(ranked)[-1])


def run_screen(rows, config, manifest_sha):
    fit_original = views(rows, "fit", "original")
    _, anchor, labels = fit_original["rr/all/original"]
    directions = [1 if auc(list(zip(anchor[:, i], labels))) >= 0.5 else -1 for i in range(len(FEATURE_NAMES))]
    outputs, rules = {}, {}
    for mode in config["candidate_modes"]:
        fit_views = fit_original if mode == "original_agreement" else views(rows, "fit", "processed")
        matrix, kept = {}, []
        for i, name in enumerate(FEATURE_NAMES):
            matrix[name] = {key: auc(list(zip(directions[i] * values[:, i], truth)))
                            for key, (_, values, truth) in fit_views.items()}
            kept.append(directions[i] if min(matrix[name].values()) >= config["minimum_fit_auc"] else 0)
        rule = choose_threshold(OrdinalRule(mode, tuple(kept), fit_manifest_sha256=manifest_sha),
                                views(rows, "threshold", "processed"))
        results = {}
        for kind in ("original", "processed"):
            for key, (records, values, truth) in views(rows, "selection", kind).items():
                scores = rule.score(values)
                metrics = evaluate([{**r, "score": float(s - rule.threshold)} for r, s in zip(records, scores)], "score")
                interval = auc_intervals({key: scores}, truth, config)[key]
                results[key] = {**metrics, "auc_ci95": interval}
        processed = [v for k, v in results.items() if not k.endswith("/original")]
        criteria = {
            "nonempty_rule": any(kept),
            "eight_auc_lower_bounds": all(v["auc_ci95"][0] > config["selection_auc_lower_gate"] for v in processed),
            "eight_both_class_accuracies": all(v[k] >= config["minimum_processed_class_accuracy"]
                                                for v in processed for k in ("real_accuracy_at_zero", "fake_accuracy_at_zero")),
        }
        aggregates = {}
        for domain in ("rr", "chimera"):
            selected = [r for r in rows if r["domain"] == domain and r["role"] == "selection"]
            values = np.asarray([[float(r[f]) for f in FEATURE_NAMES] for r in selected])
            aggregates[domain] = evaluate_detection([
                DetectionObservation(r["src"], r["condition"], Origin.AI if r["label"] == "FAKE" else Origin.NATURAL,
                                     Prediction(mode, float(s), rule.threshold, "statistical_score"))
                for r, s in zip(selected, rule.score(values))])
        outputs[mode] = {"kept_features": [n for n, d in zip(FEATURE_NAMES, kept) if d],
                         "directions": kept, "fit_direction_auc_matrix": matrix, "selection_views": results,
                         "selection_aggregate": aggregates,
                         "criteria": criteria, "gate_passed": all(criteria.values()),
                         "worst_processed_auc": min(v["auc"] for v in processed), "threshold": rule.threshold}
        rules[mode] = rule
    passing = [m for m in config["candidate_modes"] if outputs[m]["gate_passed"]]
    chosen = max(passing, key=lambda m: (outputs[m]["worst_processed_auc"], m == "original_agreement")) if passing else None
    return {"candidates": outputs, "chosen": chosen,
            "scope": "RR and previously exposed Chimera joint development; not independent validation"}, rules
