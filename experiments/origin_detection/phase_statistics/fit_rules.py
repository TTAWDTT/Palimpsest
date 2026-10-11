"""Two registered decisions, eight view gates and recompression diagnostics."""

import numpy as np

from palimpsest.contracts import Origin, Prediction
from palimpsest.detection.algorithms.phase_statistics.detector import fit_rules
from palimpsest.detection.algorithms.phase_statistics.features import FEATURE_NAMES
from palimpsest.evaluation.classification import evaluate
from palimpsest.evaluation.detection import DetectionObservation, evaluate_detection
from palimpsest.evaluation.features import feature_views
from experiments.origin_detection.ordinal_statistics.fit_rules import choose_threshold
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals


def views(rows, role, processed, variant="raw"):
    result = feature_views(rows, FEATURE_NAMES, role, processed=processed, variant=variant)
    if len(result) != (8 if processed else 4):
        raise ValueError("Registered phase view count differs")
    return result


def aggregate(rows, rule, variant):
    output = {}
    for domain in ("rr", "chimera"):
        selected = [r for r in rows if r["role"] == "selection" and r["domain"] == domain and r["variant"] == variant]
        values = np.asarray([[float(r[f]) for f in FEATURE_NAMES] for r in selected])
        output[domain] = evaluate_detection([
            DetectionObservation(r["src"], r["condition"], Origin.AI if r["label"] == "FAKE" else Origin.NATURAL,
                                 Prediction(rule.mode, float(s), rule.threshold, "statistical_score"))
            for r, s in zip(selected, rule.score(values))])
    return output


def run_screen(rows, config, manifest_sha):
    rules = fit_rules(views(rows, "fit", True), clip=config["z_clip"], manifest_sha=manifest_sha)
    outputs = {}
    for mode in config["candidate_modes"]:
        rule = choose_threshold(rules[mode], views(rows, "threshold", True))
        rules[mode] = rule
        strata = {}
        for processed in (False, True):
            for name, (records, values, labels) in views(rows, "selection", processed).items():
                scores = rule.score(values)
                strata[name] = {**evaluate([{**r, "score": float(s - rule.threshold)} for r, s in zip(records, scores)], "score"),
                                "auc_ci95": auc_intervals({name: scores}, labels, config)[name]}
        raw, recompressed = aggregate(rows, rule, "raw"), aggregate(rows, rule, "jpeg90_444_after_resize")
        processed_views = [v for k, v in strata.items() if not k.endswith("/original")]
        compression_metrics = [(raw[d]["conditions"][c]["balanced_accuracy_at_zero"], v["balanced_accuracy_at_zero"])
                               for d, report in recompressed.items() for c, v in report["conditions"].items() if c != "original"]
        criteria = {
            "eight_auc_lower_bounds": all(v["auc_ci95"][0] > config["selection_auc_lower_gate"] for v in processed_views),
            "eight_both_class_accuracies": all(v[k] >= config["minimum_processed_class_accuracy"]
                                                for v in processed_views for k in ("real_accuracy_at_zero", "fake_accuracy_at_zero")),
            "both_original_class_accuracies": all(r["conditions"]["original"][k] >= config["minimum_original_class_accuracy"]
                                                  for r in raw.values() for k in ("real_accuracy_at_zero", "fake_accuracy_at_zero")),
            "recompression_ba_floor": all(after >= config["minimum_reencoded_ba"] for _, after in compression_metrics),
            "recompression_ba_drop": all(before - after <= config["maximum_reencoded_ba_drop"] for before, after in compression_metrics),
        }
        outputs[mode] = {"criteria": {k: bool(v) for k, v in criteria.items()}, "gate_passed": all(criteria.values()),
                         "worst_processed_auc": min(v["auc"] for v in processed_views), "selection_views": strata,
                         "selection_aggregate": raw, "recompression_aggregate": recompressed,
                         "threshold": rule.threshold, "view_names": list(rule.view_names)}
    passing = [m for m in config["candidate_modes"] if outputs[m]["gate_passed"]]
    chosen = max(passing, key=lambda m: (outputs[m]["worst_processed_auc"], m == "linear")) if passing else None
    return {"candidates": outputs, "chosen": chosen,
            "scope": "Fourth joint RR/Chimera development iteration; same exposed sources; no independent validation"}, rules
