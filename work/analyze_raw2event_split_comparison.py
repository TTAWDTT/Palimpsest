"""Paired descriptive analysis of the frozen Raw2Event forward-RAW probes."""

import json
from pathlib import Path

import numpy as np


BASE = Path("work/raw2event_source_to_raw_split_v1.json")
REFINED = Path("work/raw2event_source_to_raw_split_v1_source_rgb_refined.json")
OUT = Path("work/raw2event_source_to_raw_comparison.json")
METHODS = ("vertical_rgb", "co_spatial_rgb_control", "simple_rgb_sample_control")


def source_map(report: dict, method: str, role: str) -> dict:
    return {row["prefix"]: row for row in report["conditions"][method]["per_source"]
            if row["role"] == role}


def interval(values: np.ndarray) -> dict:
    rng = np.random.default_rng(20260925)
    draws = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "bootstrap_source_95_percentile": np.quantile(draws, [0.025, 0.975]).tolist(),
            "negative_count": int((values < 0).sum()), "positive_count": int((values > 0).sum())}


def main() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    refined = json.loads(REFINED.read_text(encoding="utf-8"))
    if base["split_sha256"] != refined["split_sha256"] or base["source_count"] != 20 or refined["source_count"] != 20:
        raise RuntimeError("split mismatch")
    keys = sorted(source_map(base, METHODS[0], "development"))
    if len(keys) != 10:
        raise RuntimeError("expected ten heldout development sources")
    for report in (base, refined):
        for method in METHODS:
            if sorted(source_map(report, method, "development")) != keys:
                raise RuntimeError("source mismatch")
    comparisons = {}
    for name, report in (("tag_similarity", base), ("source_rgb_refined", refined)):
        metrics = {method: source_map(report, method, "development") for method in METHODS}
        comparisons[name] = {
            "vertical_minus_simple_mae_counts": interval(np.asarray([
                metrics["vertical_rgb"][key]["mae_counts"] - metrics["simple_rgb_sample_control"][key]["mae_counts"]
                for key in keys])),
            "vertical_minus_co_spatial_mae_counts": interval(np.asarray([
                metrics["vertical_rgb"][key]["mae_counts"] - metrics["co_spatial_rgb_control"][key]["mae_counts"]
                for key in keys])),
        }
    refinement = {}
    for method in METHODS:
        old = source_map(base, method, "development")
        new = source_map(refined, method, "development")
        refinement[method] = interval(np.asarray([new[key]["mae_counts"] - old[key]["mae_counts"]
                                                  for key in keys]))
    report = {"split_sha256": base["split_sha256"], "n_development_sources": len(keys),
              "sampling_unit": "source, one development capture per CIFAR class",
              "difference_sign": "first minus second; negative means first has lower MAE",
              "comparisons": comparisons, "refined_minus_tag_mae_counts": refinement,
              "qualification": "descriptive bootstrap across ten selected sources; no independent device replication; geometry refinement chosen after seeing initial development errors"}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
