"""Source-level paired descriptions for the exploratory CFA phase probe."""

import json
from pathlib import Path

import numpy as np


PHASE = Path("work/raw2event_cfa_phase_probe.json")
MIX = Path("work/raw2event_spectral_mix_v1.json")
OUT = Path("work/raw2event_cfa_phase_summary.json")


def development(condition: dict) -> dict:
    return {row["prefix"]: row for row in condition["per_source"] if row["role"] == "development"}


def paired(first: dict, second: dict, keys: list[str]) -> dict:
    values = np.asarray([first[key]["mae_counts"] - second[key]["mae_counts"] for key in keys])
    rng = np.random.default_rng(20260925)
    bootstrap = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    return {"n_sources": len(values), "mean_first_minus_second_counts": float(values.mean()),
            "median_first_minus_second_counts": float(np.median(values)),
            "first_lower_mae_count": int((values < 0).sum()),
            "bootstrap_source_95_percentile": np.quantile(bootstrap, [.025, .975]).tolist()}


def main() -> None:
    phase = json.loads(PHASE.read_text(encoding="utf-8"))["conditions"]
    mix = json.loads(MIX.read_text(encoding="utf-8"))["conditions"]
    vertical = phase["vertical_rgb"]
    bggr = development(vertical["BGGR"])
    rggb = development(vertical["RGGB"])
    plain = development(phase["simple_rgb_sample_control"]["BGGR"])
    full = development(mix["vertical_rgb__nonnegative_full"])
    keys = sorted(bggr)
    if len(keys) != 10 or any(set(group) != set(keys) for group in (rggb, plain, full)):
        raise RuntimeError("development sources do not match")
    report = {"sampling_unit": "source, one per CIFAR class",
              "bootstrap": "20000 source-level resamples with replacement, RNG seed 20260925",
              "phase_selected_after_development_inspection": True,
              "bggr_minus_rggb": paired(bggr, rggb, keys),
              "bggr_vertical_minus_bggr_simple": paired(bggr, plain, keys),
              "full_mix_minus_bggr_diagonal": paired(full, bggr, keys),
              "interpretation": "descriptive intervals for selected development sources, not independent confirmatory inference"}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
