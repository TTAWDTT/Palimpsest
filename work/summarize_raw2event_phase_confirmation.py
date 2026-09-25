"""Paired, source-level summary of the one-pass phase confirmation sample."""

import json
from pathlib import Path

import numpy as np


IN = Path("work/raw2event_phase_confirmation_evaluation.json")
OUT = Path("work/raw2event_phase_confirmation_summary.json")


def paired(rows: list[dict], first: str, second: str) -> dict:
    difference = np.asarray([r["conditions"][first]["mae_counts"] - r["conditions"][second]["mae_counts"]
                             for r in rows])
    rng = np.random.default_rng(20260925)
    draws = rng.choice(difference, size=(20000, len(difference)), replace=True).mean(axis=1)
    return {"mean_first_minus_second_counts": float(difference.mean()),
            "median_first_minus_second_counts": float(np.median(difference)),
            "first_lower_mae_count": int((difference < 0).sum()),
            "bootstrap_source_95_percentile": np.quantile(draws, [.025, .975]).tolist()}


def main() -> None:
    report = json.loads(IN.read_text(encoding="utf-8"))
    rows = report["per_source"]
    if len(rows) != 10 or len({r["class_name"] for r in rows}) != 10:
        raise RuntimeError("incomplete independent confirmation")
    base = "vertical_rgb__BGGR_diagonal"
    summary = {"manifest_sha256": report["manifest_sha256"], "n_sources": 10,
               "no_new_raw_used_for_weights": report["weights_fitted_only_on_prior_ten_calibration_sources"],
               "identity_rank_under_tag_similarity": [r["geometry"]["identity_without_source_refinement"]["named_source_rank_among_6000"] for r in rows],
               "bootstrap": "20000 source resamples with replacement, RNG seed 20260925; descriptive interval only",
               "bggr_minus_rggb": paired(rows, base, "vertical_rgb__RGGB_diagonal"),
               "bggr_vertical_minus_bggr_simple": paired(rows, base, "simple_rgb_sample_control__BGGR_diagonal"),
               "bggr_vertical_minus_bggr_co_spatial": paired(rows, base, "co_spatial_rgb_control__BGGR_diagonal"),
               "full_mix_minus_bggr_diagonal": paired(rows, "vertical_rgb__nonnegative_full", base)}
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
