"""Apply a frozen effective JPEG law to never-fit ImageNet-ES source classes."""

from palimpsest.paths import WORK_DIR

import json
from collections import defaultdict

import numpy as np

from experiments.screen_capture.exposure_response.imagenet_es.evaluate_aperture_pairs import (
    linear_to_srgb,
    srgb_to_linear,
)
from experiments.screen_capture.exposure_response.imagenet_es.fit_effective_tone import (
    BINS,
    PARAMS,
    histogram,
    quantiles,
    transfer,
)


DEV_MANIFEST = WORK_DIR / "imagenet_es_20class_aperture_iso_manifest.json"
HOLDOUT_MANIFEST = (
    WORK_DIR / "imagenet_es_20class_aperture_iso_content_holdout_manifest.json"
)
DEV_RESULT = WORK_DIR / "imagenet_es_effective_tone_dev.json"
OUTPUT = WORK_DIR / "imagenet_es_effective_tone_content_holdout.json"
FROZEN_GAMMA = 0.5694751716876738
FROZEN_TOE = 0.09504581492211553


def standard_srgb_exposure(encoded: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(linear_to_srgb(srgb_to_linear(encoded) * factor), 0, 1)


def main() -> None:
    dev = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
    holdout = json.loads(HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    fit = json.loads(DEV_RESULT.read_text(encoding="utf-8"))
    if (
        holdout.get("class_offset") != 20
        or len(holdout["source_classes"]) != 20
        or set(dev["source_classes"]) & set(holdout["source_classes"])
    ):
        raise ValueError("content holdout is not disjoint from development")
    dev_reference_hashes = {
        item["sha256"]
        for item in dev["rows"]
        if "sampled_tin_no_resize2" in item["member"]
    }
    holdout_reference_hashes = {
        item["sha256"]
        for item in holdout["rows"]
        if "sampled_tin_no_resize2" in item["member"]
    }
    if (
        len(dev_reference_hashes) != 20
        or len(holdout_reference_hashes) != 20
        or (dev_reference_hashes & holdout_reference_hashes)
    ):
        raise ValueError("reference images are missing, repeated or exact-overlapping")
    if (
        abs(fit["gamma"] - FROZEN_GAMMA) > 1e-12
        or abs(fit["toe"] - FROZEN_TOE) > 1e-12
        or fit["fit_param_ids"] != [4, 13]
    ):
        raise ValueError("fitted parameters differ from committed preregistration")
    groups = defaultdict(dict)
    for item in holdout["rows"]:
        parts = item["member"].split("/")
        if "param_control" not in parts:
            continue
        key = f"{parts[-2]}/{parts[-1]}"
        p = int(parts[4].split("_")[1])
        if p in groups[key]:
            raise ValueError(f"duplicate condition {key}, {p}")
        groups[key][p] = item["local_path"]
    if len(groups) != 20 or any(set(g) != set(PARAMS) for g in groups.values()):
        raise ValueError("incomplete holdout group")
    centers = (np.arange(BINS) + 0.5) / BINS
    rows = []
    for source, group in sorted(groups.items()):
        h = {p: histogram(path) for p, path in group.items()}
        q_base = quantiles(h[4])
        for p, (f_number, iso) in PARAMS.items():
            if p == 4:
                continue
            factor = (5 / f_number) ** 2 * iso / 250
            predicted_mean = float(
                (h[4] * transfer(centers, factor, FROZEN_GAMMA, FROZEN_TOE)).sum()
            )
            srgb_mean = float((h[4] * standard_srgb_exposure(centers, factor)).sum())
            observed_mean = float((h[p] * centers).sum())
            q_error = float(
                np.abs(
                    transfer(q_base, factor, FROZEN_GAMMA, FROZEN_TOE) - quantiles(h[p])
                ).mean()
            )
            srgb_q_error = float(
                np.abs(standard_srgb_exposure(q_base, factor) - quantiles(h[p])).mean()
            )
            rows.append(
                {
                    "source_key": source,
                    "param_id": p,
                    "nominal_gain_area_factor": factor,
                    "predicted_mean": predicted_mean,
                    "observed_mean": observed_mean,
                    "mean_abs_error": abs(predicted_mean - observed_mean),
                    "quantile_abs_error_mean": q_error,
                    "standard_srgb_predicted_mean": srgb_mean,
                    "standard_srgb_mean_abs_error": abs(srgb_mean - observed_mean),
                    "standard_srgb_quantile_abs_error_mean": srgb_q_error,
                }
            )
    by_param = {}
    rng = np.random.default_rng(190)
    for p in PARAMS:
        if p == 4:
            continue
        subset = [r for r in rows if r["param_id"] == p]
        improvement = np.array(
            [r["standard_srgb_mean_abs_error"] - r["mean_abs_error"] for r in subset]
        )
        boot_indexes = rng.integers(0, len(improvement), (5000, len(improvement)))
        by_param[str(p)] = {
            "n": len(subset),
            "mean_abs_error_median": float(
                np.median([r["mean_abs_error"] for r in subset])
            ),
            "quantile_abs_error_mean_median": float(
                np.median([r["quantile_abs_error_mean"] for r in subset])
            ),
            "predicted_over_observed_mean_median": float(
                np.median([r["predicted_mean"] / r["observed_mean"] for r in subset])
            ),
            "predicted_brighter_count": sum(
                r["predicted_mean"] > r["observed_mean"] for r in subset
            ),
            "standard_srgb_mean_abs_error_median": float(
                np.median([r["standard_srgb_mean_abs_error"] for r in subset])
            ),
            "standard_srgb_quantile_abs_error_mean_median": float(
                np.median([r["standard_srgb_quantile_abs_error_mean"] for r in subset])
            ),
            "frozen_tone_lower_mean_error_count": sum(
                r["mean_abs_error"] < r["standard_srgb_mean_abs_error"] for r in subset
            ),
            "paired_median_mean_error_improvement_over_srgb": float(
                np.median(improvement)
            ),
            "source_bootstrap_95_percent_improvement": np.quantile(
                np.median(improvement[boot_indexes], axis=1), [0.025, 0.975]
            ).tolist(),
        }
    report = {
        "scope": "unseen ImageNet-ES source classes SHA ranks 21-40; 20 sources, six capture conditions",
        "fixed_gamma": FROZEN_GAMMA,
        "fixed_toe": FROZEN_TOE,
        "development_classes": dev["source_classes"],
        "holdout_classes": holdout["source_classes"],
        "reference_exact_sha256_overlap": 0,
        "per_condition": by_param,
        "rows": rows,
        "qualification": "content-disjoint within same camera/dataset; parameters not refit; JPEG-level tone law only",
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
