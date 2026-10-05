"""Falsify a one-parameter-linked effective JPEG tone law on screen pairs.

This is an observational JPEG response probe, not an inverse RAW/ISP model.
Only param 4 -> 13 (f/5 -> f/9 at ISO250) is used for fitting; the other
four capture conditions are withheld interventions on the same sources.
"""

from palimpsest.paths import WORK_DIR

import json
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.optimize import least_squares


MANIFEST = WORK_DIR / "imagenet_es_20class_aperture_iso_manifest.json"
OUTPUT = WORK_DIR / "imagenet_es_effective_tone_dev.json"
BINS = 1024
QUANTILES = np.linspace(0.1, 0.9, 17)
PARAMS = {
    4: (5, 250),
    5: (5, 2000),
    13: (9, 250),
    14: (9, 2000),
    22: (16, 250),
    23: (16, 2000),
}
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def histogram(path: str) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    values = (rgb * LUMA).sum(2)
    hist = np.histogram(values, bins=BINS, range=(0, 1))[0].astype(np.float64)
    return hist / hist.sum()


def quantiles(hist: np.ndarray) -> np.ndarray:
    cumulative = np.cumsum(hist)
    indexes = np.searchsorted(cumulative, QUANTILES)
    indexes = np.minimum(indexes, BINS - 1)
    previous = np.where(indexes > 0, cumulative[np.maximum(indexes - 1, 0)], 0)
    fraction = np.clip((QUANTILES - previous) / np.maximum(hist[indexes], 1e-12), 0, 1)
    return (indexes + fraction) / BINS


def transfer(
    encoded: np.ndarray, factor: float, gamma: float, toe: float
) -> np.ndarray:
    # Phenomenological *released JPEG* response. The physical exposure/gain
    # factor is linked across interventions; gamma and toe are shared.
    return np.clip((encoded + toe) * factor**gamma - toe, 0, 1)


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    groups = defaultdict(dict)
    for item in manifest["rows"]:
        parts = item["member"].split("/")
        if "param_control" not in parts:
            continue
        key = f"{parts[-2]}/{parts[-1]}"
        param_id = int(parts[4].split("_")[1])
        groups[key][param_id] = item["local_path"]
    if len(groups) != 20 or any(set(group) != set(PARAMS) for group in groups.values()):
        raise ValueError("expected twenty complete six-condition development groups")
    observations = {}
    for key, group in sorted(groups.items()):
        observations[key] = {p: histogram(group[p]) for p in PARAMS}
    q_base = np.stack([quantiles(row[4]) for row in observations.values()])
    q_f9 = np.stack([quantiles(row[13]) for row in observations.values()])
    factor_f9 = (5 / 9) ** 2

    def residual(theta: np.ndarray) -> np.ndarray:
        gamma, toe = theta
        return (transfer(q_base, factor_f9, gamma, toe) - q_f9).ravel()

    fit = least_squares(
        residual,
        x0=[0.75, 0.02],
        bounds=([0.1, 0], [2, 0.3]),
        loss="soft_l1",
        f_scale=0.02,
    )
    gamma, toe = map(float, fit.x)
    centers = (np.arange(BINS) + 0.5) / BINS
    rows = []
    for source, histograms in observations.items():
        baseline_mean = float((histograms[4] * centers).sum())
        for p, (f_number, iso) in PARAMS.items():
            if p == 4:
                continue
            factor = (5 / f_number) ** 2 * iso / 250
            observed_mean = float((histograms[p] * centers).sum())
            predicted_mean = float(
                (histograms[4] * transfer(centers, factor, gamma, toe)).sum()
            )
            predicted_q = transfer(quantiles(histograms[4]), factor, gamma, toe)
            observed_q = quantiles(histograms[p])
            rows.append(
                {
                    "source_key": source,
                    "param_id": p,
                    "nominal_gain_area_factor": factor,
                    "baseline_mean": baseline_mean,
                    "observed_mean": observed_mean,
                    "predicted_mean": predicted_mean,
                    "mean_abs_error": abs(predicted_mean - observed_mean),
                    "quantile_abs_error_mean": float(
                        np.mean(np.abs(predicted_q - observed_q))
                    ),
                }
            )
    by_param = {}
    for p in PARAMS:
        if p == 4:
            continue
        subset = [r for r in rows if r["param_id"] == p]
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
        }
    report = {
        "scope": "effective JPEG luma law calibrated ONLY on f5 ISO250 -> f9 ISO250, 20 development sources",
        "formula": "clip((encoded_luma + toe) * ((5/f)^2 * (ISO/250))^gamma - toe, 0, 1)",
        "fit_param_ids": [4, 13],
        "withheld_condition_ids": [5, 14, 22, 23],
        "gamma": gamma,
        "toe": toe,
        "fit_success": bool(fit.success),
        "per_condition": by_param,
        "rows": rows,
        "qualification": "JPEG quantile/histogram operator, not sensor photon model; withheld conditions share the same 20 sources and are not independent content tests",
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
