"""Separate score compression from common offset on frozen Chimera development."""

from palimpsest.paths import DATA_ROOT, WORK_DIR

import csv
import json

import numpy as np

from palimpsest.data.inference import validated_inference


SPLIT = DATA_ROOT / "manifests/chimera_simulation_source_split.csv"
BASE_MANIFEST = DATA_ROOT / "manifests/chimera_bfree_manifest.csv"
REAL_MANIFEST = DATA_ROOT / "manifests/chimera_recap256_bfree_manifest.csv"
SIM_MANIFESTS = {
    "striped": (
        DATA_ROOT / "manifests/chimera_virtual_screen_mac_dev256_manifest.csv",
        WORK_DIR / "chimera_virtual_screen_mac_dev256_bfree.csv",
        WORK_DIR / "chimera_virtual_screen_mac_dev256_bfree.json",
        "virtual_mac",
    ),
    "co_spatial_control": (
        DATA_ROOT
        / "manifests/chimera_virtual_screen_mac_dev256_co_spatial_manifest.csv",
        WORK_DIR / "chimera_virtual_screen_mac_co_spatial_dev256_bfree.csv",
        WORK_DIR / "chimera_virtual_screen_mac_co_spatial_dev256_bfree.json",
        "virtual_mac_co_spatial",
    ),
    "effective_fit": (
        DATA_ROOT
        / "manifests/chimera_virtual_screen_mac_dev256_effective_fit_manifest.csv",
        WORK_DIR / "chimera_virtual_screen_mac_effective_fit_dev256_bfree.csv",
        WORK_DIR / "chimera_virtual_screen_mac_effective_fit_dev256_bfree.json",
        "virtual_mac_effective_fit",
    ),
}
OUTPUT = WORK_DIR / "chimera_score_contraction.json"


def score_map(rows: list[dict]) -> dict[str, float]:
    keyed = {row["src"]: float(row["score"]) for row in rows}
    if len(keyed) != len(rows):
        raise RuntimeError("duplicate score source ID")
    return keyed


def slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def main() -> None:
    with SPLIT.open(newline="", encoding="utf-8-sig") as stream:
        selected = {
            row["src"]: row
            for row in csv.DictReader(stream)
            if row["split"] == "development"
        }
    if len(selected) != 120:
        raise RuntimeError("expected 120 development sources")
    base = validated_inference(
        BASE_MANIFEST,
        WORK_DIR / "chimera_bfree_full.csv",
        WORK_DIR / "chimera_bfree_full.json",
        3600,
    )
    real = validated_inference(
        REAL_MANIFEST,
        WORK_DIR / "chimera_bfree_recap256.csv",
        WORK_DIR / "chimera_bfree_recap256.json",
        2400,
    )
    tables = {
        "original": score_map(base["stylegan2_orig"]),
        "real": score_map(real["recap_mac"]),
    }
    for name, (manifest, csv_file, summary, condition) in SIM_MANIFESTS.items():
        tables[name] = score_map(
            validated_inference(manifest, csv_file, summary, 120)[condition]
        )
    ids = sorted(selected)
    if any(not set(ids).issubset(table) for table in tables.values()):
        raise RuntimeError("score source groups do not align")
    x = np.array([tables["original"][source] for source in ids])
    strata = [
        np.array(
            [
                index
                for index, source in enumerate(ids)
                if selected[source]["scene"] == scene
                and selected[source]["label"] == label
            ]
        )
        for scene in ("cat", "church", "horse")
        for label in ("REAL", "FAKE")
    ]
    if any(len(group) != 20 for group in strata):
        raise RuntimeError(
            "expected 20 development sources in each scene/label stratum"
        )
    rng = np.random.default_rng(20260925)
    sampled = np.concatenate(
        [
            group[rng.integers(0, len(group), size=(5000, len(group)))]
            for group in strata
        ],
        axis=1,
    )
    result = {
        "scope": "120 frozen Chimera development sources, fixed B-Free logits; descriptive affine score mapping",
        "warning": "OLS slope/intercept are task diagnostics, not a physical image formation model; exploratory development data",
        "original_mean": float(x.mean()),
        "original_sd": float(x.std()),
        "conditions": {},
    }
    for name, table in tables.items():
        if name == "original":
            continue
        y = np.array([table[source] for source in ids])
        slope, intercept = slope_intercept(x, y)
        bootstrap_slope = np.array(
            [slope_intercept(x[index], y[index])[0] for index in sampled]
        )
        result["conditions"][name] = {
            "score_mean": float(y.mean()),
            "score_sd": float(y.std()),
            "mean_shift": float((y - x).mean()),
            "pearson_with_original": float(np.corrcoef(x, y)[0, 1]),
            "ols_slope": slope,
            "ols_intercept": intercept,
            "stratified_bootstrap_slope_ci95": [
                float(np.quantile(bootstrap_slope, 0.025)),
                float(np.quantile(bootstrap_slope, 0.975)),
            ],
        }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
