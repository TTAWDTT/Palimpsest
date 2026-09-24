"""Test whether native high-band excess tracks source content and scene.

All comparisons remain observational: Chimera has no physical intervention
grid or RAW. Calibration and development groups are reported separately;
reserved sources are never opened.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

from probe_chimera_native_bandpower import powers
from probe_chimera_native_frequency import (
    GEOMETRY, PATCH, ROOT, SPLIT, flattest_source_center, native_patches,
)


OUTPUT = Path("work/chimera_highband_confounders.json")
RECORDS = Path("work/chimera_highband_confounders.csv")
CONTROLS = ("two_stage_lanczos", "direct_linear")
MATCH_FEATURES = ("source_mean", "source_std", "source_gradient")


def source_features(source_id: str) -> dict[str, float]:
    image = cv2.imread(str(ROOT / "stylegan2_orig" / source_id), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (256, 256):
        raise RuntimeError(f"invalid source: {source_id}")
    gray = image.astype(np.float32) / 255
    cx, cy = flattest_source_center(gray)
    x, y = int(cx - 32), int(cy - 32)
    patch = gray[y:y + 64, x:x + 64]
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0) / 8
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1) / 8
    return {"source_mean": float(patch.mean()),
            "source_std": float(patch.std()),
            "source_gradient": float(np.hypot(gx, gy).mean())}


def numeric(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    return {"median": float(np.median(a)), "p10": float(np.quantile(a, .1)),
            "p90": float(np.quantile(a, .9))}


def summarize(rows: list[dict[str, object]], control: str) -> dict[str, object]:
    key = f"excess_db_{control}"
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scene"], row["label"])].append(float(row[key]))
    strata = {f"{scene}/{label}": {"count": len(values), **numeric(values)}
              for (scene, label), values in sorted(grouped.items())}
    by_label = {label: numeric([float(row[key]) for row in rows if row["label"] == label])
                for label in ("REAL", "FAKE")}
    correlates = {}
    for feature in ("source_mean", "source_std", "source_gradient", f"control_logpower_{control}"):
        x = np.asarray([float(row[feature]) for row in rows])
        y = np.asarray([float(row[key]) for row in rows])
        correlates[feature] = float(spearmanr(x, y).statistic)
    return {"all": numeric([float(row[key]) for row in rows]),
            "by_label": by_label, "scene_label": strata,
            "spearman_with_excess": correlates}


def matched_label_contrast(rows: list[dict[str, object]], scale: np.ndarray,
                           control: str, seed: int) -> dict[str, object]:
    """Within-scene REAL−FAKE excess after 1:1 source-feature matching."""
    rng = np.random.default_rng(seed)
    differences = []
    distances = []
    scenes = {}
    for scene in sorted({str(row["scene"]) for row in rows}):
        real = [row for row in rows if row["scene"] == scene and row["label"] == "REAL"]
        fake = [row for row in rows if row["scene"] == scene and row["label"] == "FAKE"]
        if len(real) != len(fake) or not real:
            raise RuntimeError(f"unbalanced scene {scene}")
        rx = np.asarray([[float(row[key]) for key in MATCH_FEATURES] for row in real]) / scale
        fx = np.asarray([[float(row[key]) for key in MATCH_FEATURES] for row in fake]) / scale
        cost = np.linalg.norm(rx[:, None, :] - fx[None, :, :], axis=2)
        ri, fi = linear_sum_assignment(cost)
        local = [float(real[i][f"excess_db_{control}"]) -
                 float(fake[j][f"excess_db_{control}"]) for i, j in zip(ri, fi)]
        differences.extend(local)
        distances.extend(float(cost[i, j]) for i, j in zip(ri, fi))
        scenes[scene] = {"pairs": len(local), "real_minus_fake_excess_db": numeric(local),
                         "match_distance": numeric([float(cost[i, j]) for i, j in zip(ri, fi)])}
    values = np.asarray(differences)
    boot = np.median(values[rng.integers(0, len(values), size=(3000, len(values)))], axis=1)
    return {"pairs": len(differences), "real_minus_fake_excess_db": numeric(differences),
            "paired_median_bootstrap_95": [float(np.quantile(boot, .025)),
                                            float(np.quantile(boot, .975))],
            "match_distance": numeric(distances), "scenes": scenes}


def main() -> None:
    with SPLIT.open(newline="", encoding="utf-8-sig") as stream:
        sources = [row for row in csv.DictReader(stream)
                   if row["split"] in ("calibration", "development")]
    if len(sources) != 360:
        raise RuntimeError("expected 360 nonreserved source groups")
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    f = np.fft.fftshift(np.fft.fftfreq(PATCH))
    fx, fy = np.meshgrid(f, f)
    r = np.hypot(fx, fy)
    masks = {"high": (r >= .30) & (r < .45)}
    window = np.outer(np.hanning(PATCH), np.hanning(PATCH)).astype(np.float32)
    records = []
    for source in sources:
        features = source_features(source["src"])
        for condition in ("recap_mac", "recap_monitor"):
            warp = np.asarray(geometry["conditions"][condition]["median_calibration_affine"],
                              dtype=np.float32)
            real, controls = native_patches(source["src"], condition, warp)
            real_power = powers(real, masks, window)["high"]
            row = {"src": source["src"], "split": source["split"],
                   "scene": source["scene"], "label": source["label"],
                   "condition": condition, **features,
                   "real_logpower": float(np.log10(real_power + 1e-12))}
            for name in CONTROLS:
                control_power = powers(controls[name], masks, window)["high"]
                row[f"control_logpower_{name}"] = float(np.log10(control_power + 1e-12))
                row[f"excess_db_{name}"] = float(10 * np.log10(
                    (real_power + 1e-12) / (control_power + 1e-12)))
            records.append(row)
    with RECORDS.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = {"scope": "source-level 240 calibration / 120 development; 2 devices",
              "warning": "descriptive confounder probe on same datasets, not causal attribution",
              "conditions": {}}
    calibration_features = np.asarray([
        [float(row[name]) for name in MATCH_FEATURES]
        for row in records if row["split"] == "calibration" and row["condition"] == "recap_mac"
    ])
    feature_scale = calibration_features.std(axis=0)
    if np.any(feature_scale <= 0):
        raise RuntimeError("source features have zero calibration scale")
    result["matching"] = {"features": MATCH_FEATURES,
                          "calibration_standard_deviations": feature_scale.tolist(),
                          "method": "Hungarian 1:1 REAL/FAKE within scene; no outcome used for matching"}
    for split in ("calibration", "development"):
        result["conditions"][split] = {}
        for condition in ("recap_mac", "recap_monitor"):
            group = [row for row in records
                     if row["split"] == split and row["condition"] == condition]
            result["conditions"][split][condition] = {
                "count": len(group),
                "source_features_by_label": {
                    label: {name: numeric([float(row[name]) for row in group if row["label"] == label])
                            for name in ("source_mean", "source_std", "source_gradient")}
                    for label in ("REAL", "FAKE")},
                "controls": {name: {**summarize(group, name),
                                    "matched_label_contrast": matched_label_contrast(
                                        group, feature_scale, name, seed=20260925)}
                             for name in CONTROLS}}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({split: {condition: result["conditions"][split][condition]["controls"]["two_stage_lanczos"]["by_label"]
                              for condition in ("recap_mac", "recap_monitor")}
                      for split in ("calibration", "development")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
