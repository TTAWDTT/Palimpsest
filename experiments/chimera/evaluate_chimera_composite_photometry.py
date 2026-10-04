"""Calibrate a shared RGB response after fixed post-crop geometry alignment.

The 3x3+b map is an empirical composite of display, camera, ISP and publication
effects in gamma-encoded PNG values. It must not be interpreted as a camera CCM.
"""

from palimpsest.paths import WORK_DIR

import json
from pathlib import Path

import cv2
import numpy as np

from experiments.chimera.evaluate_chimera_fixed_publication_geometry import (
    AUDIT,
    CONTROL,
    ROOT,
)


GEOMETRY = WORK_DIR / "chimera_fixed_publication_geometry.json"
OUTPUT = WORK_DIR / "chimera_composite_photometry.json"


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape[:2] != (256, 256):
        raise RuntimeError(f"cannot decode 256x256 RGB: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255


def align_rgb(recap: np.ndarray, warp: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(
        recap,
        warp,
        (256, 256),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )


def design(rgb: np.ndarray) -> np.ndarray:
    pixels = rgb.reshape(-1, 3)
    return np.column_stack((pixels, np.ones(len(pixels), dtype=np.float32)))


def fit_channel(
    condition: str, warp: np.ndarray, calibration: list[dict]
) -> np.ndarray:
    gram = np.zeros((4, 4), dtype=np.float64)
    cross = np.zeros((4, 3), dtype=np.float64)
    for item in calibration:
        source = item["src"]
        original = load_rgb(ROOT / "stylegan2_orig" / source)
        recapture = load_rgb(CONTROL / condition / source)
        aligned = align_rgb(recapture, warp)
        sample = (slice(16, 240, 4), slice(16, 240, 4))
        x = design(original[sample])
        y = aligned[sample].reshape(-1, 3)
        gram += x.T @ x
        cross += x.T @ y
    ridge = np.diag([1e-4, 1e-4, 1e-4, 0])
    return np.linalg.solve(gram + ridge, cross)


def fit_restricted(
    condition: str, warp: np.ndarray, calibration: list[dict], *, tied_channels: bool
) -> np.ndarray:
    """Fit y=a*x+b per channel, optionally tying a,b across all RGB."""
    accumulators = np.zeros((1 if tied_channels else 3, 5), dtype=np.float64)
    for item in calibration:
        source = item["src"]
        original = load_rgb(ROOT / "stylegan2_orig" / source)
        aligned = align_rgb(load_rgb(CONTROL / condition / source), warp)
        sample = (slice(16, 240, 4), slice(16, 240, 4))
        x = original[sample].reshape(-1, 3)
        y = aligned[sample].reshape(-1, 3)
        for channel in range(3):
            target = accumulators[0 if tied_channels else channel]
            target += (
                np.square(x[:, channel]).sum(),
                x[:, channel].sum(),
                len(x),
                (x[:, channel] * y[:, channel]).sum(),
                y[:, channel].sum(),
            )
    coefficient = np.zeros((4, 3), dtype=np.float64)
    for channel in range(3):
        x2, x1, n, xy, y1 = accumulators[0 if tied_channels else channel]
        slope, bias = np.linalg.solve(
            np.array([[x2, x1], [x1, n]]) + np.diag([1e-4, 0]), np.array([xy, y1])
        )
        coefficient[channel, channel] = slope
        coefficient[3, channel] = bias
    return coefficient


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def evaluate_map(
    condition: str, warp: np.ndarray, coefficient: np.ndarray, development: list[dict]
) -> list[float]:
    errors = []
    for item in development:
        source = item["src"]
        original = load_rgb(ROOT / "stylegan2_orig" / source)
        recapture = load_rgb(CONTROL / condition / source)
        aligned = align_rgb(recapture, warp)
        prediction = np.clip(
            (design(original) @ coefficient).reshape(256, 256, 3), 0, 1
        )
        inner = (slice(16, 240), slice(16, 240))
        errors.append(float(np.abs(prediction[inner] - aligned[inner]).mean()))
    return errors


def residual_diagnostics(
    condition: str, warp: np.ndarray, coefficient: np.ndarray, development: list[dict]
) -> dict:
    flat, edge, oracle_alignment, fixed_alignment = [], [], [], []
    for item in development:
        source = item["src"]
        original = load_rgb(ROOT / "stylegan2_orig" / source)
        recapture = load_rgb(CONTROL / condition / source)
        prediction = np.clip(
            (design(original) @ coefficient).reshape(256, 256, 3), 0, 1
        )
        aligned_fixed = align_rgb(recapture, warp)
        aligned_oracle = align_rgb(
            recapture, np.asarray(item["warp"], dtype=np.float32)
        )
        inner = (slice(16, 240), slice(16, 240))
        fixed_error = np.abs(prediction[inner] - aligned_fixed[inner]).mean(axis=2)
        oracle_error = np.abs(prediction[inner] - aligned_oracle[inner]).mean(axis=2)
        gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        gradient = np.hypot(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0) / 8, cv2.Sobel(gray, cv2.CV_32F, 0, 1) / 8
        )[inner]
        low, high = np.quantile(gradient, [0.25, 0.75])
        flat.append(float(fixed_error[gradient <= low].mean()))
        edge.append(float(fixed_error[gradient >= high].mean()))
        fixed_alignment.append(float(fixed_error.mean()))
        oracle_alignment.append(float(oracle_error.mean()))
    return {
        "flat_quartile_mae": summarize(flat),
        "edge_quartile_mae": summarize(edge),
        "edge_over_flat_mean_ratio": float(np.mean(edge) / np.mean(flat)),
        "fixed_geometry_mae": summarize(fixed_alignment),
        "per_pair_oracle_geometry_mae": summarize(oracle_alignment),
        "oracle_improves_count": int(
            (np.asarray(oracle_alignment) < np.asarray(fixed_alignment)).sum()
        ),
    }


def main() -> None:
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    registration = json.loads(AUDIT.read_text(encoding="utf-8"))
    if registration["pair_count"] != 720 or registration["success_count"] != 720:
        raise RuntimeError("full nonreserved registration audit required")
    results = {}
    learned = {}
    for condition in ("recap_mac", "recap_monitor"):
        calibration = [
            item
            for item in registration["records"]
            if item["condition"] == condition and item["split"] == "calibration"
        ]
        development = [
            item
            for item in registration["records"]
            if item["condition"] == condition and item["split"] == "development"
        ]
        if len(calibration) != 240 or len(development) != 120:
            raise RuntimeError("unexpected source split")
        warp = np.asarray(
            geometry["conditions"][condition]["median_calibration_affine"],
            dtype=np.float32,
        )
        coefficient = fit_channel(condition, warp, calibration)
        diagonal_coefficient = fit_restricted(
            condition, warp, calibration, tied_channels=False
        )
        tied_coefficient = fit_restricted(
            condition, warp, calibration, tied_channels=True
        )
        learned[condition] = {
            "warp": warp,
            "coefficient": coefficient,
            "calibration": calibration,
            "development": development,
        }
        raw_mae, geometry_mae, composite_mae = [], [], []
        for item in development:
            source = item["src"]
            original = load_rgb(ROOT / "stylegan2_orig" / source)
            recapture = load_rgb(CONTROL / condition / source)
            aligned = align_rgb(recapture, warp)
            prediction = np.clip(
                (design(original) @ coefficient).reshape(256, 256, 3), 0, 1
            )
            inner = (slice(16, 240), slice(16, 240))
            raw_mae.append(float(np.abs(original[inner] - recapture[inner]).mean()))
            geometry_mae.append(float(np.abs(original[inner] - aligned[inner]).mean()))
            composite_mae.append(
                float(np.abs(prediction[inner] - aligned[inner]).mean())
            )
        results[condition] = {
            "calibration_pairs": 240,
            "development_pairs": 120,
            "rgb3_plus_bias_coefficient": coefficient.tolist(),
            "diagonal_plus_bias_coefficient": diagonal_coefficient.tolist(),
            "tied_rgb_gain_bias_coefficient": tied_coefficient.tolist(),
            "development_rgb_mae": {
                "unaligned_identity": summarize(raw_mae),
                "fixed_geometry_identity_color": summarize(geometry_mae),
                "fixed_geometry_composite_color": summarize(composite_mae),
            },
            "composite_better_than_geometry_only_count": int(
                (np.asarray(composite_mae) < np.asarray(geometry_mae)).sum()
            ),
            "composite_better_than_unaligned_count": int(
                (np.asarray(composite_mae) < np.asarray(raw_mae)).sum()
            ),
            "restricted_color_development_mae": {
                "one_shared_gain_bias": summarize(
                    evaluate_map(condition, warp, tied_coefficient, development)
                ),
                "per_channel_gain_bias": summarize(
                    evaluate_map(condition, warp, diagonal_coefficient, development)
                ),
            },
        }
        leave_one_scene_out = {}
        for scene in ("cat", "church", "horse"):
            other_calibration = [
                item for item in calibration if not item["src"].startswith(f"{scene}/")
            ]
            scene_development = [
                item for item in development if item["src"].startswith(f"{scene}/")
            ]
            if len(other_calibration) != 160 or len(scene_development) != 40:
                raise RuntimeError("scene group balance changed")
            cross_scene_map = fit_channel(condition, warp, other_calibration)
            errors = evaluate_map(condition, warp, cross_scene_map, scene_development)
            leave_one_scene_out[scene] = {
                "train_sources": 160,
                "test_sources": 40,
                "test_mae": summarize(errors),
            }
        results[condition]["leave_one_scene_out"] = leave_one_scene_out
        results[condition]["residual_diagnostics"] = residual_diagnostics(
            condition, warp, coefficient, development
        )

    for condition in ("recap_mac", "recap_monitor"):
        other = "recap_monitor" if condition == "recap_mac" else "recap_mac"
        own = learned[condition]
        wrong_device_errors = evaluate_map(
            condition, own["warp"], learned[other]["coefficient"], own["development"]
        )
        results[condition]["wrong_device_color_map_development_mae"] = summarize(
            wrong_device_errors
        )
        own_errors = results[condition]["development_rgb_mae"][
            "fixed_geometry_composite_color"
        ]
        results[condition]["own_vs_wrong_device_mean_mae"] = [
            own_errors["mean"],
            float(np.mean(wrong_device_errors)),
        ]
    output = {
        "scope": "fixed post-crop geometry plus one empirical 3x3+b RGB map per device",
        "fit": "240 calibration sources per device, RGB samples every fourth pixel inside 16px border",
        "evaluation": "120 development sources per device, inner 224x224 RGB MAE, no per-image fitting",
        "warning": "composite gamma-encoded RGB response, not isolated screen emission or camera color matrix",
        "conditions": results,
    }
    OUTPUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
