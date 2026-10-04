"""Test a frozen Gaussian effective-blur proxy after geometry/color calibration.

Sigma is in 256px published-output coordinates, not a physical lens PSF.
The Gaussian operates on gamma-encoded RGB and is only a negative control.
"""

from palimpsest.paths import WORK_DIR

import json

import cv2
import numpy as np

from experiments.chimera.evaluate_chimera_composite_photometry import (
    CONTROL,
    ROOT,
    align_rgb,
    design,
    load_rgb,
    summarize,
)
from experiments.chimera.evaluate_chimera_fixed_publication_geometry import AUDIT


GEOMETRY = WORK_DIR / "chimera_fixed_publication_geometry.json"
PHOTOMETRY = WORK_DIR / "chimera_composite_photometry.json"
OUTPUT = WORK_DIR / "chimera_effective_blur_proxy.json"
SIGMAS = (0.0, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0)


def blurred(prediction: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0:
        return prediction
    return cv2.GaussianBlur(
        prediction, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT
    )


def evaluate_items(
    items: list[dict],
    condition: str,
    warp: np.ndarray,
    coefficient: np.ndarray,
    sigmas: tuple[float, ...],
) -> dict[float, list[float]]:
    errors = {sigma: [] for sigma in sigmas}
    for item in items:
        source = item["src"]
        original = load_rgb(ROOT / "stylegan2_orig" / source)
        target = align_rgb(load_rgb(CONTROL / condition / source), warp)
        prediction = np.clip(
            (design(original) @ coefficient).reshape(256, 256, 3), 0, 1
        )
        inner = (slice(16, 240), slice(16, 240))
        for sigma in sigmas:
            errors[sigma].append(
                float(np.abs(blurred(prediction, sigma)[inner] - target[inner]).mean())
            )
    return errors


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    color = json.loads(PHOTOMETRY.read_text(encoding="utf-8"))
    if audit["pair_count"] != 720:
        raise RuntimeError("expected full nonreserved registration audit")
    results = {}
    for condition in ("recap_mac", "recap_monitor"):
        calibration = [
            item
            for item in audit["records"]
            if item["condition"] == condition and item["split"] == "calibration"
        ]
        development = [
            item
            for item in audit["records"]
            if item["condition"] == condition and item["split"] == "development"
        ]
        if len(calibration) != 240 or len(development) != 120:
            raise RuntimeError("unexpected source split")
        warp = np.asarray(
            geometry["conditions"][condition]["median_calibration_affine"],
            dtype=np.float32,
        )
        coefficient = np.asarray(
            color["conditions"][condition]["rgb3_plus_bias_coefficient"],
            dtype=np.float64,
        )
        calibration_errors = evaluate_items(
            calibration, condition, warp, coefficient, SIGMAS
        )
        selected = min(SIGMAS, key=lambda sigma: np.mean(calibration_errors[sigma]))
        development_errors = evaluate_items(
            development, condition, warp, coefficient, (0.0, selected)
        )
        original = np.asarray(development_errors[0.0])
        processed = np.asarray(development_errors[selected])
        scene_results = {}
        for scene in ("cat", "church", "horse"):
            indices = [
                i
                for i, item in enumerate(development)
                if item["src"].startswith(f"{scene}/")
            ]
            if len(indices) != 40:
                raise RuntimeError("unbalanced development scene")
            scene_results[scene] = {
                "source_count": 40,
                "no_blur_mean_mae": float(original[indices].mean()),
                "selected_blur_mean_mae": float(processed[indices].mean()),
            }
        results[condition] = {
            "calibration_sources": 240,
            "development_sources": 120,
            "candidate_sigmas_output_pixels": list(SIGMAS),
            "calibration_mean_mae_by_sigma": {
                str(sigma): float(np.mean(calibration_errors[sigma]))
                for sigma in SIGMAS
            },
            "selected_sigma_output_pixels": selected,
            "development_no_blur_mae": summarize(original.tolist()),
            "development_selected_blur_mae": summarize(processed.tolist()),
            "selected_improves_count": int((processed < original).sum()),
            "development_by_scene": scene_results,
        }
    output = {
        "scope": "negative-control effective Gaussian blur after fixed published-image geometry and RGB fit",
        "warning": "gamma-encoded RGB blur after 256px downsample; sigma is not measured lens optics",
        "conditions": results,
    }
    OUTPUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
