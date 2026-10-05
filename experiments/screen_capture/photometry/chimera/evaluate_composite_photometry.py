"""Calibrate a shared RGB response after fixed post-crop geometry alignment.

The 3x3+b map is an empirical composite of display, camera, ISP and publication
effects in gamma-encoded PNG values. It must not be interpreted as a camera CCM."""

import json
import numpy as np
from experiments.screen_capture.photometry.chimera.evaluate_fixed_publication_geometry import (
    AUDIT,
    CONTROL,
    ROOT,
)

from experiments.screen_capture.photometry.chimera.protocol import (
    GEOMETRY,
    OUTPUT,
    load_rgb,
    align_rgb,
    design,
    fit_channel,
    fit_restricted,
    summarize,
    evaluate_map,
    residual_diagnostics,
)


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
