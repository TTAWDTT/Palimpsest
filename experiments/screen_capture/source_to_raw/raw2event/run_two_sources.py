"""Test the existing screen forward renderer against two known-source real RAW frames.

All optical/display settings are predeclared virtual hypotheses. The fit only
maps each sensor CFA plane's simulated irradiance to 10-bit counts using the
automobile capture; the airplane capture is a cross-content check.
"""

from __future__ import annotations

import json
from experiments.screen_capture.source_to_raw.raw2event.protocol import (
    prepare,
    render,
    direct_sample_control,
    fit_counts,
    evaluate,
    PREFIXES,
    AUDITS,
    DISPLAY_SIDE,
    BLUR_SIGMA_SENSOR_PIXELS,
    INNER_FRACTION,
    OUT,
)


def main() -> None:
    calibration, heldout = [
        prepare(prefix, audit) for prefix, audit in zip(PREFIXES, AUDITS)
    ]
    result = {
        "scope": "two known CIFAR sources and first-frame raw mosaics; one calibration, one heldout",
        "fixed_assumptions": {
            "display_side_pixels": DISPLAY_SIDE,
            "display_resampling": "Lanczos in encoded sRGB",
            "display_gamma": 2.2,
            "fill_fraction": 0.85,
            "optical_blur_sigma_sensor_pixels": BLUR_SIGMA_SENSOR_PIXELS,
            "raw_cfa_phase": "RGGB hypothesis, not verified",
            "geometry": "manual RGB content corners + per-recording AprilTag RAW/RGB homography",
            "roi_inner_fraction": INNER_FRACTION,
        },
        "conditions": {},
    }
    for layout in (
        "vertical_rgb",
        "co_spatial_rgb_control",
        "simple_rgb_sample_control",
    ):
        print(f"rendering {layout} automobile", flush=True)
        cal_sim, cal_seconds = (
            direct_sample_control(calibration)
            if layout == "simple_rgb_sample_control"
            else render(calibration, layout)
        )
        print(f"rendering {layout} airplane", flush=True)
        test_sim, test_seconds = (
            direct_sample_control(heldout)
            if layout == "simple_rgb_sample_control"
            else render(heldout, layout)
        )
        weights = fit_counts(cal_sim, calibration["actual"], calibration["mask"])
        result["conditions"][layout] = {
            "counts_fit": {
                "intercept": float(weights[0]),
                "r_gain": float(weights[1]),
                "g_gain": float(weights[2]),
                "b_gain": float(weights[3]),
            },
            "calibration": evaluate(
                cal_sim, calibration["actual"], calibration["mask"], weights
            ),
            "heldout": evaluate(test_sim, heldout["actual"], heldout["mask"], weights),
            "render_seconds": {"automobile": cal_seconds, "airplane": test_seconds},
        }
    result["rois"] = {
        item["prefix"]: {
            "xyxy": [int(value) for value in item["roi"]],
            "raw_content_corners": item["raw_corners"].tolist(),
            "interior_n": int(item["mask"].sum()),
        }
        for item in (calibration, heldout)
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["conditions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
