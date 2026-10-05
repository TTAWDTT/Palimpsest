"""Search for repeatable native-pixel spectral peaks in true screen recaptures.

Calibration identifies a candidate frequency. Development tests that *fixed*
frequency against a geometry/resize-only digital control. A peak is not proof
of display pixels: CFA, ISP and publication sampling remain alternative causes."""

from palimpsest.evaluation.distribution import quantile_summary as median_summary
import csv
import json
import numpy as np

from experiments.screen_capture.spectral_structure.chimera.protocol import (
    SPLIT,
    GEOMETRY,
    OUTPUT,
    PATCH,
    native_patches,
    spectrum,
)


def main() -> None:
    with SPLIT.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    calibration = [row for row in rows if row["split"] == "calibration"]
    development = [row for row in rows if row["split"] == "development"]
    if len(calibration) != 240 or len(development) != 120:
        raise RuntimeError("expected frozen 240/120 nonreserved source groups")
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    frequencies = np.fft.fftshift(np.fft.fftfreq(PATCH))
    fx, fy = np.meshgrid(frequencies, frequencies)
    radius = np.hypot(fx, fy)
    valid = (radius >= 0.08) & (radius <= 0.45) & ((fx > 0) | ((fx == 0) & (fy > 0)))
    ring_indices = np.floor(radius / 0.025).astype(np.int16)
    window = np.outer(np.hanning(PATCH), np.hanning(PATCH)).astype(np.float32)
    result = {
        "scope": "native published recapture resolution; frozen source-level calibration/development",
        "negative_control": "same original digitally warped/upscaled with five interpolation-and-blur variants, quantized to 8-bit, without display or camera",
        "warning": "calibration peak is a capture/publication periodicity candidate, not screen-subpixel ground truth",
        "conditions": {},
    }
    for condition in ("recap_mac", "recap_monitor"):
        warp = np.asarray(
            geometry["conditions"][condition]["median_calibration_affine"],
            dtype=np.float32,
        )
        calibration_map = np.zeros((PATCH, PATCH), dtype=np.float64)
        cal_native_rms = []
        for row in calibration:
            real_patch, _ = native_patches(row["src"], condition, warp)
            real_map, real_rms = spectrum(real_patch, ring_indices, valid, window)
            calibration_map += real_map
            cal_native_rms.append(real_rms)
        calibration_map /= len(calibration)
        candidate_map = np.where(valid, calibration_map, -np.inf)
        iy, ix = np.unravel_index(int(np.argmax(candidate_map)), candidate_map.shape)
        candidate = (float(fx[iy, ix]), float(fy[iy, ix]))
        neighborhood = np.zeros_like(valid)
        neighborhood[max(0, iy - 1) : iy + 2, max(0, ix - 1) : ix + 2] = True
        neighborhood &= valid
        dev_native_peak = []
        dev_native_rms = []
        controls = {
            name: {"peak": [], "rms": []}
            for name in (
                "two_stage_lanczos",
                "two_stage_lanczos_blurred",
                "direct_linear",
                "direct_cubic",
                "direct_lanczos",
            )
        }
        for row in development:
            real_patch, control_patches = native_patches(row["src"], condition, warp)
            real_map, real_rms = spectrum(real_patch, ring_indices, valid, window)
            dev_native_peak.append(float(np.max(real_map[neighborhood])))
            dev_native_rms.append(real_rms)
            for name, patch in control_patches.items():
                control_map, control_rms = spectrum(patch, ring_indices, valid, window)
                controls[name]["peak"].append(float(np.max(control_map[neighborhood])))
                controls[name]["rms"].append(control_rms)
        result["conditions"][condition] = {
            "calibration_sources": len(calibration),
            "development_sources": len(development),
            "candidate_frequency_xy_cycles_per_native_pixel": candidate,
            "calibration_mean_whitened_log10_power_at_candidate": float(
                calibration_map[iy, ix]
            ),
            "calibration_native_highpass_rms": median_summary(cal_native_rms),
            "development_native_candidate_peak_log10_above_ring": median_summary(
                dev_native_peak
            ),
            "development_native_highpass_rms": median_summary(dev_native_rms),
            "digital_controls": {
                name: {
                    "candidate_peak_log10_above_ring": median_summary(measures["peak"]),
                    "highpass_rms": median_summary(measures["rms"]),
                    "native_peak_exceeds_control_count": int(
                        (
                            np.asarray(dev_native_peak) > np.asarray(measures["peak"])
                        ).sum()
                    ),
                    "native_rms_exceeds_control_count": int(
                        (np.asarray(dev_native_rms) > np.asarray(measures["rms"])).sum()
                    ),
                }
                for name, measures in controls.items()
            },
        }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["conditions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
