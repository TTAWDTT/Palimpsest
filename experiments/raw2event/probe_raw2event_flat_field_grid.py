"""Exploratory LCD-grid spectral check in blank top bands of real RAW frames.

This does not infer a panel raster from a single peak. CFA, compression,
display grid, fixed-pattern noise and readout artifacts can share frequencies.
"""

from palimpsest.paths import WORK_DIR

import json

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

from experiments.raw2event.audit_raw2event_probe import ROOT, extract_frame


PREFIXES = (
    "252_airplane_1_3332_20251222_124809",
    "12290_bird_1_3440_20251224_182520",
    "18536_cat_1_6014_20251225_154044",
    "48083_ship_1_1702_20260117_173643",
    "54366_truck_1_4150_20260121_043143",
    "10034_automobile_5_1356_20251224_110057",
    "48340_ship_1_4036_20260117_195215",
)
OUT = WORK_DIR / "raw2event_flat_field_grid_probe.json"


def peaks(patch: np.ndarray) -> dict:
    patch = patch.astype(np.float32)
    residual = patch - gaussian_filter(patch, sigma=(4, 5), mode="reflect")
    window = np.hanning(patch.shape[1]).astype(np.float32)
    spectrum = (np.abs(np.fft.rfft(residual * window[None, :], axis=1)) ** 2).mean(
        axis=0
    )
    frequency = np.fft.rfftfreq(patch.shape[1])
    valid = (frequency > 0.03) & (frequency < 0.46)
    median_power = float(np.median(spectrum[valid]))
    candidates, _ = find_peaks(spectrum)
    candidates = [int(index) for index in candidates if valid[index]]
    chosen = sorted(candidates, key=lambda index: spectrum[index], reverse=True)[:5]
    return {
        "shape": list(patch.shape),
        "residual_std_counts": float(residual.std()),
        "top_horizontal_cycles_per_plane_pixel": [
            {
                "frequency": float(frequency[index]),
                "peak_to_median_power": float(spectrum[index] / median_power),
            }
            for index in chosen
        ],
    }


def main() -> None:
    records = []
    for prefix in PREFIXES:
        raw = extract_frame(
            ROOT / "frames_raw" / f"{prefix}.mkv", 0, "gray16le", 1, "<u2"
        )
        # The first 112 sensor rows and center x span lie above the source/Tag
        # for the inspected first frames; no per-source frequency tuning.
        patch = raw[8:112, 32:660]
        channels = {
            f"phase_{dy}_{dx}": peaks(patch[dy::2, dx::2])
            for dy in range(2)
            for dx in range(2)
        }
        records.append(
            {
                "prefix": prefix,
                "raw_patch_yx": [8, 112, 32, 660],
                "cfa_planes": channels,
            }
        )
        print(
            prefix,
            {
                name: round(
                    value["top_horizontal_cycles_per_plane_pixel"][0]["frequency"], 4
                )
                for name, value in channels.items()
            },
            flush=True,
        )
    report = {
        "scope": "exploratory first-frame horizontal spectrum in fixed blank top ROI",
        "n": len(records),
        "status": "no physical display pixel pitch assigned; spectrum may mix CFA, encoder and panel",
        "records": records,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
