"""Contrast native recapture high-band power with pure digital resampling.

This is a process diagnostic, not screen-lattice or sensor-noise attribution.
Only frozen development source groups are read; bands and controls are fixed.
"""

import csv
import json
from pathlib import Path

import numpy as np

from probe_chimera_native_frequency import GEOMETRY, PATCH, SPLIT, native_patches


OUTPUT = Path("work/chimera_native_bandpower_probe.json")
BANDS = {"mid_015_030": (.15, .30), "high_030_045": (.30, .45)}


def powers(patch: np.ndarray, masks: dict[str, np.ndarray], window: np.ndarray) -> dict[str, float]:
    centered = patch - patch.mean()
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered * window))) ** 2
    return {name: float(np.mean(power[mask])) for name, mask in masks.items()}


def summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {"median": float(np.median(array)), "p05": float(np.quantile(array, .05)),
            "p95": float(np.quantile(array, .95))}


def main() -> None:
    with SPLIT.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "development"]
    if len(rows) != 120:
        raise RuntimeError("expected 120 frozen development groups")
    geometry = json.loads(GEOMETRY.read_text(encoding="utf-8"))
    f = np.fft.fftshift(np.fft.fftfreq(PATCH))
    fx, fy = np.meshgrid(f, f)
    radius = np.hypot(fx, fy)
    masks = {name: (radius >= low) & (radius < high)
             for name, (low, high) in BANDS.items()}
    window = np.outer(np.hanning(PATCH), np.hanning(PATCH)).astype(np.float32)
    result = {"scope": "120 development sources per device, native published resolution",
              "warning": "band energy cannot separate screen, camera, ISP and publication effects",
              "conditions": {}}
    for condition in ("recap_mac", "recap_monitor"):
        warp = np.asarray(geometry["conditions"][condition]["median_calibration_affine"],
                          dtype=np.float32)
        by_control: dict[str, dict[str, list[float]]] = {}
        by_label: dict[str, dict[str, dict[str, list[float]]]] = {}
        for row in rows:
            native, controls = native_patches(row["src"], condition, warp)
            native_power = powers(native, masks, window)
            for name, control in controls.items():
                target = by_control.setdefault(name, {band: [] for band in BANDS})
                label_target = by_label.setdefault(row["label"], {}).setdefault(
                    name, {band: [] for band in BANDS})
                control_power = powers(control, masks, window)
                for band in BANDS:
                    difference = float(10 * np.log10(
                        (native_power[band] + 1e-12) / (control_power[band] + 1e-12)))
                    target[band].append(difference)
                    label_target[band].append(difference)
        result["conditions"][condition] = {"all": {
            name: {band: {"real_minus_control_db": summary(values),
                          "real_power_exceeds_control_count": int((np.asarray(values) > 0).sum())}
                   for band, values in bands.items()}
            for name, bands in by_control.items()},
            "by_label": {label: {name: {band: summary(values) for band, values in bands.items()}
                                 for name, bands in controls.items()}
                         for label, controls in by_label.items()}}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["conditions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
