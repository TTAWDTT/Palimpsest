"""Fit only effective exposure/blur to real RGB observables on calibration IDs.

The selected settings remain confounded with display placement, ISP and crop;
this script never uses development/reserved pixels or B-Free scores to fit.
"""

import csv
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from experiments.paths import simulation_code_sha256

import numpy as np
from PIL import Image
from scipy.ndimage import sobel

from origin_simulation.capture_kit import sha256
from origin_simulation.screen_pipeline import run_screen_pipeline
from experiments.chimera.materialize_chimera_virtual_screen_mac import (
    SPLIT,
    SOURCE_ROOT,
    virtual_settings,
    image_seed,
)
from experiments.chimera.probe_chimera_screen_observables import CONTROL


OUTPUT = Path("work/chimera_virtual_screen_mac_effective_fit.json")
EXPOSURES = (40000, 55000, 70000)
SIGMAS = (0.55, 2.5, 4.5)
NORMALIZERS = {
    "luma_mean": 0.05,
    "gradient_mean": 0.005,
    "high_to_mid_spectral_power": 0.04,
}


def choose_calibration() -> list[dict[str, str]]:
    with SPLIT.open(newline="", encoding="utf-8-sig") as stream:
        calibration = [
            row for row in csv.DictReader(stream) if row["split"] == "calibration"
        ]
    if len(calibration) != 240:
        raise RuntimeError("expected 240 frozen calibration sources")
    groups = {}
    for row in calibration:
        groups.setdefault((row["scene"], row["label"]), []).append(row)
    if len(groups) != 6 or any(len(group) != 40 for group in groups.values()):
        raise RuntimeError("calibration groups differ from the frozen 6x40 design")
    selected = []
    for key in sorted(groups):
        ranked = sorted(
            groups[key],
            key=lambda item: hashlib.sha256(
                ("effective_screen_fit_20260925/" + item["src"]).encode()
            ).hexdigest(),
        )
        selected.extend(ranked[:10])
    return selected


def observable(rgb8: np.ndarray) -> dict[str, float]:
    rgb = rgb8.astype(np.float32) / 255
    luma = np.tensordot(
        rgb, np.array([0.2126, 0.7152, 0.0722], dtype=np.float32), axes=1
    )
    gradient = np.hypot(sobel(luma, axis=0) / 8, sobel(luma, axis=1) / 8)
    window = np.outer(np.hanning(256), np.hanning(256)).astype(np.float32)
    power = np.abs(np.fft.fftshift(np.fft.fft2((luma - luma.mean()) * window))) ** 2
    frequencies = np.fft.fftshift(np.fft.fftfreq(256))
    radius = np.hypot(frequencies[:, None], frequencies[None, :])
    mid = float(power[(radius >= 0.05) & (radius < 0.25)].sum())
    high = float(power[(radius >= 0.25) & (radius < 0.45)].sum())
    return {
        "luma_mean": float(luma.mean()),
        "gradient_mean": float(gradient.mean()),
        "high_to_mid_spectral_power": high / max(mid, 1e-12),
    }


def main() -> None:
    selected = choose_calibration()
    originals = {}
    targets = {}
    for row in selected:
        source = row["src"]
        with Image.open(SOURCE_ROOT / source) as opened:
            originals[source] = np.asarray(opened.convert("RGB")).copy()
        with Image.open(CONTROL / "recap_mac" / source) as opened:
            target = np.asarray(opened.convert("RGB")).copy()
        if target.shape != (256, 256, 3):
            raise RuntimeError("real recapture control is not 256x256")
        targets[source] = observable(target)
    display, base_camera, publication = virtual_settings("vertical_rgb")
    records = []
    start = time.perf_counter()
    for exposure in EXPOSURES:
        for sigma in SIGMAS:
            camera = replace(
                base_camera,
                exposure_electrons_per_unit=exposure,
                optical_blur_sigma_sensor_pixels=sigma,
            )
            errors = {key: [] for key in NORMALIZERS}
            shifts = {key: [] for key in NORMALIZERS}
            for row in selected:
                source = row["src"]
                simulated = run_screen_pipeline(
                    originals[source],
                    display,
                    (1026, 1026),
                    camera,
                    publication,
                    seed=image_seed(source),
                    spatial_method="analytic",
                )
                observed = observable(simulated.publication.decoded_rgb)
                for name in NORMALIZERS:
                    difference = observed[name] - targets[source][name]
                    errors[name].append(abs(difference))
                    shifts[name].append(difference)
            median_absolute_errors = {
                name: float(np.median(values)) for name, values in errors.items()
            }
            median_signed_errors = {
                name: float(np.median(values)) for name, values in shifts.items()
            }
            objective = float(
                sum(
                    median_absolute_errors[name] / NORMALIZERS[name]
                    for name in NORMALIZERS
                )
            )
            record = {
                "exposure_electrons_per_unit": exposure,
                "optical_blur_sigma_sensor_pixels": sigma,
                "objective": objective,
                "median_absolute_error": median_absolute_errors,
                "median_signed_error": median_signed_errors,
            }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    selected_record = min(records, key=lambda item: item["objective"])
    result = {
        "scope": "60 class-and-scene balanced Chimera calibration sources only; published RGB effective exposure/blur fit",
        "warning": "parameter pair is not camera exposure or lens PSF identification; no development/reserved images or B-Free scores used for selection",
        "selected_source_ids": [row["src"] for row in selected],
        "source_split_sha256": sha256(SPLIT),
        "script_sha256": sha256(Path(__file__)),
        "display_pipeline_sha256": sha256(Path("origin_simulation/screen_pipeline.py")),
        "simulation_package_sha256": simulation_code_sha256(),
        "objective": "sum over three published-RGB feature median absolute errors divided by fixed normalizers",
        "normalizers": NORMALIZERS,
        "grid": records,
        "selected": selected_record,
        "elapsed_seconds": time.perf_counter() - start,
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"selected": selected_record, "elapsed_seconds": result["elapsed_seconds"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
