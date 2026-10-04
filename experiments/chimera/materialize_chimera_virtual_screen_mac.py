"""Render a frozen, unfitted screen-camera hypothesis on Chimera development.

This tests causal stage order, not a claimed MacBook/iPhone device calibration.
The 840 reserved sources are excluded. All chosen parameters are declared here.
"""

from palimpsest.paths import DATA_ROOT, WORK_DIR

import argparse
import csv
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from palimpsest.io.provenance import simulation_code_sha256

import numpy as np
from PIL import Image

from palimpsest.simulation.capture_kit import sha256
from palimpsest.simulation.publication import PublicationParameters
from palimpsest.simulation.screen_capture import ScreenCaptureParameters
from palimpsest.simulation.screen_pipeline import (
    DisplayRasterParameters,
    run_screen_pipeline,
)


SPLIT = DATA_ROOT / "manifests/chimera_simulation_source_split.csv"
SOURCE_ROOT = DATA_ROOT / "derived/chimera_paired/stylegan2_orig"
OUTPUT_ROOT = DATA_ROOT / "derived/chimera_virtual_screen_mac_dev256"
MANIFEST = DATA_ROOT / "manifests/chimera_virtual_screen_mac_dev256_manifest.csv"
AUDIT = WORK_DIR / "chimera_virtual_screen_mac_dev256_audit.json"


def virtual_settings(emitter_layout: str):
    display = DisplayRasterParameters(
        raster_size=(1024, 1024),
        resampling="lanczos",
        resample_space="encoded_srgb",
        drive_bits=8,
    )
    camera = ScreenCaptureParameters(
        sensor_to_display=np.array(
            [
                [1024 / 1026, 0, 0.23],
                [0, 1024 / 1026, 0.37],
                [0, 0, 1],
            ]
        ),
        fill_fraction=0.85,
        emitter_layout=emitter_layout,
        display_gamma=2.2,
        optical_blur_sigma_sensor_pixels=0.55,
        exposure_electrons_per_unit=40000,
        full_well_electrons=10000,
        read_noise_electrons=2.0,
    )
    publication = PublicationParameters(
        output_size=(256, 256), resampling="lanczos", encoding="png"
    )
    return display, camera, publication


def image_seed(source: str) -> int:
    return int.from_bytes(hashlib.sha256(source.encode("utf-8")).digest()[:8], "little")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emitter-layout",
        choices=("vertical_rgb", "co_spatial_rgb_control"),
        default="vertical_rgb",
    )
    parser.add_argument(
        "--effective-fit",
        type=Path,
        help="calibration-only RGB effective exposure/blur fit JSON",
    )
    args = parser.parse_args()
    if args.effective_fit and args.emitter_layout != "vertical_rgb":
        raise ValueError("effective fit was selected for vertical RGB only")
    suffix = "" if args.emitter_layout == "vertical_rgb" else "_co_spatial"
    fit = None
    if args.effective_fit:
        fit = json.loads(args.effective_fit.read_text(encoding="utf-8"))
        if (
            fit["source_split_sha256"] != sha256(SPLIT)
            or len(fit["selected_source_ids"]) != 60
        ):
            raise RuntimeError(
                "effective fit is not bound to the frozen 60 calibration sources"
            )
        suffix += "_effective_fit"
    output_root = Path(str(OUTPUT_ROOT) + suffix)
    manifest = Path(str(MANIFEST).replace("_manifest.csv", f"{suffix}_manifest.csv"))
    audit_path = Path(str(AUDIT).replace("_audit.json", f"{suffix}_audit.json"))
    prior_hashes = None
    if audit_path.exists():
        old = json.loads(audit_path.read_text(encoding="utf-8"))
        prior_hashes = {row["filename"]: row["sha256"] for row in old["image_sha256"]}
    condition = "virtual_mac" + suffix
    with SPLIT.open(newline="", encoding="utf-8-sig") as stream:
        development = [
            row for row in csv.DictReader(stream) if row["split"] == "development"
        ]
    if len(development) != 120 or {row["label"] for row in development} != {
        "REAL",
        "FAKE",
    }:
        raise RuntimeError("expected 120 frozen development sources with both classes")
    display, camera, publication = virtual_settings(args.emitter_layout)
    if fit is not None:
        chosen = fit["selected"]
        camera = replace(
            camera,
            exposure_electrons_per_unit=chosen["exposure_electrons_per_unit"],
            optical_blur_sigma_sensor_pixels=chosen["optical_blur_sigma_sensor_pixels"],
        )
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    rows = []
    start = time.perf_counter()
    for source_row in development:
        source = source_row["src"]
        input_path = SOURCE_ROOT / source
        with Image.open(input_path) as opened:
            if opened.mode != "RGB" or opened.size != (256, 256):
                raise ValueError(f"unexpected Chimera source: {input_path}")
            original = np.asarray(opened).copy()
        result = run_screen_pipeline(
            original,
            display,
            (1026, 1026),
            camera,
            publication,
            seed=image_seed(source),
            spatial_method="analytic",
        )
        relative = Path(condition) / source
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(result.publication.encoded_bytes)
        with Image.open(destination) as check:
            check.load()
            if check.mode != "RGB" or check.size != (256, 256):
                raise RuntimeError(f"invalid published image: {destination}")
        rows.append(
            {
                "filename": relative.as_posix(),
                "src": source,
                "label": source_row["label"],
                "condition": condition,
            }
        )
        records.append({"filename": relative.as_posix(), "sha256": sha256(destination)})
    if prior_hashes is not None and prior_hashes != {
        row["filename"]: row["sha256"] for row in records
    }:
        raise RuntimeError("rerendered images differ from prior frozen audit")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("filename", "src", "label", "condition")
        )
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "scope": f"120 Chimera development sources, one fixed unfitted virtual Mac-like screen-camera hypothesis; emitter={args.emitter_layout}",
        "warning": "not a MacBook/iPhone calibration; selected scale/phase/PSF/exposure are virtual",
        "display": display.__dict__,
        "sensor_shape": [1026, 1026],
        "camera": {
            **camera.__dict__,
            "sensor_to_display": camera.sensor_to_display.tolist(),
        },
        "publication": publication.__dict__,
        "spatial_method": "analytic",
        "effective_fit_sha256": sha256(args.effective_fit)
        if args.effective_fit
        else None,
        "split_sha256": sha256(SPLIT),
        "script_sha256": sha256(Path(__file__)),
        "pipeline_code_sha256": sha256(Path("origin_simulation/screen_pipeline.py")),
        "simulation_package_sha256": simulation_code_sha256(),
        "output_root": str(output_root),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "output_count": len(records),
        "image_sha256": records,
        "render_and_write_seconds": time.perf_counter() - start,
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: audit[key]
                for key in (
                    "output_count",
                    "manifest_sha256",
                    "render_and_write_seconds",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
