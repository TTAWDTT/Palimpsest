"""Run the explicit source-to-display-to-camera-to-publication screen chain."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from palimpsest.io.hashing import file_sha256 as sha256
from palimpsest.simulation.digital.publication import PublicationParameters
from palimpsest.simulation.screen_capture.capture import ScreenCaptureParameters
from palimpsest.simulation.screen_capture.pipeline import (
    DisplayRasterParameters,
    run_screen_pipeline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--intermediates", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("schema") != "screen-pipeline-v1":
        raise ValueError("unsupported screen pipeline schema")
    display = DisplayRasterParameters(**config["display"])
    camera_options = dict(config["camera"])
    camera_options["sensor_to_display"] = np.asarray(
        camera_options["sensor_to_display"], dtype=np.float64
    )
    camera = ScreenCaptureParameters(**camera_options)
    publication = PublicationParameters(**config["publication"])
    extension = ".jpg" if publication.encoding == "jpeg" else ".png"
    output_image = args.output_prefix.with_suffix(extension)
    output_json = args.output_prefix.with_suffix(".json")
    output_npz = args.output_prefix.with_suffix(".npz")
    if any(path.exists() for path in (output_image, output_json, output_npz)):
        raise FileExistsError("one or more output files already exist")
    with Image.open(args.input) as opened:
        if opened.mode != "RGB":
            raise ValueError("source image must be explicit RGB")
        source = np.asarray(opened).copy()
    start = time.perf_counter()
    result = run_screen_pipeline(
        source,
        display,
        tuple(config["sensor_shape"]),
        camera,
        publication,
        seed=int(config.get("seed", 0)),
        spatial_method=config.get("spatial_method", "fine"),
        samples_per_sensor_pixel=config.get("samples_per_sensor_pixel"),
        tile_size_sensor_pixels=config.get("tile_size_sensor_pixels"),
    )
    compute_seconds = time.perf_counter() - start
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_image.write_bytes(result.publication.encoded_bytes)
    if args.intermediates:
        np.savez_compressed(
            output_npz,
            display_drive_rgb=result.display_drive_rgb,
            emitter_band_irradiance=result.capture.emitter_band_irradiance,
            sensor_channel_irradiance=result.capture.irradiance,
            raw_mosaic=result.capture.raw_mosaic,
            camera_srgb=result.capture.srgb,
            published_decoded_rgb=result.publication.decoded_rgb,
        )
    report = {
        "schema": "screen-pipeline-output-v1",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "config": str(args.config),
        "config_sha256": sha256(args.config),
        "output_image": str(output_image),
        "output_image_sha256": sha256(output_image),
        "intermediates_npz": str(output_npz) if args.intermediates else None,
        "intermediates_sha256": sha256(output_npz) if args.intermediates else None,
        "output_shape": list(result.publication.decoded_rgb.shape),
        "compute_seconds": compute_seconds,
        "spatial_method": config.get("spatial_method", "fine"),
        "parameter_status": config.get("parameter_status", "unspecified"),
        "interpretation": "explicit virtual process; device and publication fidelity need independent calibration",
    }
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
