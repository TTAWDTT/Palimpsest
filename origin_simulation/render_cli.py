"""Reproducible single-image entry point for the uncalibrated screen model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from origin_simulation.capture_kit import sha256
from origin_simulation.screen_capture import ScreenCaptureParameters, render_screen_capture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="exact digital display frame PNG")
    parser.add_argument("--config", type=Path, required=True, help="virtual or calibrated parameter JSON")
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--intermediates", action="store_true", help="save float32 irradiance/RAW arrays")
    args = parser.parse_args()
    prefix = args.output_prefix
    png_path = prefix.with_suffix(".png")
    json_path = prefix.with_suffix(".json")
    npz_path = prefix.with_suffix(".npz")
    if any(path.exists() for path in (png_path, json_path, npz_path)):
        raise FileExistsError("one or more output files already exist; choose a new prefix")
    configuration = json.loads(args.config.read_text(encoding="utf-8"))
    if configuration.get("schema") != "screen-forward-config-v1":
        raise ValueError("unsupported screen simulation config schema")
    sensor_shape = tuple(configuration["sensor_shape"])
    kwargs = dict(configuration["parameters"])
    kwargs["sensor_to_display"] = np.asarray(kwargs["sensor_to_display"], dtype=np.float64)
    setup = ScreenCaptureParameters(**kwargs)
    with Image.open(args.input) as source:
        if source.mode != "RGB":
            raise ValueError("input frame must be RGB; convert explicitly and record that transform")
        frame = np.asarray(source, dtype=np.float32) / 255
    start = time.perf_counter()
    result = render_screen_capture(
        frame, sensor_shape, setup, seed=int(configuration.get("seed", 0)),
        samples_per_sensor_pixel=configuration.get("samples_per_sensor_pixel"),
        tile_size_sensor_pixels=configuration.get("tile_size_sensor_pixels"),
    )
    elapsed = time.perf_counter() - start
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(result.srgb * 255).clip(0, 255).astype(np.uint8), mode="RGB").save(png_path)
    if args.intermediates:
        np.savez_compressed(npz_path, irradiance=result.irradiance,
                            noiseless_mosaic=result.noiseless_mosaic,
                            raw_mosaic=result.raw_mosaic,
                            srgb_before_sharpen=result.srgb_before_sharpen,
                            row_exposure_gain=result.row_exposure_gain)
    report = {
        "schema": "screen-forward-output-v1",
        "input_file": str(args.input), "input_sha256": sha256(args.input),
        "config_file": str(args.config), "config_sha256": sha256(args.config),
        "output_png": str(png_path), "output_png_sha256": sha256(png_path),
        "intermediates_npz": str(npz_path) if args.intermediates else None,
        "intermediates_npz_sha256": sha256(npz_path) if args.intermediates else None,
        "sensor_shape": list(sensor_shape), "seconds_render_only": elapsed,
        "parameter_status": configuration.get("parameter_status", "unspecified"),
        "geometry_provenance": configuration.get("geometry_provenance"),
        "interpretation": "forward-model output; physical fidelity requires device calibration and held-out real captures",
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
