"""Transfer a measured display-to-camera projection into the forward model.

The checkerboard measurement maps display *pixel-boundary* coordinates to
OpenCV camera pixel-center coordinates. The renderer uses continuous sensor
coordinates with the first sensor pixel occupying [0, 1] in each dimension.
The half-pixel offset below is therefore part of the coordinate contract.
This calibrates geometry only, not display emission, optics, sensor or ISP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from origin_simulation.capture_kit import sha256


def sensor_to_display_from_measurement(
    display_to_camera: np.ndarray, crop_xyxy: tuple[int, int, int, int]
) -> np.ndarray:
    """Map continuous crop-sensor coordinates into display pixel boundaries."""
    H = np.asarray(display_to_camera, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        raise ValueError("display_to_camera must be a finite invertible 3x3 homography")
    if len(crop_xyxy) != 4 or any(isinstance(value, bool) or int(value) != value for value in crop_xyxy):
        raise ValueError("crop_xyxy must contain four integer camera pixel coordinates")
    x0, y0, x1, y1 = map(int, crop_xyxy)
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        raise ValueError("crop_xyxy must be a nonempty camera rectangle")
    sensor_continuous_to_camera_centers = np.array([
        [1.0, 0.0, x0 - .5],
        [0.0, 1.0, y0 - .5],
        [0.0, 0.0, 1.0],
    ])
    result = np.linalg.inv(H) @ sensor_continuous_to_camera_centers
    return result / result[2, 2]


def make_geometry_config(
    measurement: dict, base_config: dict, crop_xyxy: tuple[int, int, int, int],
    *, measurement_sha256: str,
) -> dict:
    """Keep virtual appearance parameters while replacing measured geometry."""
    if base_config.get("schema") != "screen-forward-config-v1":
        raise ValueError("base config has unsupported schema")
    if "display_to_camera_homography" not in measurement or "image_size_px" not in measurement:
        raise ValueError("geometry measurement lacks homography or camera image dimensions")
    width, height = measurement["image_size_px"]
    x0, y0, x1, y1 = map(int, crop_xyxy)
    if x1 > width or y1 > height:
        raise ValueError("crop exceeds measured camera image dimensions")
    if measurement.get("detected_inner_corners") != 77:
        raise ValueError("measurement has not detected the complete 77-corner pattern")
    transform = sensor_to_display_from_measurement(
        measurement["display_to_camera_homography"], crop_xyxy
    )
    config = json.loads(json.dumps(base_config))
    config["sensor_shape"] = [y1 - y0, x1 - x0]
    config["parameters"]["sensor_to_display"] = transform.tolist()
    config["parameter_status"] = "measured checkerboard geometry; appearance parameters virtual and uncalibrated"
    config["geometry_provenance"] = {
        "measurement_json_sha256": measurement_sha256,
        "camera_file": measurement.get("camera_file"),
        "camera_file_sha256": measurement.get("camera_file_sha256"),
        "crop_xyxy_camera_pixels": list(crop_xyxy),
        "median_reprojection_error_px": measurement.get("median_reprojection_error_px"),
        "coordinate_contract": "display boundaries to OpenCV camera centers; renderer centers at sensor +0.5",
    }
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-json", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--crop-xyxy", type=int, nargs=4, required=True,
                        metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--output-config", type=Path, required=True)
    args = parser.parse_args()
    if args.output_config.exists():
        raise FileExistsError(args.output_config)
    measurement = json.loads(args.geometry_json.read_text(encoding="utf-8"))
    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    output = make_geometry_config(measurement, base, tuple(args.crop_xyxy),
                                  measurement_sha256=sha256(args.geometry_json))
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_config": str(args.output_config),
                      "parameter_status": output["parameter_status"],
                      "sensor_shape": output["sensor_shape"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
