"""Coordinate preflight before any real checkerboard capture is available."""

import numpy as np
import pytest

from origin_simulation.geometry_bridge import make_geometry_config, sensor_to_display_from_measurement


def test_measured_homography_maps_crop_pixel_centers_to_display_boundaries():
    display_to_camera = np.array([
        [1.1, .08, 50], [-.03, .9, 40], [.0003, -.0001, 1],
    ])
    crop = (100, 80, 140, 110)
    sensor_to_display = sensor_to_display_from_measurement(display_to_camera, crop)
    for col, row in ((0, 0), (21, 15), (39, 29)):
        sensor_center = np.array([col + .5, row + .5, 1])
        display = sensor_to_display @ sensor_center
        display /= display[2]
        projected_camera = display_to_camera @ display
        projected_camera /= projected_camera[2]
        np.testing.assert_allclose(projected_camera[:2], [crop[0] + col, crop[1] + row], atol=1e-10)


def test_bridge_marks_only_geometry_as_measured_and_checks_crop():
    measurement = {
        "display_to_camera_homography": np.eye(3).tolist(),
        "image_size_px": [200, 100], "detected_inner_corners": 77,
        "camera_file_sha256": "example", "median_reprojection_error_px": .3,
    }
    base = {
        "schema": "screen-forward-config-v1", "sensor_shape": [16, 16],
        "samples_per_sensor_pixel": 24,
        "parameters": {"sensor_to_display": np.eye(3).tolist(), "fill_fraction": .85},
    }
    converted = make_geometry_config(measurement, base, (20, 30, 60, 70), measurement_sha256="measured")
    assert converted["sensor_shape"] == [40, 40]
    assert "physical panel pixel mapping and appearance uncalibrated" in converted["parameter_status"]
    assert converted["geometry_provenance"]["crop_xyxy_camera_pixels"] == [20, 30, 60, 70]
    assert "samples_per_sensor_pixel" not in converted
    assert base["sensor_shape"] == [16, 16]
    with pytest.raises(ValueError, match="exceeds"):
        make_geometry_config(measurement, base, (20, 30, 220, 70), measurement_sha256="measured")
    with pytest.raises(ValueError, match="horizon"):
        sensor_to_display_from_measurement(
            np.array([[1, 0, 0], [0, 1, 0], [.1, 0, 1.]]), (0, 0, 20, 10)
        )
