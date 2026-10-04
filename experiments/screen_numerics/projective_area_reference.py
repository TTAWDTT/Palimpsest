"""Slow exact no-blur pixel-area reference for projective screen emitters.

This computes the sensor-area intersection of each display emitter's inverse
homography polygon and each sensor pixel. It is a numerical reference for
small virtual scenes, not an optimized production renderer or optical model.
"""

import numpy as np


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.c_[points, np.ones(len(points))]
    projected = homogeneous @ matrix.T
    if np.any(np.abs(projected[:, 2]) < 1e-12):
        raise ValueError("polygon crosses the homography horizon")
    return projected[:, :2] / projected[:, 2:3]


def clip_polygon(
    polygon: np.ndarray, axis: int, threshold: float, greater: bool
) -> np.ndarray:
    if len(polygon) == 0:
        return polygon
    output = []
    previous = polygon[-1]
    previous_in = (
        previous[axis] >= threshold if greater else previous[axis] <= threshold
    )
    for current in polygon:
        current_in = (
            current[axis] >= threshold if greater else current[axis] <= threshold
        )
        if current_in != previous_in:
            fraction = (threshold - previous[axis]) / (current[axis] - previous[axis])
            output.append(previous + fraction * (current - previous))
        if current_in:
            output.append(current)
        previous, previous_in = current, current_in
    return np.asarray(output, dtype=np.float64).reshape(-1, 2)


def rectangle_overlap_area(polygon: np.ndarray, x: int, y: int) -> float:
    for axis, threshold, greater in (
        (0, x, True),
        (0, x + 1, False),
        (1, y, True),
        (1, y + 1, False),
    ):
        polygon = clip_polygon(polygon, axis, threshold, greater)
        if len(polygon) == 0:
            return 0.0
    return float(
        0.5
        * abs(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
    )


def projective_area_reference(
    emitted_frame: np.ndarray,
    sensor_shape: tuple[int, int],
    sensor_to_display: np.ndarray,
    fill_fraction: float,
) -> np.ndarray:
    """Exact geometric area integral under a homography with zero optical PSF.

    Rectangular RGB subpixel emitters are constant in display coordinates.
    Under the inverse homography each emitter is a quadrilateral in sensor
    coordinates. Clipping that quadrilateral to the unit sensor pixel gives
    its contribution without point samples or resampling filters.
    """
    display_h, display_w, channels = emitted_frame.shape
    if channels != 3 or not 0 < fill_fraction <= 1:
        raise ValueError("expected RGB display frame and positive fill")
    inverse = np.linalg.inv(sensor_to_display)
    result = np.zeros(sensor_shape + (3,), dtype=np.float64)
    gap = (1 - fill_fraction) / 2
    for sensor_y in range(sensor_shape[0]):
        for sensor_x in range(sensor_shape[1]):
            sensor_corners = np.array(
                [
                    [sensor_x, sensor_y],
                    [sensor_x + 1, sensor_y],
                    [sensor_x + 1, sensor_y + 1],
                    [sensor_x, sensor_y + 1],
                ],
                dtype=np.float64,
            )
            display_corners = transform_points(sensor_to_display, sensor_corners)
            first_x = max(0, int(np.floor(display_corners[:, 0].min())) - 1)
            last_x = min(display_w, int(np.ceil(display_corners[:, 0].max())) + 1)
            first_y = max(0, int(np.floor(display_corners[:, 1].min())) - 1)
            last_y = min(display_h, int(np.ceil(display_corners[:, 1].max())) + 1)
            for display_y in range(first_y, last_y):
                for display_x in range(first_x, last_x):
                    for channel in range(3):
                        value = emitted_frame[display_y, display_x, channel]
                        if value == 0:
                            continue
                        left = display_x + (channel + gap) / 3
                        right = display_x + (channel + 1 - gap) / 3
                        top = display_y + gap
                        bottom = display_y + 1 - gap
                        emitter_corners = np.array(
                            [
                                [left, top],
                                [right, top],
                                [right, bottom],
                                [left, bottom],
                            ],
                            dtype=np.float64,
                        )
                        sensor_polygon = transform_points(inverse, emitter_corners)
                        area = rectangle_overlap_area(
                            sensor_polygon, sensor_x, sensor_y
                        )
                        result[sensor_y, sensor_x, channel] += value * area
    return result.astype(np.float32)
