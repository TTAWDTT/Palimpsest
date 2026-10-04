"""Display emitter geometry and radiance before optical filtering."""

import numpy as np

from palimpsest.simulation._screen.parameters import ScreenCaptureParameters


def _screen_radiance(
    frame: np.ndarray,
    parameters: ScreenCaptureParameters,
    display_x: np.ndarray,
    display_y: np.ndarray,
    samples_per_sensor_pixel: int,
) -> np.ndarray:
    """Integrate rectangular RGB emitters over each fine sensor cell."""
    height, width, _ = frame.shape
    base_x = np.floor(display_x).astype(np.int64)
    base_y = np.floor(display_y).astype(np.int64)
    cell_width = parameters.sensor_to_display[0, 0] / samples_per_sensor_pixel
    cell_height = parameters.sensor_to_display[1, 1] / samples_per_sensor_pixel
    left, right = display_x - cell_width / 2, display_x + cell_width / 2
    top, bottom = display_y - cell_height / 2, display_y + cell_height / 2
    gap = (1 - parameters.fill_fraction) / 2
    radiance = np.zeros(display_x.shape + (3,), dtype=np.float32)
    for y_offset in (-1, 0, 1):
        row = base_y + y_offset
        row_weight = (
            np.maximum(
                0, np.minimum(bottom, row + 1 - gap) - np.maximum(top, row + gap)
            )
            / cell_height
        )
        for x_offset in (-1, 0, 1):
            column = base_x + x_offset
            valid = (row >= 0) & (row < height) & (column >= 0) & (column < width)
            if not np.any(valid):
                continue
            safe_row = np.clip(row, 0, height - 1)
            safe_column = np.clip(column, 0, width - 1)
            for channel in range(3):
                if parameters.emitter_layout == "vertical_rgb":
                    emitter_left = column + (channel + gap) / 3
                    emitter_right = column + (channel + 1 - gap) / 3
                else:
                    emitter_left = column + gap
                    emitter_right = column + 1 - gap
                column_weight = (
                    np.maximum(
                        0,
                        np.minimum(right, emitter_right)
                        - np.maximum(left, emitter_left),
                    )
                    / cell_width
                )
                radiance[..., channel] += (
                    valid
                    * row_weight
                    * column_weight
                    * frame[safe_row, safe_column, channel]
                    * (1 if parameters.emitter_layout == "vertical_rgb" else 1 / 3)
                )
    return radiance


def _screen_radiance_projective(
    frame: np.ndarray,
    parameters: ScreenCaptureParameters,
    display_x: np.ndarray,
    display_y: np.ndarray,
) -> np.ndarray:
    """Point-quadrature of the projected rectangular RGB emitter lattice.

    Each point is one fine-grid cell center. The later pixel integration takes
    their mean. This is slower and less exact than the axis-aligned overlap
    integral, and should be checked for convergence at the chosen crop/pose.
    """
    height, width, _ = frame.shape
    x = np.floor(display_x).astype(np.int64)
    y = np.floor(display_y).astype(np.int64)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    safe_x = np.clip(x, 0, width - 1)
    safe_y = np.clip(y, 0, height - 1)
    phase_x = display_x - x
    phase_y = display_y - y
    gap = (1 - parameters.fill_fraction) / 2
    active_y = (phase_y >= gap) & (phase_y < 1 - gap) & valid
    radiance = np.zeros(display_x.shape + (3,), dtype=np.float32)
    for channel in range(3):
        if parameters.emitter_layout == "vertical_rgb":
            active_x = (phase_x >= (channel + gap) / 3) & (
                phase_x < (channel + 1 - gap) / 3
            )
            factor = 1
        else:
            active_x = (phase_x >= gap) & (phase_x < 1 - gap)
            factor = 1 / 3
        radiance[..., channel] = (
            active_y * active_x * frame[safe_y, safe_x, channel] * factor
        )
    return radiance


def _display_emitter_raster(
    emitted_frame: np.ndarray, parameters: ScreenCaptureParameters, display_samples: int
) -> np.ndarray:
    """Area coverage of rectangular RGB emitters on a display subgrid."""
    display_h, display_w, _ = emitted_frame.shape
    S = display_samples
    indices_x = np.arange(display_w * S, dtype=np.int32)
    indices_y = np.arange(display_h * S, dtype=np.int32)
    px, py = indices_x // S, indices_y // S
    fx, fy = (indices_x % S) / S, (indices_y % S) / S
    gap = (1 - parameters.fill_fraction) / 2
    vertical_coverage = S * np.maximum(
        0, np.minimum(fy + 1 / S, 1 - gap) - np.maximum(fy, gap)
    )
    raster = np.empty((display_h * S, display_w * S, 3), dtype=np.float32)
    for channel in range(3):
        if parameters.emitter_layout == "vertical_rgb":
            left, right = (channel + gap) / 3, (channel + 1 - gap) / 3
            factor = 1.0
        else:
            left, right = gap, 1 - gap
            factor = 1 / 3
        horizontal_coverage = S * np.maximum(
            0, np.minimum(fx + 1 / S, right) - np.maximum(fx, left)
        )
        raster[..., channel] = (
            emitted_frame[py[:, None], px[None, :], channel]
            * vertical_coverage[:, None]
            * horizontal_coverage[None, :]
            * factor
        )
    return raster
