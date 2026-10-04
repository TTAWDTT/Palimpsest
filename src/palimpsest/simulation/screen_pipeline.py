"""Explicit digital-display, physical-capture, publication composition.

This chain separates an image's digital placement on a display raster from
screen emission and camera capture, then applies crop/resize/encoding. The
display and camera choices are virtual until device measurements constrain
them; the pipeline does not infer missing settings from a published image.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .publication import PublicationParameters, PublicationResult, apply_publication
from .screen_capture import (
    ScreenCaptureParameters,
    ScreenCaptureResult,
    render_screen_capture,
)

_RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


@dataclass(frozen=True)
class DisplayRasterParameters:
    raster_size: tuple[int, int]  # width, height in physical display pixels
    content_xyxy: tuple[int, int, int, int] | None = None
    resampling: str = "lanczos"
    resample_space: str = "encoded_srgb"  # or linear_light
    background_rgb: tuple[int, int, int] = (0, 0, 0)
    drive_bits: int = 8

    def __post_init__(self) -> None:
        width, height = self.raster_size
        if not (
            isinstance(width, int)
            and isinstance(height, int)
            and width > 0
            and height > 0
        ):
            raise ValueError("raster_size must be positive width, height")
        if self.content_xyxy is not None:
            x0, y0, x1, y1 = self.content_xyxy
            if not (
                all(isinstance(value, int) for value in self.content_xyxy)
                and 0 <= x0 < x1 <= width
                and 0 <= y0 < y1 <= height
            ):
                raise ValueError("display content rectangle must fit the raster")
        if self.resampling not in _RESAMPLING:
            raise ValueError("unsupported display resampling filter")
        if self.resample_space not in ("encoded_srgb", "linear_light"):
            raise ValueError(
                "display resample_space must be encoded_srgb or linear_light"
            )
        if len(self.background_rgb) != 3 or not all(
            isinstance(value, int) and 0 <= value <= 255
            for value in self.background_rgb
        ):
            raise ValueError("background_rgb must contain three 8-bit values")
        if self.drive_bits not in (8, 10):
            raise ValueError("drive_bits must be 8 or 10")


@dataclass(frozen=True)
class ScreenPipelineResult:
    display_drive_rgb: np.ndarray
    capture: ScreenCaptureResult
    publication: PublicationResult


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def rasterize_display_source(
    source_rgb: np.ndarray, parameters: DisplayRasterParameters
) -> np.ndarray:
    """Place one 8-bit digital source on a display drive raster explicitly."""
    source = np.asarray(source_rgb)
    if (
        source.dtype != np.uint8
        or source.ndim != 3
        or source.shape[2] != 3
        or 0 in source.shape
    ):
        raise ValueError("display source must be a nonempty uint8 RGB image")
    width, height = parameters.raster_size
    x0, y0, x1, y1 = parameters.content_xyxy or (0, 0, width, height)
    source_values = source.astype(np.float32) / 255
    if parameters.resample_space == "linear_light":
        source_values = _srgb_to_linear(source_values)
    resized = np.empty((y1 - y0, x1 - x0, 3), dtype=np.float32)
    for channel in range(3):
        image = Image.fromarray(source_values[..., channel], mode="F")
        resized[..., channel] = np.asarray(
            image.resize((x1 - x0, y1 - y0), _RESAMPLING[parameters.resampling]),
            dtype=np.float32,
        )
    if parameters.resample_space == "linear_light":
        resized = _linear_to_srgb(resized)
    levels = (1 << parameters.drive_bits) - 1
    resized = np.rint(np.clip(resized, 0, 1) * levels).astype(np.float32) / levels
    raster = np.empty((height, width, 3), dtype=np.float32)
    raster[...] = np.asarray(parameters.background_rgb, dtype=np.float32) / 255
    raster[y0:y1, x0:x1] = resized
    return raster


def run_screen_pipeline(
    source_rgb: np.ndarray,
    display: DisplayRasterParameters,
    sensor_shape: tuple[int, int],
    camera: ScreenCaptureParameters,
    publication: PublicationParameters,
    *,
    seed: int = 0,
    spatial_method: str = "fine",
    samples_per_sensor_pixel: int | None = None,
    tile_size_sensor_pixels: int | None = None,
) -> ScreenPipelineResult:
    drive = rasterize_display_source(source_rgb, display)
    capture = render_screen_capture(
        drive,
        sensor_shape,
        camera,
        seed=seed,
        spatial_method=spatial_method,
        samples_per_sensor_pixel=samples_per_sensor_pixel,
        tile_size_sensor_pixels=tile_size_sensor_pixels,
    )
    published = apply_publication(capture.srgb, publication)
    return ScreenPipelineResult(drive, capture, published)
