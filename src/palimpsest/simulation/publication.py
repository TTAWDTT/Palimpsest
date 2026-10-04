"""Explicit digital publication stage after any physical capture.

This stage records crop, resampling and encoding separately from printer,
display, camera and scanner parameters. It is a deterministic operator, not
an estimate of any dataset's undocumented publishing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PublicationParameters:
    crop_xyxy: tuple[int, int, int, int] | None = None
    output_size: tuple[int, int] | None = None  # width, height
    resampling: str = "lanczos"
    encoding: str = "png"
    jpeg_quality: int | None = None
    jpeg_subsampling: int | None = None  # 0=4:4:4, 1=4:2:2, 2=4:2:0

    def __post_init__(self) -> None:
        if self.resampling not in ("nearest", "bilinear", "bicubic", "lanczos"):
            raise ValueError("unsupported resampling filter")
        if self.encoding not in ("png", "jpeg"):
            raise ValueError("encoding must be png or jpeg")
        if self.crop_xyxy is not None and (
            len(self.crop_xyxy) != 4
            or not all(isinstance(v, int) for v in self.crop_xyxy)
            or self.crop_xyxy[0] < 0
            or self.crop_xyxy[1] < 0
            or self.crop_xyxy[2] <= self.crop_xyxy[0]
            or self.crop_xyxy[3] <= self.crop_xyxy[1]
        ):
            raise ValueError("crop must be a positive xyxy rectangle")
        if self.output_size is not None and (
            len(self.output_size) != 2
            or not all(isinstance(v, int) and v > 0 for v in self.output_size)
        ):
            raise ValueError("output_size must be positive width, height")
        if self.encoding == "png" and (
            self.jpeg_quality is not None or self.jpeg_subsampling is not None
        ):
            raise ValueError("JPEG settings cannot be applied to PNG")
        if self.encoding == "jpeg":
            if self.jpeg_quality is None or not 1 <= self.jpeg_quality <= 100:
                raise ValueError("JPEG requires explicit quality in [1,100]")
            if self.jpeg_subsampling not in (0, 1, 2):
                raise ValueError("JPEG requires explicit subsampling 0, 1 or 2")


@dataclass(frozen=True)
class PublicationResult:
    after_crop_rgb: np.ndarray
    after_resize_rgb: np.ndarray
    decoded_rgb: np.ndarray
    encoded_bytes: bytes


def apply_publication(
    rgb: np.ndarray, parameters: PublicationParameters
) -> PublicationResult:
    """Convert capture RGB to 8-bit, crop, resize, encode and decode in order."""
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) <= 0:
        raise ValueError("input must be nonempty HxWx3 RGB")
    if array.dtype == np.uint8:
        data = array
    else:
        data = np.asarray(array, dtype=np.float32)
        if not np.isfinite(data).all() or data.min() < 0 or data.max() > 1:
            raise ValueError("float RGB must be finite within [0,1]")
        data = np.uint8(np.rint(data * 255))
    image = Image.fromarray(data, "RGB")
    if parameters.crop_xyxy is not None:
        x0, y0, x1, y1 = parameters.crop_xyxy
        if x1 > image.width or y1 > image.height:
            raise ValueError("crop extends outside capture frame")
        image = image.crop((x0, y0, x1, y1))
    cropped = np.asarray(image).copy()
    if parameters.output_size is not None:
        filter_name = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }[parameters.resampling]
        image = image.resize(parameters.output_size, filter_name)
    resized = np.asarray(image).copy()
    stream = BytesIO()
    if parameters.encoding == "jpeg":
        image.save(
            stream,
            format="JPEG",
            quality=parameters.jpeg_quality,
            subsampling=parameters.jpeg_subsampling,
        )
    else:
        image.save(stream, format="PNG")
    data_bytes = stream.getvalue()
    with Image.open(BytesIO(data_bytes)) as decoded:
        output = np.asarray(decoded.convert("RGB")).copy()
    return PublicationResult(cropped, resized, output, data_bytes)
