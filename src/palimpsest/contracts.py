"""Shared contracts for region localization and original-content detection.

Images are decoded uint8 HxWx3 RGB, in stored pixel orientation (no automatic
EXIF rotation). Coordinates are integer pixels, xyxy, right/bottom exclusive.
AI photographed from a screen still has AI original content.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

RGBImage = NDArray[np.uint8]


def validate_rgb(image: RGBImage) -> None:
    if (
        not isinstance(image, np.ndarray)
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape[2] != 3
        or min(image.shape[:2]) == 0
    ):
        raise ValueError("Expected a nonempty uint8 HxWx3 RGB image")


class Origin(str, Enum):
    NATURAL = "natural"
    AI = "ai"


@dataclass(frozen=True)
class Box:
    """Image pixel bounds; invalid/out-of-frame boxes are rejected, not clipped."""

    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if any(type(v) is not int for v in self.xyxy):
            raise ValueError("Box coordinates must be Python integers")
        if min(self.x0, self.y0) < 0 or self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("Invalid xyxy box")

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1

    @property
    def area(self) -> int:
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def validate_shape(self, shape: tuple[int, int]) -> None:
        height, width = shape
        if self.x1 > width or self.y1 > height:
            raise ValueError("Box extends beyond the source image")


@dataclass(frozen=True)
class Region:
    """Candidate area, not an origin judgement or a guaranteed image surface.

    mask, when present, is a full-source-size boolean array. confidence is
    localization quality only; it is never an AI probability.
    """

    region_id: str
    box: Box
    kind: str
    confidence: float | None = None
    mask: NDArray[np.bool_] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.region_id or not self.kind:
            raise ValueError("Region requires an id and candidate kind")
        if self.confidence is not None and (
            not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1
        ):
            raise ValueError("Localization confidence must be in [0,1]")

    def validate_shape(self, shape: tuple[int, int]) -> None:
        self.box.validate_shape(shape)
        if self.mask is not None:
            if self.mask.shape != shape or self.mask.dtype != np.bool_:
                raise ValueError(
                    "Region mask must be full-frame bool with matching shape"
                )
            y, x = np.nonzero(self.mask)
            if not len(x) or (
                x.min() < self.box.x0
                or x.max() >= self.box.x1
                or y.min() < self.box.y0
                or y.max() >= self.box.y1
            ):
                raise ValueError("Mask must be nonempty and contained in its box")


@dataclass(frozen=True)
class Prediction:
    """Higher score means more AI evidence. Strict score>threshold selects AI.

    A raw logit/margin is not a calibrated probability. Only explicitly
    calibrated implementations may provide ai_probability. Timings exclude
    file decode, localization and crop, which the pipeline measures itself.
    """

    method: str
    score: float
    threshold: float = 0.0
    score_kind: str = "logit"
    ai_probability: float | None = None
    timing_ms: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.method or not self.score_kind:
            raise ValueError("Prediction requires method and score semantics")
        if not math.isfinite(self.score) or not math.isfinite(self.threshold):
            raise ValueError("Score and threshold must be finite")
        if self.ai_probability is not None and (
            not math.isfinite(self.ai_probability) or not 0 <= self.ai_probability <= 1
        ):
            raise ValueError("AI probability must be in [0,1]")
        if any(not math.isfinite(v) or v < 0 for v in self.timing_ms.values()):
            raise ValueError("Timings must be finite and nonnegative")

    @property
    def origin(self) -> Origin:
        return Origin.AI if self.score > self.threshold else Origin.NATURAL

    @property
    def margin(self) -> float:
        return self.score - self.threshold
