"""A direct region-to-origin chain with separately measured stage costs."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image
from palimpsest.contracts import Box, Prediction, RGBImage, Region, validate_rgb
from palimpsest.detection.interfaces import OriginDetector
from palimpsest.localization.interfaces import RegionLocator


@dataclass(frozen=True)
class RegionPrediction:
    region: Region
    prediction: Prediction
    crop_ms: float
    detection_ms: float
    crop_policy: str = "unwarped_bounding_box_including_background"


@dataclass(frozen=True)
class ImageResult:
    width: int
    height: int
    mode: str
    regions: tuple[RegionPrediction, ...]
    timing_ms: dict[str, float]

    def to_dict(self) -> dict:
        """JSON summary. Full masks remain available on the Python region objects."""
        return {
            "image_size": [self.width, self.height],
            "mode": self.mode,
            "timing_ms": self.timing_ms,
            "mask_representation": "source-sized bool in Python API; JSON contains area only",
            "regions": [
                {
                    "region_id": r.region.region_id,
                    "box_xyxy": r.region.box.xyxy,
                    "kind": r.region.kind,
                    "localization_confidence": r.region.confidence,
                    "mask_area": int(r.region.mask.sum())
                    if r.region.mask is not None
                    else None,
                    "crop_policy": r.crop_policy,
                    "origin": r.prediction.origin.value,
                    "score": r.prediction.score,
                    "score_kind": r.prediction.score_kind,
                    "threshold": r.prediction.threshold,
                    "ai_probability": r.prediction.ai_probability,
                    "method": r.prediction.method,
                    "metadata": dict(r.prediction.metadata),
                    "timing_ms": {
                        "crop": r.crop_ms,
                        "detection": r.detection_ms,
                        **r.prediction.timing_ms,
                    },
                }
                for r in self.regions
            ],
        }


class ImageOriginPipeline:
    def __init__(self, detector: OriginDetector, locator: RegionLocator | None = None):
        """locator=None explicitly selects whole-image mode, not a locator fallback."""
        self.detector = detector
        self.locator = locator

    def predict(self, image: RGBImage) -> ImageResult:
        validate_rgb(image)
        start = perf_counter()
        height, width = image.shape[:2]
        regions = (
            self.locator.locate(image)
            if self.locator
            else (Region("whole_image", Box(0, 0, width, height), "whole_image"),)
        )
        localized = perf_counter()
        ids = set()
        results = []
        for region in regions:
            region.validate_shape((height, width))
            if region.region_id in ids:
                raise ValueError("Locator returned duplicate region IDs")
            ids.add(region.region_id)
            crop_start = perf_counter()
            x0, y0, x1, y1 = region.box.xyxy
            # No warping/masking: preserve source pixels. Baselines were trained
            # on whole images; ROI distribution shift remains a research question.
            crop = np.ascontiguousarray(image[y0:y1, x0:x1])
            cropped = perf_counter()
            prediction = self.detector.predict(crop)
            results.append(
                RegionPrediction(
                    region,
                    prediction,
                    (cropped - crop_start) * 1000,
                    (perf_counter() - cropped) * 1000,
                )
            )
        end = perf_counter()
        return ImageResult(
            width,
            height,
            self.locator.name if self.locator else "whole_image",
            tuple(results),
            {
                "decode": 0.0,
                "localization": (localized - start) * 1000,
                "crop": sum(r.crop_ms for r in results),
                "detection": sum(r.detection_ms for r in results),
                "end_to_end": (end - start) * 1000,
            },
        )

    def predict_file(self, path: Path) -> ImageResult:
        start = perf_counter()
        with Image.open(path) as loaded:
            image = np.asarray(loaded.convert("RGB"))
        decoded = perf_counter()
        result = self.predict(image)
        result.timing_ms["decode"] = (decoded - start) * 1000
        result.timing_ms["end_to_end"] = (perf_counter() - start) * 1000
        return result
