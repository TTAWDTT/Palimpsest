"""Decode once and measure file-level inference without hiding decode costs."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image
from palimpsest.contracts import Prediction
from palimpsest.detection.interfaces import OriginDetector


@dataclass(frozen=True)
class FilePrediction:
    prediction: Prediction
    width: int
    height: int
    decode_ms: float
    end_to_end_ms: float


def predict_file(detector: OriginDetector, path: Path) -> FilePrediction:
    start = perf_counter()
    with Image.open(path) as loaded:
        rgb = np.asarray(loaded.convert("RGB"))
    decoded = perf_counter()
    prediction = detector.predict(rgb)
    return FilePrediction(
        prediction,
        rgb.shape[1],
        rgb.shape[0],
        (decoded - start) * 1000,
        (perf_counter() - start) * 1000,
    )
