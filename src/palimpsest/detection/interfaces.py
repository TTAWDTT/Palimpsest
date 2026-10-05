"""Shared origin API for learned models, classical algorithms and baselines."""

from typing import Protocol
from palimpsest.contracts import RGBImage, Prediction


class OriginDetector(Protocol):
    name: str

    def predict(self, image: RGBImage) -> Prediction:
        """Predict one RGB image; higher score always indicates AI content."""
        ...
