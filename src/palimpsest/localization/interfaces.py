"""Implement this protocol to add a locator without changing the pipeline."""

from typing import Protocol
from palimpsest.contracts import RGBImage, Region


class RegionLocator(Protocol):
    name: str

    def locate(self, image: RGBImage) -> tuple[Region, ...]:
        """Return validated candidates, or an empty tuple when none are found."""
        ...
