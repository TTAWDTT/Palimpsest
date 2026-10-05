"""Explicit fit boundary with source groups and separate validation examples."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol
from palimpsest.contracts import Origin, RGBImage, validate_rgb


@dataclass(frozen=True)
class TrainingExample:
    image: RGBImage
    origin: Origin
    source_group: str
    condition: str

    def __post_init__(self) -> None:
        validate_rgb(self.image)
        if (
            not isinstance(self.origin, Origin)
            or not self.source_group
            or not self.condition
        ):
            raise ValueError(
                "Training examples require original-content label, source group and condition"
            )


class OriginTrainer(Protocol):
    def fit(
        self,
        training: Iterable[TrainingExample],
        validation: Iterable[TrainingExample],
        output: Path,
    ) -> Path:
        """Fit and persist an artifact. Reject source-group overlap before fitting.

        The returned artifact must record data/code/weight fingerprints and
        score/threshold semantics. There is no default trainer yet.
        """
        ...
