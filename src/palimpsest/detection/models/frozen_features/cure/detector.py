"""File-only CuRe quantile detector: one frozen forward and a fitted readout.

Keep the filename because the released CuRe preprocessing has a PNG branch.
This research interface does not silently turn an RGB crop into a JPEG query.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from palimpsest.contracts import Prediction
from palimpsest.detection.representations.frozen_cure_quantiles import (
    FEATURE_NAMES,
    FrozenCureQuantiles,
)
from .quantile_score_consistency import QuantileScoreRule


@dataclass(frozen=True)
class CureReadoutFilePrediction:
    prediction: Prediction
    feature_extraction_ms: float
    readout_ms: float
    end_to_end_ms: float


class CureQuantileDetector:
    """Compose an explicitly loaded encoder and the saved five-score rule.

    Construction accepts an owned extractor. Importing this module never loads
    Torch, vendor code or weights. The raw score is not an AI probability.
    """

    name = "cure-frozen-quantile-readout"

    def __init__(self, extractor: FrozenCureQuantiles, rule: QuantileScoreRule):
        if rule.bank.feature_names != FEATURE_NAMES:
            raise ValueError(
                "CuRe quantile detector requires the complete 3920-feature schema"
            )
        self.extractor = extractor
        self.rule = rule
        self.rule_sha256 = sha256(
            json.dumps(rule.to_payload(), sort_keys=True).encode()
        ).hexdigest()

    @classmethod
    def from_paths(
        cls,
        rule_path: Path,
        vendor: Path,
        adapter: Path,
        base: Path,
        *,
        source_pins,
        device="cuda:0",
    ):
        rule = QuantileScoreRule.load(rule_path)
        if rule.bank.feature_names != FEATURE_NAMES:
            raise ValueError(
                "CuRe quantile detector requires the complete 3920-feature schema"
            )
        extractor = FrozenCureQuantiles(
            vendor, adapter, base, source_pins=source_pins, device=device
        )
        return cls(extractor, rule)

    def predict_file(self, path: Path) -> CureReadoutFilePrediction:
        start = perf_counter()
        features = self.extractor.extract_file(Path(path))
        ready = perf_counter()
        values = np.asarray(features.values, dtype=float)
        if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
            raise ValueError("Invalid complete CuRe quantile descriptor")
        score = float(self.rule.score(values[None])[0])
        end = perf_counter()
        readout_ms = 1000 * (end - ready)
        prediction = Prediction(
            self.name,
            score,
            self.rule.threshold,
            "statistical_score",
            timing_ms={"readout_ms": readout_ms},
            metadata={
                "rule_sha256": self.rule_sha256,
                "representation": "frozen CuRe; includes a neural network",
                "validation": "development candidate; independent real-data acceptance pending",
            },
        )
        return CureReadoutFilePrediction(
            prediction, 1000 * (ready - start), readout_ms, 1000 * (end - start)
        )

    def close(self):
        """Release the owned extractor's token hooks."""
        self.extractor.close()
