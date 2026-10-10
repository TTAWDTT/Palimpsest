"""File identity, strict threshold, complete schema and old JSON portability."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from palimpsest.contracts import Origin
from palimpsest.detection.algorithms.csp_readout import CSPMap, CSPRule
from palimpsest.detection.algorithms.readouts.stable_rule import StableRule
from palimpsest.detection.models.frozen_features.cure.detector import (
    CureQuantileDetector,
)
from palimpsest.detection.models.frozen_features.cure.quantile_score_consistency import (
    QuantileScoreBank,
    QuantileScoreRule,
    TERM_NAMES,
)
from palimpsest.detection.models.frozen_features.cure.semantic_score_margin import (
    SemanticScoreBank,
)
from palimpsest.detection.models.frozen_features.cure.source_score_subspace import (
    DiscriminantBank,
)
from palimpsest.detection.representations.frozen_cure_quantiles import FEATURE_NAMES


def fixed_rule(names=FEATURE_NAMES):
    def linear(fields, active=None):
        weights = tuple(1.0 if i == active else 0.0 for i in range(len(fields)))
        return StableRule(
            tuple(fields),
            (0.0,) * len(fields),
            (1.0,) * len(fields),
            weights,
            0.0,
            0.0,
            0.01,
        )

    old_names = names[1024:3600]
    mapper = CSPMap(old_names, 32, tuple(tuple(row) for row in np.eye(32, 1)))
    prefix = old_names[: mapper.prefix_size]
    old = DiscriminantBank(
        old_names,
        linear(prefix),
        linear(prefix),
        CSPRule(mapper, linear(mapper.output_names)),
    )
    legacy = SemanticScoreBank(names[:3600], old, linear(names[:1024], 0))
    bank = QuantileScoreBank(names, legacy, linear(names[3600:]))
    return QuantileScoreRule(
        bank, (0.0,) * 5, (1.0,) * 5, replace(linear(TERM_NAMES, 3), threshold=0.25)
    )


class Extractor:
    def __init__(self, values):
        self.values = values
        self.calls = []
        self.closed = False

    def extract_file(self, path):
        self.calls.append(path)
        return SimpleNamespace(values=self.values, probability_fake=1.0)

    def close(self):
        self.closed = True


def test_file_prediction_matches_saved_rule_and_keeps_ties_natural(tmp_path):
    rule = fixed_rule()
    path = tmp_path / "rule.json"
    rule.save(path)
    restored = QuantileScoreRule.load(path)
    values = np.zeros(len(FEATURE_NAMES))
    values[0] = 0.25
    extractor = Extractor(values)
    detector = CureQuantileDetector(extractor, restored)
    query = Path("unmodified query.PNG")
    result = detector.predict_file(query)
    assert extractor.calls == [query]  # one extraction, original suffix preserved
    assert result.prediction.score == float(rule.score(values[None])[0]) == 0.25
    assert result.prediction.origin == Origin.NATURAL
    assert (
        result.prediction.ai_probability is None
    )  # official probability is not our score
    assert result.end_to_end_ms >= result.feature_extraction_ms >= 0
    assert result.readout_ms >= 0
    values[0] = 0.5
    assert detector.predict_file(query).prediction.origin == Origin.AI
    assert "includes a neural network" in result.prediction.metadata["representation"]
    detector.close()
    assert extractor.closed


@pytest.mark.parametrize("values", [np.zeros(3919), np.full(3920, np.nan)])
def test_invalid_extractor_descriptor_refused(values):
    with pytest.raises(ValueError, match="descriptor"):
        CureQuantileDetector(Extractor(values), fixed_rule()).predict_file(
            Path("query.jpg")
        )


def test_foreign_schema_refused_before_encoder_load(tmp_path, monkeypatch):
    from palimpsest.detection.models.frozen_features.cure import detector as module

    names = ("foreign_coordinate",) + FEATURE_NAMES[1:]
    foreign = fixed_rule(names)
    path = tmp_path / "foreign_rule.json"
    foreign.save(path)

    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid schema must be refused before loading the encoder")

    monkeypatch.setattr(module, "FrozenCureQuantiles", unexpected_load)
    with pytest.raises(ValueError, match="3920-feature schema"):
        module.CureQuantileDetector.from_paths(
            path, Path("vendor"), Path("adapter"), Path("base"), source_pins={}
        )
