"""Pixel-residual adapter for the shared stable score rule."""

from time import perf_counter

from palimpsest.contracts import Prediction
from .readouts.stable_rule import StableRule
from .residual_statistics.features import FEATURE_NAMES, extract_features


class StableDetector:
    name = 'paired-stable-residual-score'

    def __init__(self, rule: StableRule):
        if rule.feature_names != FEATURE_NAMES:
            raise ValueError('Residual stable feature schema differs')
        self.rule = rule
        self.rule_sha256 = rule.fingerprint

    def predict(self, image):
        start = perf_counter(); f = extract_features(image)
        score = float(self.rule.score(f.values[None])[0])
        return Prediction(self.name, score, self.rule.threshold, 'statistical_score',
                          timing_ms={'preprocess_ms': f.preprocess_ms, 'statistics_ms': f.statistics_ms,
                                     'predict_ms': 1000*(perf_counter()-start)},
                          metadata={'rule_sha256': self.rule_sha256, 'fit_manifest_sha256': self.rule.fit_manifest_sha256})
