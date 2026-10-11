"""Pixel-only residual/ordinal detector, requiring a portable calibrated forest."""

from time import perf_counter

from palimpsest.contracts import Prediction
from palimpsest.detection.algorithms.forest import ForestRule
from .features import FEATURE_NAMES, extract_features


class ResidualDetector:
    def __init__(self, rule: ForestRule):
        if rule.feature_names not in (FEATURE_NAMES, FEATURE_NAMES[:28]):
            raise ValueError("Residual feature identity differs")
        self.rule = rule
        self.name = 'residual-ordinal-forest' if len(rule.feature_names) > 28 else 'ordinal256-forest'
        self.rule_sha256 = rule.fingerprint

    def predict(self, image):
        start = perf_counter()
        result = extract_features(image, residual=len(self.rule.feature_names) > 28)
        score = float(self.rule.score(result.values[None, :len(self.rule.feature_names)])[0])
        return Prediction(self.name, score, self.rule.threshold, 'statistical_score',
                          timing_ms={'preprocess_ms': result.preprocess_ms, 'statistics_ms': result.statistics_ms,
                                     'predict_ms': (perf_counter()-start)*1000},
                          metadata={'rule_sha256': self.rule_sha256, 'fit_manifest_sha256': self.rule.fit_manifest_sha256})
