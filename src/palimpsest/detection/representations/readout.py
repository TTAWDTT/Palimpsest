"""Explicit feature-to-prediction adapter, with no default trained method."""

from time import perf_counter
from palimpsest.contracts import Prediction, validate_rgb


class FeatureReadoutDetector:
    def __init__(self, extractor, rule, *, feature_names, name):
        if tuple(feature_names) != rule.feature_names or not name:
            raise ValueError('Readout feature schema/name differs')
        self.extractor = extractor; self.rule = rule; self.name = name

    def predict(self, image):
        validate_rgb(image); start = perf_counter()
        features = self.extractor(image); ready = perf_counter()
        score = float(self.rule.score([features.values])[0]); end = perf_counter()
        return Prediction(self.name, score, self.rule.threshold, 'statistical_score', timing_ms={
            'preprocess_ms': features.preprocess_ms, 'feature_ms': features.statistics_ms,
            'readout_ms': 1000*(end-ready), 'predict_ms': 1000*(end-start)},
            metadata={'status': 'explicit development candidate, not a validated default'})
