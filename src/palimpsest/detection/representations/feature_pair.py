"""Explicit pair of existing frozen feature extractors, with complete live cost."""

from time import perf_counter

import numpy as np

from .frozen_clip import ClipFeatures


class FrozenFeaturePair:
    def __init__(self, first, second, *, first_names, second_names):
        self.first, self.second = first, second
        self.feature_names = tuple(first_names)+tuple(second_names)
        self.dimensions = (len(first_names),len(second_names))
        if min(self.dimensions) <= 0 or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError('Frozen pair requires nonempty distinct feature names')
        self.provenance = {'first':first.provenance, 'second':second.provenance,
            'execution':'Sequential two encoders,not a cache-only deployment',
            'encoder_training_performed_by_composer':False, 'descriptor_dimension':sum(self.dimensions)}

    def extract(self, image):
        start = perf_counter()
        parts = [encoder.extract(image) for encoder in (self.first,self.second)]
        for part,dimension in zip(parts,self.dimensions):
            if (part.values.shape != (dimension,) or not np.isfinite(part.values).all()
                    or not np.isfinite([part.preprocess_ms,part.statistics_ms]).all()
                    or min(part.preprocess_ms,part.statistics_ms) < 0):
                raise ValueError('Frozen pair component feature dimension or values differ')
        values = np.concatenate([p.values for p in parts])
        preprocess_ms = sum(p.preprocess_ms for p in parts)
        elapsed_ms = 1000*(perf_counter()-start)
        return ClipFeatures(values,preprocess_ms,max(0.,elapsed_ms-preprocess_ms))
