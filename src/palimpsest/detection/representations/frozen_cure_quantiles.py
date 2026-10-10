"""Read an existing capture without another hook/copy or frozen forward."""

from dataclasses import replace
from time import perf_counter

import numpy as np

from .frozen_cure_covariance import FrozenCureCovariance, FEATURE_NAMES as PARENT_NAMES
from .token_quantiles import projected_quantiles, LEVELS

QUANTILE_NAMES = tuple(f'cure/token_projection_shape/{direction}/{level}'
    for direction in range(32) for level in LEVELS)
FEATURE_NAMES = PARENT_NAMES+QUANTILE_NAMES


class FrozenCureQuantiles(FrozenCureCovariance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shape_tokens = None
        self.provenance.update({'quantile_extension': '32 fixed projection directions;10 linear-interpolated quantiles;median/MAD with homogeneous RMS fallback',
            'quantile_levels': LEVELS, 'descriptor_dimension': len(FEATURE_NAMES),
            'extra_quantile_hook': False, 'extra_quantile_forward': False})

    def _capture(self, module, inputs, output):
        super()._capture(module, inputs, output)
        self._shape_tokens = self._tokens

    def extract_file(self, path):
        self._shape_tokens = None
        start = perf_counter()
        try:
            original = super().extract_file(path)
            if self._shape_tokens is None:
                raise ValueError('Missing quantile token reference')
            ready = perf_counter()
            values, diagnostics = projected_quantiles(self._shape_tokens, self.projection)
            end = perf_counter()
            return replace(original, values=np.r_[original.values, values], elapsed_ms=1000*(end-start),
                statistics_ms=original.statistics_ms+1000*(end-ready),
                diagnostics={**original.diagnostics, **diagnostics})
        finally:
            self._shape_tokens = None

    def close(self):
        try:
            super().close()
        finally:
            self._shape_tokens = None
