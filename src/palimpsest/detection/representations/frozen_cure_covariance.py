"""Extend the signed CuRe token descriptor with one fixed covariance sketch."""

from dataclasses import replace
from hashlib import sha256
from time import perf_counter

import numpy as np

from .frozen_cure_tokens import FrozenCureTokens, FEATURE_NAMES as PARENT_NAMES
from .token_covariance import fixed_projection, projected_covariance

COVARIANCE_NAMES = tuple(f'cure/token_covariance_sqrt/{i}' for i in range(528))
FEATURE_NAMES = PARENT_NAMES+COVARIANCE_NAMES


class FrozenCureCovariance:
    """Second read-only hook, same frozen forward; parent statistics unchanged."""
    def __init__(self,*args,**kwargs):
        self.base = FrozenCureTokens(*args,**kwargs)
        self.projection = fixed_projection(1024,32,20261008)
        self._tokens = None
        self.handle = self.base.base.base.model.backbone.ln_post.register_forward_hook(self._capture)
        self.provenance = {**self.base.provenance,
            'extension':'Raw token covariance in fixed32-dimensional orthonormal projection;trace-normalized PSD square root;svec528',
            'projection_seed':20261008,'projection_sha256':sha256(self.projection.astype('<f8').tobytes()).hexdigest(),
            'descriptor_dimension':3600,'additional_encoder_forward':False}

    def _capture(self,module,inputs,output):
        if self._tokens is not None or tuple(output.shape)!=(1,577,1024):
            raise ValueError('Unexpected covariance token capture')
        self._tokens = output[0,1:].detach().float().cpu().numpy()

    def extract_file(self,path):
        self._tokens = None;start = perf_counter()
        original = self.base.extract_file(path)
        if self._tokens is None:raise ValueError('Missing covariance tokens')
        ready = perf_counter();values,diagnostics = projected_covariance(self._tokens,self.projection)
        end = perf_counter();self._tokens = None
        return replace(original,values=np.r_[original.values,values],elapsed_ms=1000*(end-start),
            statistics_ms=original.statistics_ms+1000*(end-ready),diagnostics={**original.diagnostics,**diagnostics})

    def direct_features(self,path):
        self.handle.remove();self._tokens = None
        try:return self.base.direct_features(path)
        finally:self.handle=self.base.base.base.model.backbone.ln_post.register_forward_hook(self._capture)

    def close(self):
        self.handle.remove();self._tokens = None;self.base.close()
