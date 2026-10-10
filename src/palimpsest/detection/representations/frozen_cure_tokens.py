"""Read-only PECore ln_post hook; one unchanged frozen CuRe forward."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .frozen_cure import FrozenCure,FULL_NAMES
from .token_statistics import clipped_token_statistics

CLIPPED_NAMES=tuple(f'cure/token_clipped_center/{i}' for i in range(1024))
SPREAD_NAMES=tuple(f'cure/token_clipped_spread_unit/{i}' for i in range(1024))
MIXED_NAMES=CLIPPED_NAMES+SPREAD_NAMES
FEATURE_NAMES=FULL_NAMES+MIXED_NAMES


@dataclass(frozen=True)
class CureTokenFeatures:
    values:np.ndarray
    probability_fake:float
    elapsed_ms:float
    statistics_ms:float
    diagnostics:dict


class FrozenCureTokens:
    def __init__(self,*args,**kwargs):
        self.base=FrozenCure(*args,**kwargs);self._captured=None
        self.handle=self.base.base.model.backbone.ln_post.register_forward_hook(self._capture)
        self.provenance={**self.base.provenance,'token_layer':'PECore ln_post,excludeCLS,576x1024',
            'statistics':'FP64 coordinate median;median radial clipping;center+L2normalized clipped RMS',
            'extra_encoder_forward':False,'descriptor_dimension':3072}

    def _capture(self,module,inputs,output):
        if self._captured is not None or tuple(output.shape)!=(1,577,1024):
            raise ValueError('Unexpected PECore token capture/count')
        patches=output[:,1:].detach()
        # Exactly the author's reduction path, before floating storage cast.
        mean=patches.mean(dim=1).float()[0].cpu().numpy()
        self._captured=mean,patches[0].float().cpu().numpy()

    def extract_file(self,path):
        self._captured=None;start=perf_counter()
        original=self.base.extract_file(path)
        if self._captured is None:raise ValueError('No PECore token capture')
        mean,tokens=self._captured
        if not np.array_equal(mean,original.values[:1024]):raise ValueError('Hook token mean differs from author vector')
        ready=perf_counter();center,spread,diagnostics=clipped_token_statistics(tokens)
        values=np.concatenate((original.values[:1024],center,spread))
        end=perf_counter();self._captured=None
        return CureTokenFeatures(values,original.probability_fake,1000*(end-start),1000*(end-ready),diagnostics)

    def close(self):
        self.handle.remove();self._captured=None

    def direct_features(self,path):
        self.handle.remove();self._captured=None
        try:return self.base.extract_file(path)
        finally:self.handle=self.base.base.model.backbone.ln_post.register_forward_hook(self._capture)
