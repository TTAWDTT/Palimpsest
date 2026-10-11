"""Prepare immutable Fourier arrays once for portable kernel-rule inference.

Numerical operations match the portable rule; this changes allocation cost,
not the learned mapping, classifier or decision threshold.
"""

import numpy as np
from .kernel_readout import KernelRule


class CompiledKernelRule:
    def __init__(self,rule:KernelRule):
        self.rule=rule
        self.feature_names=rule.mapper.feature_names
        self.threshold=rule.threshold
        self._center=np.asarray(rule.mapper.center)
        self._scale=np.asarray(rule.mapper.scale)
        self._frequencies=np.asarray(rule.mapper.frequencies)
        for values in (self._center,self._scale,self._frequencies):values.flags.writeable=False

    def score(self,values):
        x=np.asarray(values,float)
        if x.ndim!=2 or x.shape[1]!=len(self.feature_names) or not np.isfinite(x).all():
            raise ValueError('Invalid compiled-kernel input')
        z=(x-self._center)/self._scale
        if len(z):
            angles=np.array([self._frequencies@row for row in z])
            mapped=np.c_[np.cos(angles),np.sin(angles)]/np.sqrt(len(self._frequencies))
        else:mapped=np.empty((0,2*len(self._frequencies)))
        if self.rule.concatenate_input:mapped=np.c_[x,mapped]
        return self.rule.readout.score(mapped)
