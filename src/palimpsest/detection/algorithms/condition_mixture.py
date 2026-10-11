"""Conventional feature-only condition gate and two origin readouts.

Established mixture-of-experts/probability-pooling machinery. No inference
metadata, physical-channel guarantee, or new neural encoder training.
"""

from dataclasses import dataclass
import json

import numpy as np
from scipy.special import expit

from palimpsest.detection.algorithms.readouts.stable_rule import StableRule


def probability_pool_logits(raw, processed, gate):
    """Stable Bernoulli probability pool; supports finite extreme logits."""
    a,b,g = (np.asarray(v,float) for v in (raw,processed,gate))
    if a.shape != b.shape or a.shape != g.shape or not all(np.isfinite(v).all() for v in (a,b,g)):
        raise ValueError('Invalid mixture logits')
    log_raw_weight,log_processed_weight = -np.logaddexp(0,g),-np.logaddexp(0,-g)
    log_ai = np.logaddexp(log_raw_weight-np.logaddexp(0,-a),log_processed_weight-np.logaddexp(0,-b))
    log_real = np.logaddexp(log_raw_weight-np.logaddexp(0,a),log_processed_weight-np.logaddexp(0,b))
    value = log_ai-log_real
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite pooled logits')
    return value


@dataclass(frozen=True)
class ConditionMixtureRule:
    original: StableRule
    processed: StableRule
    gate: StableRule
    uniform: bool = False
    threshold: float = 0.

    def __post_init__(self):
        if (self.original.feature_names != self.processed.feature_names
                or self.original.feature_names != self.gate.feature_names
                or not np.isfinite(self.threshold)):
            raise ValueError('Condition mixture schema or threshold differs')

    @property
    def feature_names(self):
        return self.original.feature_names

    def gate_probability(self,values):
        return np.full(len(values),.5) if self.uniform else expit(self.gate.score(values))

    def score(self,values):
        raw,processed = self.original.score(values),self.processed.score(values)
        gate = np.zeros_like(raw) if self.uniform else self.gate.score(values)
        return probability_pool_logits(raw,processed,gate)

    def save(self,path):
        value = {'schema':1,'kind':'condition_mixture','original':self.original.payload(),
            'processed':self.processed.payload(),'gate':self.gate.payload(),
            'uniform':self.uniform,'threshold':self.threshold,
            'scope':'Feature-only conventional gate;no calibrated wild-channel posterior guarantee'}
        path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

    @classmethod
    def load(cls,path):
        value = json.loads(path.read_text(encoding='utf-8'))
        if value['schema']!=1 or value['kind']!='condition_mixture':
            raise ValueError('Condition mixture artifact differs')
        def read_rule(payload):
            if payload['schema']!=1 or payload['kind']!='paired_stability':
                raise ValueError('Condition component artifact differs')
            params = payload['rule']
            for name in ('feature_names','center','scale','weights'):
                params[name] = tuple(params[name])
            return StableRule(**params)
        return cls(*(read_rule(value[n]) for n in ('original','processed','gate')),
            uniform=value['uniform'],threshold=value['threshold'])
