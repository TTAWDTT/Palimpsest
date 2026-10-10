"""Classical covariance readouts of frozen pixel statistics.

The fitted Gaussian model is an approximation to pooled development features;
it supplies no invariance theorem and consumes no target batch statistics.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from palimpsest.contracts import Prediction
from .residual_statistics.features import FEATURE_NAMES as RESIDUAL_NAMES, extract_features
from .wavelet_envelopes import FEATURE_NAMES as ENVELOPE_NAMES, extract_envelopes


@dataclass(frozen=True)
class GaussianRule:
    feature_names: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    linear: tuple[float, ...]
    diagonal: tuple[float, ...]
    quadratic: tuple[tuple[float, ...], ...]
    bias: float
    mode: str
    ridge: float
    shrinkage: float
    threshold: float = 0.0
    fit_manifest_sha256: str = ''

    def __post_init__(self):
        n = len(self.feature_names)
        if (not n or len(set(self.feature_names)) != n
                or any(len(v) != n for v in (self.center, self.scale, self.linear, self.diagonal))
                or self.mode not in ('pooled', 'diagonal', 'class_full')):
            raise ValueError('Invalid Gaussian rule dimensions/mode')
        scalars = [*self.center, *self.scale, *self.linear, *self.diagonal,
                   self.bias, self.ridge, self.shrinkage, self.threshold]
        if not np.isfinite(scalars).all() or min(self.scale) <= 0 or self.ridge <= 0 or not 0 <= self.shrinkage <= 1:
            raise ValueError('Invalid Gaussian rule values')
        if self.quadratic:
            matrix = np.asarray(self.quadratic)
            if (matrix.shape != (n, n) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0)):
                raise ValueError('Invalid Gaussian quadratic matrix')

    def score(self, values):
        x = np.asarray(values, float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names) or not np.isfinite(x).all():
            raise ValueError('Invalid Gaussian feature matrix')
        z = (x-self.center)/self.scale
        scores = np.sum(z*np.asarray(self.linear) + z*z*np.asarray(self.diagonal), axis=1) + self.bias
        if self.quadratic:
            matrix = np.asarray(self.quadratic)
            # Same matrix-vector reduction for single and batch calls.
            scores += np.array([np.sum(row*(matrix@row)) for row in z])
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite Gaussian scores')
        return scores

    def payload(self):
        return {'schema': 1, 'kind': 'gaussian_readout', 'rule': asdict(self)}

    @property
    def fingerprint(self):
        return sha256(json.dumps(self.payload(), sort_keys=True).encode()).hexdigest()

    def save(self, path: Path):
        path.write_text(json.dumps(self.payload(), indent=2)+'\n', encoding='utf-8')

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['schema'] != 1 or data['kind'] != 'gaussian_readout':
            raise ValueError('Gaussian schema differs')
        params = data['rule']
        for name in ('feature_names', 'center', 'scale', 'linear', 'diagonal'):
            params[name] = tuple(params[name])
        params['quadratic'] = tuple(tuple(row) for row in params['quadratic'])
        return cls(**params)


def fit_gaussian_rule(values, labels, sample_weights, *, feature_names, mode,
                      ridge=.1, shrinkage=.5, scale_floor=.001, manifest_sha=''):
    """Weighted class moments; shared covariance, diagonal NB, or shrunken QDA.

    Features are standardized using fit-only moments. Class priors are fixed
    at 1/2; covariance ridge and shrinkage are fixed before the experiment.
    Output is log p(x|AI)/p(x|natural), with a separately calibrated threshold.
    """
    x, y, sw = np.asarray(values, float), np.asarray(labels), np.asarray(sample_weights, float)
    if (x.ndim != 2 or not len(x) or x.shape[1] != len(feature_names) or y.shape != (len(x),)
            or set(y.tolist()) != {0, 1} or sw.shape != y.shape or mode not in ('pooled', 'diagonal', 'class_full')):
        raise ValueError('Invalid Gaussian fitting dimensions/labels/mode')
    if (not np.isfinite(x).all() or not np.isfinite(sw).all() or min(sw) <= 0
            or not np.isfinite([ridge, shrinkage, scale_floor]).all()
            or ridge <= 0 or scale_floor <= 0 or not 0 <= shrinkage <= 1):
        raise ValueError('Invalid Gaussian fitting values')
    sw = sw/sw.sum()
    center = np.sum(sw[:, None]*x, axis=0)
    scale = np.maximum(np.sqrt(np.sum(sw[:, None]*(x-center)**2, axis=0)), scale_floor)
    z = (x-center)/scale
    means, covariances = [], []
    for label in (0, 1):
        weights = sw[y == label]; weights = weights/weights.sum()
        selected = z[y == label]
        mean = np.sum(weights[:, None]*selected, axis=0)
        residual = selected-mean
        covariance = residual.T@(weights[:, None]*residual)
        means.append(mean); covariances.append((covariance+covariance.T)/2)
    if mode == 'pooled':
        covariances = [sum(covariances)/2]*2
    precisions, logdets, minimum_eigenvalues = [], [], []
    for covariance in covariances:
        diagonal = np.diag(np.diag(covariance))
        adjusted = (diagonal if mode == 'diagonal' else (1-shrinkage)*covariance+shrinkage*diagonal)
        adjusted = adjusted+ridge*np.eye(x.shape[1])
        factor = np.linalg.cholesky(adjusted)
        precision = np.linalg.solve(adjusted, np.eye(x.shape[1]))
        precision = (precision+precision.T)/2
        if np.linalg.norm(adjusted@precision-np.eye(x.shape[1]), ord=np.inf) > 1e-7:
            raise ValueError('Gaussian inverse residual exceeds tolerance')
        precisions.append(precision)
        logdets.append(float(2*np.log(np.diag(factor)).sum()))
        minimum_eigenvalues.append(float(np.linalg.eigvalsh(adjusted)[0]))
    q = .5*(precisions[0]-precisions[1])
    linear = precisions[1]@means[1]-precisions[0]@means[0]
    bias = float(.5*(means[0]@precisions[0]@means[0]-means[1]@precisions[1]@means[1]+logdets[0]-logdets[1]))
    diagonal = np.diag(q) if mode == 'diagonal' else np.zeros(x.shape[1])
    quadratic = tuple(tuple(v) for v in q) if mode == 'class_full' else ()
    rule = GaussianRule(tuple(feature_names), tuple(center), tuple(scale), tuple(linear), tuple(diagonal),
                        quadratic, bias, mode, ridge, shrinkage, fit_manifest_sha256=manifest_sha)
    return rule, {'records': len(x), 'dimensions': x.shape[1], 'class_weight_sums': [float(sw[y == k].sum()) for k in (0, 1)],
                  'minimum_covariance_eigenvalues': minimum_eigenvalues, 'logdet_covariances': logdets,
                  'prior_ai': .5, 'fit_score_finite': bool(np.isfinite(rule.score(x)).all())}


class GaussianDetector:
    name = 'gaussian-statistics-readout'

    def __init__(self, rule: GaussianRule):
        if rule.feature_names not in (RESIDUAL_NAMES, RESIDUAL_NAMES+ENVELOPE_NAMES):
            raise ValueError('Gaussian pixel feature schema differs')
        self.rule = rule
        self.rule_sha256 = rule.fingerprint

    def predict(self, image):
        start = perf_counter(); old = extract_features(image)
        values = old.values
        if len(self.rule.feature_names) != len(RESIDUAL_NAMES):
            values = np.r_[values, extract_envelopes(image).values]
        score = float(self.rule.score(values[None])[0])
        return Prediction(self.name, score, self.rule.threshold, 'statistical_score',
                          timing_ms={'predict_ms': 1000*(perf_counter()-start)},
                          metadata={'rule_sha256': self.rule_sha256, 'fit_manifest_sha256': self.rule.fit_manifest_sha256})
