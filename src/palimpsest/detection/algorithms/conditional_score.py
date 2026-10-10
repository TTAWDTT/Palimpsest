"""A frozen scalar detector score calibrated by a measured radial covariate.

The covariate is encoder log norm, not a certified perceptual quality index.
This conventional heteroskedastic likelihood does not restore lost information.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from palimpsest.detection.algorithms.readouts.stable_rule import StableRule


def radial_coordinate(values):
    x = np.asarray(values, float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Invalid radial feature matrix')
    norms = np.sqrt(np.sum(x*x, axis=1))
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise ValueError('Invalid radial norm')
    return np.log(norms)


def gaussian_objective(theta, design, target, weights, *, ridge):
    """Weighted Gaussian NLL with a derivative consistent with log-var clipping."""
    count = design.shape[1]
    mean = design @ theta[:count]
    raw_log_var = design @ theta[count:]
    log_var = np.clip(raw_log_var, -30., 30.)
    inverse_var = np.exp(-log_var)
    residual = target-mean
    value = .5*np.sum(weights*(log_var+residual*residual*inverse_var)) + ridge*(theta@theta)
    mean_gradient = design.T @ (-weights*residual*inverse_var)
    active = (raw_log_var > -30.) & (raw_log_var < 30.)
    var_gradient = design.T @ (.5*weights*(1-residual*residual*inverse_var)*active)
    return float(value), np.r_[mean_gradient, var_gradient] + 2*ridge*theta


@dataclass(frozen=True)
class ConditionalScoreRule:
    base: StableRule
    conditional: bool
    score_center: float
    score_scale: float
    radius_center: float
    radius_scale: float
    real_parameters: tuple[float, ...]
    fake_parameters: tuple[float, ...]
    threshold: float = 0.

    def __post_init__(self):
        n = 4 if self.conditional else 2
        if (len(self.real_parameters) != n or len(self.fake_parameters) != n
                or min(self.score_scale, self.radius_scale) <= 0
                or not np.isfinite([self.score_center, self.score_scale, self.radius_center,
                    self.radius_scale, *self.real_parameters, *self.fake_parameters, self.threshold]).all()):
            raise ValueError('Invalid conditional score parameters')

    @property
    def feature_names(self):
        return self.base.feature_names

    def score(self, values):
        x = np.asarray(values, float)
        scalar = (self.base.score(x)-self.base.threshold-self.score_center)/self.score_scale
        radius = (radial_coordinate(x)-self.radius_center)/self.radius_scale
        design = np.column_stack((radius, np.ones(len(x)))) if self.conditional else np.ones((len(x), 1))
        outputs = []
        for params in (self.real_parameters, self.fake_parameters):
            p = np.asarray(params)
            n = design.shape[1]
            # Per-row reduction preserves single-image and batch score identity.
            mean = np.sum(design*p[:n], axis=1)
            log_var = np.clip(np.sum(design*p[n:], axis=1), -30., 30.)
            outputs.append(.5*(log_var+(scalar-mean)**2*np.exp(-log_var)))
        result = outputs[0]-outputs[1]
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite calibrated score')
        return result

    def save(self, path: Path):
        path.write_text(json.dumps({'schema': 1, 'kind': 'conditional_score', 'rule': asdict(self)},
                                   indent=2)+'\n', encoding='utf-8')

    @classmethod
    def load(cls, path: Path):
        value = json.loads(path.read_text(encoding='utf-8'))
        if value['schema'] != 1 or value['kind'] != 'conditional_score':
            raise ValueError('Conditional score schema differs')
        params = value['rule']
        base = params['base']
        for k in ('feature_names', 'center', 'scale', 'weights'):
            base[k] = tuple(base[k])
        params['base'] = StableRule(**base)
        for k in ('real_parameters', 'fake_parameters'):
            params[k] = tuple(params[k])
        return cls(**params)


def fit_conditional_score(x, labels, weights, base, *, conditional=True, radius_override=None,
                          ridge=1e-4, gradient_tolerance=1e-5, maximum_iterations=300):
    x, labels, weights = np.asarray(x, float), np.asarray(labels), np.asarray(weights, float)
    if (x.ndim != 2 or labels.shape != (len(x),) or weights.shape != labels.shape
            or set(labels.tolist()) != {0, 1} or not np.isfinite(weights).all() or np.any(weights <= 0)
            or not np.isfinite([ridge, gradient_tolerance]).all() or min(ridge, gradient_tolerance) <= 0
            or maximum_iterations < 1):
        raise ValueError('Invalid conditional score fit arrays')
    weights = weights/weights.sum()
    scalar = base.score(x)-base.threshold
    radius = radial_coordinate(x) if radius_override is None else np.asarray(radius_override, float)
    if radius.shape != labels.shape or not np.isfinite(radius).all():
        raise ValueError('Invalid fit radial override')
    score_center = float(weights@scalar)
    score_scale = max(float(np.sqrt(weights@((scalar-score_center)**2))), .001)
    radius_center = float(weights@radius)
    radius_scale = max(float(np.sqrt(weights@((radius-radius_center)**2))), .001)
    target = (scalar-score_center)/score_scale
    normalized = (radius-radius_center)/radius_scale
    design = np.column_stack((normalized, np.ones(len(x)))) if conditional else np.ones((len(x), 1))
    parameters, diagnostics = [], []
    for label in (0, 1):
        mask = labels == label
        a, b, w = design[mask], target[mask], weights[mask]
        w = w/w.sum()
        mean = np.linalg.solve(a.T@(w[:, None]*a)+ridge*np.eye(a.shape[1]), a.T@(w*b))
        log_var = np.zeros(a.shape[1])
        log_var[-1] = np.log(max(float(w@((b-a@mean)**2)), .001))
        initial = np.r_[mean, log_var]
        def objective(theta):
            return gaussian_objective(theta, a, b, w, ridge=ridge)
        solution = minimize(objective, initial, jac=True, method='L-BFGS-B',
            options={'maxiter': maximum_iterations, 'gtol': gradient_tolerance/10, 'ftol': 1e-14, 'maxls': 40})
        value, gradient = objective(solution.x)
        residual = float(np.max(np.abs(gradient)))
        if (not solution.success or not np.isfinite(solution.x).all()
                or residual > gradient_tolerance or value > objective(initial)[0]+1e-10):
            raise ValueError(f'Conditional score optimizer failed: {solution.message};gradient={residual}')
        parameters.append(tuple(solution.x))
        diagnostics.append({'class': label, 'objective': value, 'initial_objective': objective(initial)[0],
                            'maximum_absolute_gradient': residual, 'iterations': int(solution.nit)})
    return ConditionalScoreRule(base, conditional, score_center, score_scale, radius_center, radius_scale,
                                *parameters), diagnostics
