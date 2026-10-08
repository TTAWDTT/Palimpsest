"""Fit-local shape direction beside an unchanged four-direction legacy bank.

Only the legacy prefix enters covariance-aware code. A fixed polynomial readout
is conventional supervised compression, not physical channel invariance.
"""

from dataclasses import asdict, dataclass, replace
import json

import numpy as np

from .semantic_score_margin import SemanticMarginFitter, SemanticScoreBank, SCORE_NAMES
from .quadratic_score_margin import _bank_fields
from .source_score_subspace import stable_from_payload
from .source_view_risk import fit_source_risk
from .consistent_source_risk import fit_consistent_source_risk
from .paired_stability import StableRule

FIVE_NAMES = SCORE_NAMES+('discriminant/token_projection_shape',)
TERM_NAMES = FIVE_NAMES+tuple(f'shape_score_quadratic/{i}/{j}' for i in range(5) for j in range(i, 5))


def five_score_terms(z):
    z = np.asarray(z, float)
    if z.ndim != 2 or z.shape[1] != 5 or not np.isfinite(z).all():
        raise ValueError('Invalid five-score matrix')
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.c_[z, *(z[:, i]*z[:, j]*(1 if i == j else np.sqrt(2))
                            for i in range(5) for j in range(i, 5))]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite five-score expansion')
    return result


@dataclass(frozen=True)
class QuantileScoreBank:
    feature_names: tuple[str, ...]
    legacy: SemanticScoreBank
    shape: StableRule

    def __post_init__(self):
        n = len(self.legacy.feature_names)
        if self.feature_names[:n] != self.legacy.feature_names or self.feature_names[n:] != self.shape.feature_names:
            raise ValueError('Shape/legacy prefix layout differs')

    def transform(self, values):
        x = np.asarray(values, float)
        n = len(self.legacy.feature_names)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names) or not np.isfinite(x).all():
            raise ValueError('Invalid shape score input')
        return np.c_[self.legacy.transform(x[:, :n]), self.shape.score(x[:, n:])]


@dataclass(frozen=True)
class QuantileScoreRule:
    bank: QuantileScoreBank
    center: tuple[float, ...]
    scale: tuple[float, ...]
    readout: StableRule

    def __post_init__(self):
        if (len(self.center) != 5 or len(self.scale) != 5 or min(self.scale) <= 0
                or not np.isfinite([*self.center, *self.scale]).all() or self.readout.feature_names != TERM_NAMES):
            raise ValueError('Invalid shape-score rule')

    @property
    def threshold(self):
        return self.readout.threshold

    def transform(self, values):
        return five_score_terms((self.bank.transform(values)-self.center)/self.scale)

    def score(self, values):
        return self.readout.score(self.transform(values))

    def save(self, path):
        path.write_text(json.dumps({'schema': 1, 'kind': 'quantile_score_consistency', 'rule': asdict(self)},
            indent=2)+'\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        value = json.loads(path.read_text())
        if value['schema'] != 1 or value['kind'] != 'quantile_score_consistency':
            raise ValueError('Unknown quantile-score schema')
        fields = value['rule']; b = fields['bank']; old = b['legacy']
        stable = lambda row: stable_from_payload({'schema': 1, 'kind': 'paired_stability', 'rule': row})
        legacy = SemanticScoreBank(tuple(old['feature_names']), _bank_fields(old['old']), stable(old['raw']))
        bank = QuantileScoreBank(tuple(b['feature_names']), legacy, stable(b['shape']))
        return cls(bank, tuple(fields['center']), tuple(fields['scale']), stable(fields['readout']))


class QuantileScoreFitter:
    def __init__(self, names, *, legacy_dimensions=3600, raw_dimensions=1024, channels=32,
                 filter_count=16, manifest_sha=''):
        self.names = tuple(names); self.n = legacy_dimensions; self.manifest_sha = manifest_sha
        if not isinstance(legacy_dimensions, int) or isinstance(legacy_dimensions, bool) or not 0 < self.n < len(names):
            raise ValueError('Invalid legacy feature boundary')
        self.provider = SemanticMarginFitter(self.names[:self.n], raw_dimensions=raw_dimensions,
            channels=channels, filter_count=filter_count, manifest_sha=manifest_sha)
        self.old_banks = {}; self.banks = {}; self.maps = {}; self.details = {}
        self.initialization_lp_fits = 0; self.shape_fits = 0; self.readout_calls = 0

    def fit(self, x, labels, weights, sources, strength, wrong=None):
        x = np.asarray(x, float)
        if x.ndim != 2 or x.shape[1] != len(self.names) or not np.isfinite(x).all():
            raise ValueError('Invalid shape fitter input')
        key = self.provider.provider.key(x, labels, weights, sources)
        if key not in self.banks:
            old_key = self.provider.provider.key(x[:, :self.n], labels, weights, sources)
            if old_key not in self.old_banks:
                template, diagnostic = self.provider.fit(x[:, :self.n], labels, weights, sources, 0)
                self.initialization_lp_fits += 1
                self.old_banks[old_key] = template.bank, diagnostic
            old, old_diagnostic = self.old_banks[old_key]
            shape, shape_diagnostic = fit_source_risk(x[:, self.n:], labels, weights, sources,
                feature_names=self.names[self.n:], temperature=.1, ridge=.01, scale_floor=.001,
                maximum_iterations=2000, gradient_tolerance=1e-5, manifest_sha=self.manifest_sha)
            self.shape_fits += 1
            bank = QuantileScoreBank(self.names, old, shape)
            scores = bank.transform(x); w = np.asarray(weights, float); w = w/w.sum()
            center = np.sum(scores*w[:, None], axis=0)
            scale = np.maximum(np.sqrt(np.sum((scores-center)**2*w[:, None], axis=0)), .001)
            self.banks[key] = bank; self.maps[key] = tuple(center), tuple(scale)
            self.details[key] = {'legacy': old_diagnostic, 'shape': shape_diagnostic, 'legacy_array_key': old_key}
        bank = self.banks[key]; center, scale = self.maps[key]
        terms = five_score_terms((bank.transform(x)-center)/scale)
        head, diagnostic = fit_consistent_source_risk(terms, labels, weights, sources, strength=strength,
            consistency_groups=wrong, feature_names=TERM_NAMES, temperature=.1, ridge=.01,
            scale_floor=.001, maximum_iterations=2000, gradient_tolerance=1e-5, manifest_sha=self.manifest_sha)
        self.readout_calls += 1
        return QuantileScoreRule(bank, center, scale, head), {**diagnostic, 'training_arrays_sha256': key,
            'basis_diagnostics': self.details[key], 'basis_count': 5, 'mapped_dimensions': 20,
            'fit_scope': 'All bases/map/readout on supplied training sources;legacy layout unchanged;not OOF stacking'}

    def audit(self):
        if len(self.banks) != len(self.maps) or len(self.banks) != self.shape_fits:
            raise ValueError('Shape bank ledger differs')
        legacy = self.provider.audit()
        return {'passed': True, 'banks': len(self.banks), 'maps': len(self.maps),
            'legacy_full_array_banks': len(self.old_banks), 'legacy': legacy,
            'shape_head_fits': self.shape_fits, 'initialization_lp_fits': self.initialization_lp_fits,
            'source_consistency_fit_calls': self.readout_calls,
            'scope': 'Execution counts;not optimizer objective count or independent evidence'}


def calibrate_quantile_score(rule, views, calibrate):
    mapped = {key: (records, rule.transform(values), labels) for key, (records, values, labels) in views.items()}
    head, diagnostic = calibrate(rule.readout, mapped)
    return replace(rule, readout=head), diagnostic
