"""Add one fit-local raw semantic direction to the existing score bank.

Conventional supervised score compression and polynomial expansion. The cached
encoder stays frozen; all four directions and moments must be fitted locally.
"""

from dataclasses import asdict, dataclass, replace
import json

import numpy as np

from .source_hinge import fit_source_hinge
from .source_view_risk import fit_source_risk
from .source_score_subspace import BANK_NAMES, DiscriminantBank, ScoreSubspaceFitter, stable_from_payload
from .quadratic_score_margin import _bank_fields
from .paired_stability import StableRule

SCORE_NAMES = BANK_NAMES + ('discriminant/raw_semantic_source',)
TERM_NAMES = SCORE_NAMES + tuple(f'score_quadratic/{i}/{j}' for i in range(4) for j in range(i, 4))


def four_score_terms(z):
    z = np.asarray(z, float)
    if z.ndim != 2 or z.shape[1] != 4 or not np.isfinite(z).all(): raise ValueError('Invalid four-score input')
    with np.errstate(over='ignore', invalid='ignore'):
        terms = np.c_[z, *(z[:, i]*z[:, j]*(1 if i == j else np.sqrt(2))
                           for i in range(4) for j in range(i, 4))]
    if not np.isfinite(terms).all(): raise ValueError('Nonfinite four-score terms')
    return terms


@dataclass(frozen=True)
class SemanticScoreBank:
    feature_names: tuple[str, ...]
    old: DiscriminantBank
    raw: StableRule

    def __post_init__(self):
        n = len(self.raw.feature_names)
        if self.feature_names[:n] != self.raw.feature_names or self.feature_names[n:] != self.old.feature_names:
            raise ValueError('Semantic bank feature order differs')

    def transform(self, values):
        x = np.asarray(values, float); n = len(self.raw.feature_names)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names) or not np.isfinite(x).all():
            raise ValueError('Invalid semantic bank input')
        return np.c_[self.old.transform(x[:, n:]), self.raw.score(x[:, :n])]


@dataclass(frozen=True)
class SemanticScoreRule:
    bank: SemanticScoreBank
    center: tuple[float, ...]
    scale: tuple[float, ...]
    readout: StableRule

    def __post_init__(self):
        if (len(self.center) != 4 or len(self.scale) != 4 or min(self.scale) <= 0
                or not np.isfinite([*self.center, *self.scale]).all() or self.readout.feature_names != TERM_NAMES):
            raise ValueError('Invalid semantic score rule')

    @property
    def threshold(self): return self.readout.threshold

    def transform(self, values): return four_score_terms((self.bank.transform(values)-self.center)/self.scale)

    def score(self, values): return self.readout.score(self.transform(values))

    def save(self, path):
        path.write_text(json.dumps({'schema': 1, 'kind': 'semantic_score_margin', 'rule': asdict(self)}, indent=2)+'\n',
                        encoding='utf-8')

    @classmethod
    def load(cls, path):
        value = json.loads(path.read_text())
        if value['schema'] != 1 or value['kind'] != 'semantic_score_margin': raise ValueError('Invalid semantic payload')
        fields = value['rule']; b = fields['bank']
        wrap = lambda r: stable_from_payload({'schema': 1, 'kind': 'paired_stability', 'rule': r})
        bank = SemanticScoreBank(tuple(b['feature_names']), _bank_fields(b['old']), wrap(b['raw']))
        return cls(bank, tuple(fields['center']), tuple(fields['scale']), wrap(fields['readout']))


class SemanticMarginFitter:
    """One extra raw-mean source-risk fit per exact bank; no new pixel passes."""
    def __init__(self, names, *, raw_dimensions=1024, channels=32, filter_count=16, manifest_sha=''):
        self.names = tuple(names); self.raw_dimensions = raw_dimensions; self.manifest_sha = manifest_sha
        if (not isinstance(raw_dimensions, int) or isinstance(raw_dimensions, bool)
                or not 0 < raw_dimensions < len(self.names)):
            raise ValueError('Invalid raw semantic dimensions')
        self.provider = ScoreSubspaceFitter(self.names[raw_dimensions:], channels=channels,
                                           filter_count=filter_count, manifest_sha=manifest_sha)
        self.banks = {}; self.maps = {}; self.auxiliary_heads = 0

    def fit(self, x, y, weights, sources, penalty, wrong=None):
        x = np.asarray(x, float); key = self.provider.key(x, y, weights, sources); n = self.raw_dimensions
        if key not in self.banks:
            old_rule, old_diagnostic = self.provider.fit(x[:, n:], y, weights, sources, 0)
            self.auxiliary_heads += 1
            raw, raw_diagnostic = fit_source_risk(x[:, :n], y, weights, sources,
                feature_names=self.names[:n], temperature=.1, ridge=.01, scale_floor=.001,
                maximum_iterations=2000, gradient_tolerance=1e-5, manifest_sha=self.manifest_sha)
            bank = SemanticScoreBank(self.names, old_rule.bank, raw)
            self.banks[key] = bank, {'existing': old_diagnostic['basis_diagnostics'], 'raw': raw_diagnostic}
            values = bank.transform(x); w = np.asarray(weights, float); w = w/w.sum()
            center = np.sum(values*w[:, None], axis=0)
            scale = np.maximum(np.sqrt(np.sum((values-center)**2*w[:, None], axis=0)), .001)
            self.maps[key] = tuple(center), tuple(scale)
        bank, diagnostics = self.banks[key]; center, scale = self.maps[key]
        terms = four_score_terms((bank.transform(x)-center)/scale)
        head, diagnostic = fit_source_hinge(terms, y, weights, sources if wrong is None else wrong,
            feature_names=TERM_NAMES, maximum_source=penalty != 0, penalty=penalty if penalty else .001,
            scale_floor=.001, time_limit=120, manifest_sha=self.manifest_sha)
        return SemanticScoreRule(bank, center, scale, head), {**diagnostic, 'training_arrays_sha256': key,
            'basis_diagnostics': diagnostics, 'basis_count': 4, 'mapped_dimensions': 14,
            'fit_scope': 'Supplied fit rows only for existing/raw directions,map and head;not OOF stacking'}

    def audit(self):
        count = len(self.banks)
        old_count = self.provider.builds
        if old_count > count or set(self.banks) != set(self.maps): raise ValueError('Semantic bank cache differs')
        return {'passed': True, 'banks': count, 'old_bank_builds': old_count, 'raw_head_fits': count,
            'base_fits': 3*old_count+count, 'auxiliary_logistic_head_fits': self.auxiliary_heads,
            'maps': len(self.maps), 'training_array_keys': sorted(self.banks), 'scope': 'Execution accounting;not independent evidence'}


def calibrate_semantic_score(rule, views, calibrate):
    mapped = {key: (records, rule.transform(values), labels) for key, (records, values, labels) in views.items()}
    head, diagnostic = calibrate(rule.readout, mapped)
    return replace(rule, readout=head), diagnostic
