"""Existing five-direction bank with a conventional rate-aware penalty head."""

import numpy as np

from .quantile_score_consistency import QuantileScoreFitter, QuantileScoreRule, TERM_NAMES
from .rate_penalty import training_rate_panel, fit_rate_source_risk


class RateScoreFitter:
    def __init__(self, names, *, variants, manifest_sha='', **kwargs):
        self.provider = QuantileScoreFitter(names, manifest_sha=manifest_sha, **kwargs)
        self.variants = tuple(variants); self.manifest_sha = manifest_sha
        self.templates = {}; self.template_diagnostics = {}; self.readout_calls = 0

    def fit(self, x, labels, weights, sources, strength, wrong=None, *, records=None):
        if records is None:
            raise ValueError('Rate fitting requires explicit training records')
        labels = np.asarray(labels)
        if labels.ndim != 1 or not set(labels.tolist()) <= {0, 1}:
            raise ValueError('Invalid rate wrapper labels')
        labels = labels.astype(np.int64)
        panel = training_rate_panel(records, labels, weights, sources, variants=self.variants)
        if wrong is not None and (np.asarray(wrong).shape != labels.shape or np.asarray(wrong).dtype.kind not in 'iu'):
            raise ValueError('Invalid wrong source control')
        key = self.provider.provider.provider.key(x, labels, weights, sources)
        if key not in self.templates:
            template, diagnostic = self.provider.fit(x, labels, weights, sources, 0)
            self.templates[key] = template; self.template_diagnostics[key] = diagnostic
        template = self.templates[key]
        head, diagnostic = fit_rate_source_risk(template.transform(x), labels, weights, sources, panel,
            strength=strength, feature_names=TERM_NAMES, temperature=.1, ridge=.01,
            scale_floor=.001, maximum_iterations=2000, gradient_tolerance=1e-5, manifest_sha=self.manifest_sha)
        self.readout_calls += 1
        return QuantileScoreRule(template.bank, template.center, template.scale, head), {**diagnostic,
            'training_arrays_sha256': key, 'basis_diagnostics': self.template_diagnostics[key],
            'basis_count': 5, 'mapped_dimensions': 20, 'wrong_source_affects_objective': False,
            'control_scope': 'Aggregate rate difference is pairing-invariant;wrong source is a structural identity control'}

    def audit(self):
        provider = self.provider.audit()
        if len(self.templates) != provider['banks']:
            raise ValueError('Rate bank ledger differs')
        return {'passed': True, 'banks': len(self.templates), 'maps': len(self.templates),
            'provider': provider, 'initialization_source_readout_calls': provider['source_consistency_fit_calls'],
            'rate_readout_fit_calls': self.readout_calls,
            'scope': 'Initialization and readout entries;nonzero readouts also solve baseline then penalized objective'}
