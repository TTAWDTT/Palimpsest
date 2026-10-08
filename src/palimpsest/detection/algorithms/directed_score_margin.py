"""Same five-direction bank, explicitly contextual processing-order readout.

Prediction uses only the existing numeric rule layout. Training records provide
pair edges; they are never guessed from embeddings or retained in the detector.
"""

from .quantile_score_consistency import QuantileScoreFitter, QuantileScoreRule, TERM_NAMES
from .directed_margin import processing_pairs, fit_directed_source_risk


class DirectedScoreFitter:
    def __init__(self, names, *, variants, manifest_sha='', **kwargs):
        self.provider = QuantileScoreFitter(names, manifest_sha=manifest_sha, **kwargs)
        self.variants = tuple(variants); self.manifest_sha = manifest_sha
        self.templates = {}; self.template_diagnostics = {}; self.readout_calls = 0

    def fit(self, x, labels, weights, sources, strength, wrong=None, *, records=None):
        if records is None:
            raise ValueError('Directed fitting requires explicit training records')
        pairs = processing_pairs(records, labels, weights, sources if wrong is None else wrong,
            variants=self.variants, allow_virtual_sources=wrong is not None)
        key = self.provider.provider.provider.key(x, labels, weights, sources)
        if key not in self.templates:
            # Public construction path also fits an unused zero-strength source
            # head. Ledger records it; no private reconstruction of old banks.
            template, diagnostic = self.provider.fit(x, labels, weights, sources, 0)
            self.templates[key] = template; self.template_diagnostics[key] = diagnostic
        template = self.templates[key]
        terms = template.transform(x)
        head, diagnostic = fit_directed_source_risk(terms, labels, weights, sources, pairs,
            strength=strength, feature_names=TERM_NAMES, temperature=.1, ridge=.01,
            scale_floor=.001, maximum_iterations=2000, gradient_tolerance=1e-5, manifest_sha=self.manifest_sha)
        self.readout_calls += 1
        rule = QuantileScoreRule(template.bank, template.center, template.scale, head)
        return rule, {**diagnostic, 'training_arrays_sha256': key,
            'basis_diagnostics': self.template_diagnostics[key], 'basis_count': 5,
            'mapped_dimensions': 20, 'cross_source_edges': pairs.cross_source_edges,
            'pair_scope': 'Given training metadata only;wrong changes pair groups,not risk/banks/maps',
            'serialized_rule_kind_scope': 'Existing five-score layout,not a certificate oftraining loss'}

    def audit(self):
        provider = self.provider.audit()
        if len(self.templates) != provider['banks']:
            raise ValueError('Directed bank/initialization ledger differs')
        return {'passed': True, 'banks': len(self.templates), 'maps': len(self.templates),
            'provider': provider, 'initialization_source_readout_calls': provider['source_consistency_fit_calls'],
            'directed_readout_fit_calls': self.readout_calls,
            'scope': 'Wrapper/initialization counts;baseline+positive-penalty optimizations are separate'}
