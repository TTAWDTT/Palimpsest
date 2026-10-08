"""L2 smooth source-risk/variance head over the same four semantic scores.

Reuses the existing supervised bank/map and convex consistency solver. Changes
the surrogate and regularizer,not just sparsity; neither assures image stability.
"""

from dataclasses import replace

from .semantic_score_margin import SemanticMarginFitter, TERM_NAMES
from .consistent_source_risk import fit_consistent_source_risk


class SemanticConsistencyFitter:
    def __init__(self, names, *, raw_dimensions=1024, channels=32, filter_count=16, manifest_sha=''):
        self.provider = SemanticMarginFitter(names, raw_dimensions=raw_dimensions, channels=channels,
            filter_count=filter_count, manifest_sha=manifest_sha)
        self.templates = {}; self.template_diagnostics = {}; self.readout_calls = 0; self.manifest_sha = manifest_sha

    def fit(self, x, y, weights, sources, strength, wrong=None):
        key = self.provider.provider.key(x, y, weights, sources)
        if key not in self.templates:
            # Reuse the provider's public construction path; count its unused
            # average-hinge head, rather than pretending only the bank was fit.
            template, diagnostic = self.provider.fit(x, y, weights, sources, 0)
            self.templates[key] = template; self.template_diagnostics[key] = diagnostic
        template = self.templates[key]
        head, d = fit_consistent_source_risk(template.transform(x), y, weights, sources,
            strength=strength, consistency_groups=wrong, feature_names=TERM_NAMES,
            temperature=.1, ridge=.01, scale_floor=.001, maximum_iterations=2000,
            gradient_tolerance=1e-5, manifest_sha=self.manifest_sha)
        self.readout_calls += 1
        return replace(template, readout=head), {**d, 'training_arrays_sha256': key,
            'basis_diagnostics': self.template_diagnostics[key]['basis_diagnostics'],
            'mapped_dimensions': 14, 'fit_scope': 'Supplied fit rows only;whole-pipeline held-source CV;not OOF stacking',
            'readout_method': 'L2 entropy-smoothed source logistic plus score variance',
            'rule_layout': 'semantic_score_margin layout encodes bank/map/head;does not identify fitting loss'}

    def audit(self):
        audit = self.provider.audit()
        if set(self.templates) != set(self.provider.banks): raise ValueError('Consistency template cache differs')
        return {**audit, 'initialization_lp_fits': len(self.templates), 'source_consistency_fit_calls': self.readout_calls,
            'readout_count_scope': 'Model fitting entry calls;nonzero strength additionally solves baseline source loss'}
