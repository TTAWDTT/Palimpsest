"""Shared conventional source-risk readouts over fixed feature candidates.

Callbacks keep campaign-specific scoring and threshold policy explicit. This
module neither chooses pixels nor changes the frozen feature representation.
"""

import json

from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.readouts.source_view_risk import fit_source_risk
from .features import feature_views
from .source_training import weighted_source_arrays


def fit_readouts(rows, candidates, *, fit_variants, manifest_sha, config, calibrate, score):
    results, rules, margins = {}, {}, {}
    with threadpool_limits(limits=1):
        for key, names, temperature, shuffled in candidates:
            x, y, weights, sources = weighted_source_arrays(rows, names, fit_variants,
                expected_source_counts={'rr':540, 'chimera':720}, seed=config['seed'], shuffled=shuffled)
            rule, diagnostic = fit_source_risk(x, y, weights, sources, feature_names=names,
                temperature=temperature, ridge=.01, scale_floor=.001,
                maximum_iterations=500, gradient_tolerance=1e-5, manifest_sha=manifest_sha)
            views = {key+'/'+variant:view for variant in fit_variants for processed in (False, True)
                for key,view in feature_views(rows, names, 'threshold', processed=processed, variant=variant).items()}
            if len(views) != 24:
                raise ValueError('Fixed24 calibration views changed')
            rule, diagnostic['calibration'] = calibrate(rule, views)
            scored, result = score(rows, names, rule, config)
            result.update({'fit_diagnostics':diagnostic, 'threshold':rule.threshold})
            results[key], rules[key], margins[key] = result, rule, scored
            print(json.dumps({'candidate':key, 'minimum_ba':result['minimum_domain_ba'],
                'minimum_scene_ba':result['minimum_scene_ba'],
                'maximum_scene_drop':result['maximum_any_scene_drop']}), flush=True)
    return results, rules, margins
