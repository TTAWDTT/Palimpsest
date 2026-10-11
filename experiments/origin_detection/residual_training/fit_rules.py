"""A bounded 2x2 intervention, keeping sixth raw threshold calibration fixed."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from palimpsest.detection.algorithms.forest import ForestRule
from palimpsest.detection.algorithms.residual_statistics.features import FEATURE_NAMES
from palimpsest.evaluation.features import feature_views
from experiments.origin_detection.ordinal_statistics.fit_rules import choose_threshold
from experiments.origin_detection.residual_statistics.fit_rules import all_views

MODES = ('equal_view_raw', 'equal_domain_raw', 'equal_view_augmented', 'equal_domain_augmented')
TRAINING_VARIANT = 'jpeg90_444_after_resize256'


def training_arrays(rows, mode, seed, *, shuffled=False):
    if mode not in MODES:
        raise ValueError('Unregistered training mode')
    variants = ('raw', TRAINING_VARIANT) if mode.endswith('_augmented') else ('raw',)
    sources = {r['src']: r['label'] for r in rows if r['role'] == 'fit' and r['variant'] == 'raw'}
    if shuffled:
        rng = np.random.default_rng(seed)
        for domain, scene in sorted({(r['domain'], r['scene']) for r in rows if r['role'] == 'fit'}):
            ids = sorted({r['src'] for r in rows if r['role'] == 'fit' and r['domain'] == domain and r['scene'] == scene})
            sources.update(zip(ids, rng.permutation([sources[s] for s in ids]).tolist()))
    values, labels, weights, identities = [], [], [], []
    for variant in variants:
        views = {**feature_views(rows, FEATURE_NAMES, 'fit', processed=False, variant=variant),
                 **feature_views(rows, FEATURE_NAMES, 'fit', processed=True, variant=variant)}
        if len(views) != 12:
            raise ValueError('Expected twelve fit views per encoding')
        for records, x, truth in views.values():
            if sum(truth == 0) != sum(truth == 1) or not np.isfinite(x).all():
                raise ValueError('Expected finite balanced fit views')
            domain = records[0]['domain']
            if domain not in ('rr', 'chimera'):
                raise ValueError('Unknown fit domain')
            factor = 3 if mode.startswith('equal_domain') and domain == 'rr' else 1
            values.append(x)
            labels.append(np.array([sources[r['src']] == 'FAKE' for r in records], dtype=int))
            weights.append(np.full(len(x), factor / (len(x) * len(variants))))
            identities.extend((r['src'], domain, variant) for r in records)
    return np.concatenate(values), np.concatenate(labels), np.concatenate(weights), identities


def fit_rule(rows, mode, *, seed, manifest_sha, shuffled=False):
    x, y, weights, identities = training_arrays(rows, mode, seed, shuffled=shuffled)
    estimator = RandomForestClassifier(n_estimators=128, max_depth=6,
                                       min_samples_leaf=16 if mode.endswith('_augmented') else 8,
                                       max_features='sqrt', bootstrap=False, n_jobs=1, random_state=seed)
    estimator.fit(x, y, sample_weight=weights)
    rule = ForestRule.from_estimator(estimator, FEATURE_NAMES, manifest_sha=manifest_sha)
    np.testing.assert_allclose(rule.score(x), estimator.predict_proba(x)[:, 1] - .5, atol=1e-12, rtol=0)
    threshold = all_views(rows, FEATURE_NAMES, 'threshold')
    if len(threshold) != 12:
        raise ValueError('Expected twelve raw threshold views')
    rule = choose_threshold(rule, threshold)
    totals = {d: float(sum(w for w, (_, domain, _) in zip(weights, identities) if domain == d))
              for d in ('rr', 'chimera')}
    return rule, {'records': len(x), 'domain_weight_sums': totals,
                  'sources': len({s for s, _, _ in identities}), 'threshold_variant': 'raw'}
