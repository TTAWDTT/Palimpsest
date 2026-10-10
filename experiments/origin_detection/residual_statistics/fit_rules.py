"""Twelve equal-weight views and portable readout; never selection at fit."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from palimpsest.detection.algorithms.forest import ForestRule
from palimpsest.evaluation.features import feature_views
from experiments.origin_detection.ordinal_statistics.fit_rules import choose_threshold


def all_views(rows, names, role):
    return {**feature_views(rows, names, role, processed=False),
            **feature_views(rows, names, role, processed=True)}


def fit_rule(rows, names, *, seed, manifest_sha, shuffled=False):
    views = all_views(rows, names, 'fit')
    if len(views) != 12:
        raise ValueError("Expected twelve original/processed fit views")
    sources = {r['src']: r['label'] for r in rows if r['role']=='fit' and r['variant']=='raw'}
    if shuffled:
        rng = np.random.default_rng(seed)
        for domain, scene in sorted({(r['domain'],r['scene']) for r in rows if r['role']=='fit'}):
            ids = sorted({r['src'] for r in rows if r['role']=='fit' and r['domain']==domain and r['scene']==scene})
            sources.update(zip(ids, rng.permutation([sources[s] for s in ids]).tolist()))
    values, labels, weights = [], [], []
    for records, x, truth in views.values():
        if (len(set(truth)) != 2 or sum(truth==0) != sum(truth==1) or not np.isfinite(x).all()
                or any(r['role']!='fit' for r in records)):
            raise ValueError("Expected finite balanced fit view")
        values.append(x)
        labels.append(np.array([sources[r['src']]=='FAKE' for r in records],dtype=int))
        weights.append(np.full(len(x),1.0/len(x)))
    x = np.concatenate(values)
    estimator = RandomForestClassifier(n_estimators=128,max_depth=6,min_samples_leaf=8,
                                       max_features='sqrt',bootstrap=False,n_jobs=1,random_state=seed)
    estimator.fit(x,np.concatenate(labels),sample_weight=np.concatenate(weights))
    rule = ForestRule.from_estimator(estimator,names,manifest_sha=manifest_sha)
    np.testing.assert_allclose(rule.score(x),estimator.predict_proba(x)[:,1]-0.5,atol=1e-12,rtol=0)
    threshold = all_views(rows,names,'threshold')
    if len(threshold)!=12:
        raise ValueError("Expected twelve threshold views")
    return choose_threshold(rule,threshold)
