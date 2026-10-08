"""Reusable fit-only source cross-validation for a fixed consistency grid.

The caller supplies a fitter and freezes its policy. Entire sources remain in
one fold; moment estimation belongs inside the fitter on training rows only.
"""

import numpy as np

from .source_crossfit import source_folds, crossfit_rates, choose_strength


def crossfit_strengths(records, values, labels, weights, sources, strengths,
                      variants, fit, *, fold_count=5, fold_seed=20261007, progress=None):
    """Return all OOF scores and the declared BA/decline choice, without outer data."""
    x, y, w, s = map(np.asarray, (values, labels, weights, sources))
    n = len(records)
    if (x.ndim != 2 or len(x) != n or y.shape != (n,) or w.shape != (n,)
            or s.shape != (n,) or s.dtype.kind not in 'iu' or not n
            or not np.isfinite(x).all() or not np.isfinite(w).all()
            or np.any(w <= 0) or not set(y.tolist()) <= {0, 1}
            or not strengths or len(set(strengths)) != len(strengths)
            or any(not np.isfinite(k) or k < 0 for k in strengths)):
        raise ValueError('Invalid source cross-fit arrays/grid')
    folds = source_folds(records, folds=fold_count, seed=fold_seed)
    ids = np.array([folds[r['domain'], r['src']] for r in records])
    source_keys = {}
    for r, group in zip(records, s):
        key = r['domain'], r['src']
        if int(group) in source_keys and source_keys[int(group)] != key:
            raise ValueError('Numeric source group merges different identities')
        source_keys[int(group)] = key
    if len(set(source_keys.values())) != len(source_keys):
        raise ValueError('Source identity split across numeric groups')
    results = {}
    for strength in strengths:
        predictions = np.full(n, np.nan)
        diagnostics = []
        for fold in range(fold_count):
            train, test = ids != fold, ids == fold
            if not np.any(train) or not np.any(test) or set(s[train]) & set(s[test]):
                raise ValueError('Empty or leaking source fold')
            _, train_sources = np.unique(s[train], return_inverse=True)
            rule, diagnostic = fit(x[train], y[train], w[train], train_sources, strength)
            scores = np.asarray(rule.score(x[test]))
            if scores.shape != (int(test.sum()),) or not np.isfinite(scores).all():
                raise ValueError('Invalid held-source predictions')
            predictions[test] = scores
            diagnostics.append(diagnostic)
        if not np.isfinite(predictions).all():
            raise ValueError('Incomplete out-of-fold predictions')
        scored = [{**{k: r[k] for k in ('domain', 'scene', 'condition', 'variant', 'src', 'role')},
                   'label': 'FAKE' if label else 'REAL', 'score': float(score), 'fold': int(fold)}
                  for r, label, score, fold in zip(records, y, predictions, ids)]
        result = crossfit_rates(scored, variants)
        results[str(strength)] = {**result, 'fit_diagnostics': diagnostics, 'scores': scored}
        if progress is not None:
            progress(strength, results[str(strength)])
    return results, choose_strength(results), folds
