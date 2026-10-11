"""Calibrated source CV with an explicitly registered view panel.

The original six-view implementation remains immutable. This panel version
validates complete conditions/variants and derives row budgets from source
counts, so adding stronger views cannot bypass an older cardinality gate.
"""

from collections import defaultdict
from dataclasses import dataclass
import random

import numpy as np

from .source_crossfit import source_folds, crossfit_rates, choose_strength
from .numeric_features import numeric_rows
from .features import feature_views
from .training_fit import fit_training_rows


@dataclass(frozen=True)
class SourcePanel:
    variants: tuple[str, ...]
    sources: int = 1260
    conditions: int = 3
    folds: int = 5
    calibration_groups_per_variant: int = 12
    metric_groups_per_variant: int = 15
    physical_pairs_per_variant: int = 10

    def __post_init__(self):
        counts = (self.sources, self.conditions, self.folds, self.calibration_groups_per_variant,
                  self.metric_groups_per_variant, self.physical_pairs_per_variant)
        if (any(not isinstance(v, int) or isinstance(v, bool) for v in counts)
                or self.folds < 3 or not self.variants or any(not isinstance(v, str) or not v for v in self.variants)
                or len(set(self.variants)) != len(self.variants)
                or self.sources <= 0 or self.sources % self.folds
                or min(self.conditions, self.calibration_groups_per_variant,
                       self.metric_groups_per_variant, self.physical_pairs_per_variant) < 1):
            raise ValueError('Invalid source panel')

    @property
    def views_per_source(self): return self.conditions*len(self.variants)

    @property
    def metric_count(self): return self.metric_groups_per_variant*len(self.variants)

    @property
    def pair_count(self):
        n = len(self.variants)
        return self.physical_pairs_per_variant*n+self.metric_groups_per_variant*n*(n-1)//2

    def check(self, records):
        by_source = defaultdict(list)
        for row in records: by_source[row['domain'], row['src']].append(row)
        if len(by_source) != self.sources: raise ValueError('Panel source count differs')
        for rows in by_source.values():
            conditions = {r['condition'] for r in rows}
            identities = {(r['condition'], r['variant']) for r in rows}
            if (len(conditions) != self.conditions or len(rows) != self.views_per_source
                    or identities != {(c, v) for c in conditions for v in self.variants}
                    or any(r['role'] != 'fit' for r in rows)
                    or len({(r['scene'], r['label']) for r in rows}) != 1):
                raise ValueError('Incomplete/conflicting source panel')


def panel_calibration_views(records, values, labels, mask, names, panel):
    temporary = [{**{k: r[k] for k in ('domain', 'scene', 'condition', 'variant', 'src')},
                  'role': 'threshold', 'label': 'FAKE' if y else 'REAL'}
                 for r, y, selected in zip(records, labels, mask) if selected]
    rows = numeric_rows(temporary, values[mask], names)
    views = {key+'/'+variant: view for variant in panel.variants for processed in (False, True)
             for key, view in feature_views(rows, names, 'threshold', processed=processed, variant=variant).items()}
    if len(views) != panel.calibration_groups_per_variant*len(panel.variants):
        raise ValueError('Panel calibration view coverage differs')
    return views


def panel_masks(records, sources, panel, held=0):
    panel.check(records); folds = source_folds(records, folds=panel.folds)
    ids = np.array([folds[r['domain'], r['src']] for r in records]); cal_fold = (held+1) % panel.folds
    train = (ids != held) & (ids != cal_fold); cal = ids == cal_fold; test = ids == held
    per_fold = panel.sources//panel.folds; views = panel.views_per_source
    if (int(train.sum()) != (panel.folds-2)*per_fold*views or int(cal.sum()) != per_fold*views
            or int(test.sum()) != per_fold*views or set(sources[train]) & set(sources[cal])
            or set(sources[train]) & set(sources[test]) or set(sources[cal]) & set(sources[test])):
        raise ValueError('Panel role cardinality/source leakage')
    return train, cal, test, ids, folds


def calibrated_panel_crossfit(records, x, labels, weights, sources, parameters, names,
                              panel, fit, calibrate, progress=None, *, record_aware=False):
    x, y, w, s = map(np.asarray, (x, labels, weights, sources))
    if (x.ndim != 2 or len(x) != len(records) or x.shape[1] != len(names)
            or any(a.shape != (len(records),) for a in (y, w, s))):
        raise ValueError('Invalid panel CV dimensions')
    panel.check(records); results = {}; folds = source_folds(records, folds=panel.folds)
    per_fold = panel.sources//panel.folds
    for parameter in parameters:
        prediction = np.full(len(x), np.nan); cal_ids = np.empty(len(x), int); diagnostics = []
        for held in range(panel.folds):
            train, cal, test, ids, actual = panel_masks(records, s, panel, held)
            if actual != folds: raise ValueError('Panel folds changed')
            _, groups = np.unique(s[train], return_inverse=True)
            training_records = tuple(r for r, flag in zip(records, train) if flag)
            rule, d = fit_training_rows(fit, x[train], y[train], w[train], groups, parameter,
                records=training_records, record_aware=record_aware)
            views = panel_calibration_views(records, x, y, cal, names, panel)
            fixed, calibration = calibrate(rule, views)
            scores = np.asarray(fixed.score(x[test]))-fixed.threshold
            if scores.shape != (per_fold*panel.views_per_source,) or not np.isfinite(scores).all():
                raise ValueError('Panel held margin invalid')
            prediction[test] = scores; cal_ids[test] = (held+1) % panel.folds
            d.update({'held_fold': held, 'calibration_fold': (held+1) % panel.folds,
                'training_source_count': (panel.folds-2)*per_fold, 'calibration_source_count': per_fold,
                'held_source_count': per_fold, 'calibration_records': int(cal.sum()),
                'calibration': calibration, 'calibration_threshold': fixed.threshold})
            diagnostics.append(d)
        if not np.isfinite(prediction).all(): raise ValueError('Missing panel OOF score')
        scored = [{**{k: r[k] for k in ('domain', 'scene', 'condition', 'variant', 'src', 'role')},
                   'label': 'FAKE' if label else 'REAL', 'score': float(score), 'fold': int(fold),
                   'calibration_fold': int(cal_fold)}
                  for r, label, score, fold, cal_fold in zip(records, y, prediction, ids, cal_ids)]
        rates = crossfit_rates(scored, panel.variants)
        if len(rates['metrics']) != panel.metric_count or len(rates['paired_drops']) != panel.pair_count:
            raise ValueError('Panel metric/pair coverage differs')
        results[str(parameter)] = {**rates, 'fit_diagnostics': diagnostics, 'scores': scored}
        if progress is not None: progress(parameter, results[str(parameter)])
    return results, choose_strength(results), folds


def wrong_panel_sources(records, sources, *, seed):
    sources = np.asarray(sources)
    if sources.shape != (len(records),) or sources.dtype.kind not in 'iu':
        raise ValueError('Wrong-panel source dimensions differ')
    strata = defaultdict(list)
    for index, row in enumerate(records):
        strata[tuple(row[k] for k in ('domain', 'scene', 'label', 'condition', 'variant'))].append(index)
    randomizer = random.Random(seed); wrong = sources.copy()
    for key in sorted(strata):
        indices = strata[key]; values = sources[indices].tolist(); randomizer.shuffle(values); wrong[indices] = values
    if np.array_equal(wrong, sources) or not np.array_equal(np.bincount(wrong), np.bincount(sources)):
        raise ValueError('Wrong-panel control did not alter complete panels')
    return wrong
