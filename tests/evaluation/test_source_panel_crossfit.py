"""Three-encoding source-role plant; incomplete panel is refused before fitting."""

from types import SimpleNamespace

import numpy as np
import pytest

from palimpsest.evaluation.source_panel_crossfit import SourcePanel, calibrated_panel_crossfit, wrong_panel_sources


def planted_panel():
    panel = SourcePanel(('raw', 'q90', 'q60')); records = []
    for domain, scenes, count in (('rr', ('all',), 540), ('chimera', ('cat', 'church', 'horse'), 720)):
        for i in range(count):
            scene = scenes[i % len(scenes)]; label = 'FAKE' if (i//len(scenes)) % 2 else 'REAL'
            for condition in ('original', 'transfer', 'redigital'):
                for variant in panel.variants:
                    records.append(dict(domain=domain, scene=scene, src=str(i), label=label,
                                        condition=condition, variant=variant, role='fit'))
    index = {key: i for i, key in enumerate(sorted({(r['domain'], r['src']) for r in records}))}
    x = np.array([[index[r['domain'], r['src']], int(r['label'] == 'FAKE')] for r in records], float)
    return panel, records, x, x[:, 1].astype(int), x[:, 0].astype(int)


def test_complete_three_view_roles_and_known_rates():
    panel, records, x, y, sources = planted_panel(); calls = []
    def fit(values, labels, weights, groups, parameter):
        trained = set(values[:, 0]); assert len(trained) == 756 and len(values) == 6804
        state = {'trained': trained, 'cal': set()}; calls.append(state)
        def score(query):
            assert trained.isdisjoint(query[:, 0]) and state['cal'].isdisjoint(query[:, 0])
            return 2*query[:, 1]-1
        return SimpleNamespace(score=score, threshold=0.), {'fit_records': 6804, 'sources': 756}
    def calibrate(rule, views):
        assert len(views) == 36
        for rows, values, labels in views.values():
            assert all(r['role'] == 'threshold' for r in rows)
            assert calls[-1]['trained'].isdisjoint(values[:, 0]); calls[-1]['cal'].update(values[:, 0])
            assert np.array_equal(labels, values[:, 1])
        assert len(calls[-1]['cal']) == 252
        return rule, {'passed': True}
    results, chosen, _ = calibrated_panel_crossfit(records, x, y, np.ones(len(x)), sources,
        (.001,), ('id', 'class'), panel, fit, calibrate)
    result = results['0.001']
    assert len(calls) == 5 and len(result['scores']) == 11340 and chosen[0] == '0.001'
    assert result['minimum_all_scope_ba'] == '1' and result['maximum_all_scope_drop'] == '0'
    assert len(result['metrics']) == 45 and len(result['paired_drops']) == 75
    wrong = wrong_panel_sources(records, sources, seed=20261006)
    assert not np.array_equal(wrong, sources) and np.array_equal(np.bincount(wrong), np.bincount(sources))
    for group in range(1260): assert len(set(y[wrong == group])) == 1


def test_missing_strong_view_refused():
    panel, records, x, y, sources = planted_panel()
    with pytest.raises(ValueError, match='Incomplete'):
        calibrated_panel_crossfit(records[:-1], x[:-1], y[:-1], np.ones(len(x)-1), sources[:-1],
            (.001,), ('id', 'class'), panel, lambda *args: pytest.fail('Fitter should not run'), lambda *args: None)
    for folds in (0, 2, 5.5):
        with pytest.raises(ValueError, match='Invalid'): SourcePanel(('raw', 'q90', 'q60'), folds=folds)
