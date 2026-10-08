"""Context is required and excluded from prediction; old zero readout is exact."""

import numpy as np
import pytest

from palimpsest.detection.algorithms.directed_score_margin import DirectedScoreFitter
from palimpsest.detection.algorithms.quantile_score_consistency import QuantileScoreFitter, QuantileScoreRule


def test_contextual_bank_zero_parity_and_portable_prediction(tmp_path):
    names = ('raw0', 'raw1', 'clip0', 'clip1', 'r0', 'r1', 'r2', 'shape')
    records = []; x = []; labels = []; groups = []
    for source, label in enumerate((0, 0, 1, 1)):
        for condition in ('original', 'a', 'b'):
            for variant in ('raw', 'q90', 'q60'):
                records.append(dict(domain='d', scene='s', src=str(source), role='fit', condition=condition, variant=variant))
                x.append([0., 0., 0., 0., 1/np.sqrt(2), 0., 1/np.sqrt(2),
                    (2*label-1)*(3 if condition == 'original' else 1)])
                labels.append(label); groups.append(source)
    x = np.array(x); y = np.array(labels); s = np.array(groups); w = np.ones(len(x))
    options = dict(legacy_dimensions=7, raw_dimensions=2, channels=2, filter_count=2)
    fitter = DirectedScoreFitter(names, variants=('raw', 'q90', 'q60'), **options)
    with pytest.raises(ValueError, match='records'):
        fitter.fit(x, y, w, s, 0)
    zero, _ = fitter.fit(x, y, w, s, 0, records=records)
    ordinary, _ = QuantileScoreFitter(names, **options).fit(x, y, w, s, 0)
    np.testing.assert_array_equal(zero.score(x), ordinary.score(x))
    rule, diagnostic = fitter.fit(x, y, w, s, 1, records=records)
    assert diagnostic['directed_edges'] == 60 and diagnostic['cross_source_edges'] == 0
    assert diagnostic['maximum_absolute_gradient'] <= 1e-5 and np.array_equal(rule.score(x)>0, y.astype(bool))
    path = tmp_path/'rule.json'; rule.save(path)
    np.testing.assert_array_equal(QuantileScoreRule.load(path).score(x), rule.score(x))
    assert fitter.audit()['initialization_source_readout_calls'] == 1
    assert fitter.audit()['directed_readout_fit_calls'] == 2
