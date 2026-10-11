"""Wrong-source control must keep only registered training views."""

import numpy as np

from experiments.origin_detection.semantic_group_risk.run_iteration import wrong_weak_pairs, VARIANTS


def test_wrong_pairs_exclude_extra_encoding_without_changing_weak_control():
    rows = []
    for domain in ('rr', 'chimera'):
        for label in ('REAL', 'FAKE'):
            for source in range(2):
                for condition in ('original', 'a', 'b'):
                    for index, variant in enumerate(VARIANTS[:3]):
                        rows.append({'domain': domain, 'scene': 'test', 'label': label,
                            'src': f'{domain}/{label}/{source}', 'role': 'fit',
                            'condition': condition, 'variant': variant,
                            'feature': source * 100 + index + len(condition)})
    complete, weights, control = wrong_weak_pairs(rows, ('feature',), 20261006)
    weak, weak_weights, weak_control = wrong_weak_pairs(
        [r for r in rows if r['variant'] in VARIANTS[:2]], ('feature',), 20261006)
    assert complete.shape == (40, 1)
    assert control['same_source_pairs'] == 0 and control['pairs'] == 40
    assert np.isclose(weights.sum(), 1)
    np.testing.assert_array_equal(complete, weak)
    np.testing.assert_array_equal(weights, weak_weights)
    assert control == weak_control
