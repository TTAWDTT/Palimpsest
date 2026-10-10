"""Known six-view assembly and purposeful processing-name corruption."""

import pytest

from experiments.origin_detection.cure_mean_shift.run_iteration import EXPECTED_VIEWS, panel_views
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS


def test_known_panel_and_wrong_processing():
    rows = [{'domain': 'rr', 'scene': 'all', 'src': 'toy', 'role': 'fit',
             'label': 'REAL', 'condition': c, 'variant': v}
            for c in ('original', 'transfer', 'redigital') for v in VARIANTS[:2]]
    ordered, panels, views = panel_views(rows)
    assert len(ordered) == 6 and panels == ['rr/all']*6
    # Records follow lexical source-assembly order (JPEG precedes raw), while
    # EXPECTED_VIEWS describes a set and the mean estimator's reference order.
    assert views == [c+'/'+v for c in ('original', 'processed1', 'processed2')
                     for v in ('jpeg90_444_after_resize256', 'raw')]
    assert set(views) == set(EXPECTED_VIEWS)
    wrong, _, corrupted = panel_views(rows, corrupt=True)
    assert wrong == ordered and sorted(corrupted) == sorted(views) and corrupted != views
    assert panel_views(rows, corrupt=True)[2] == corrupted
    with pytest.raises(ValueError):
        panel_views([r for r in rows if r['condition'] != 'redigital'])
