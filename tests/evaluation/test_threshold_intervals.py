import numpy as np
import pytest

from palimpsest.evaluation.thresholds import threshold_interval


def test_hand_interval_open_boundary_and_conflicting_views():
    result = threshold_interval([-3, -2, -1, 0, -.5, .5, 1.5, 2.5], [0]*4+[1]*4)
    assert result['lower_inclusive'] == -1 and result['upper_exclusive'] == .5
    assert result['required_natural_correct'] == result['required_ai_correct'] == 3
    for cut, feasible in [(-1, True), (0, True), (.5, False), (-2, False)]:
        scores = np.array([-3, -2, -1, 0, -.5, .5, 1.5, 2.5])
        meets = np.mean(scores[:4] <= cut) >= .55 and np.mean(scores[4:] > cut) >= .55
        assert meets == feasible
    a = threshold_interval([-2, -1, 0, 1], [0, 0, 1, 1])
    b = threshold_interval([2, 3, 4, 5], [0, 0, 1, 1])
    assert a['feasible'] and b['feasible']
    assert max(a['lower_inclusive'], b['lower_inclusive']) >= min(a['upper_exclusive'], b['upper_exclusive'])
    assert not threshold_interval([0, 0], [0, 1])['feasible']
    for s, y, minimum in [([float('nan'), 0], [0, 1], .55), ([0, 1], [0, 0], .55), ([0, 1], [0, 1], 0)]:
        with pytest.raises(ValueError): threshold_interval(s, y, minimum)
