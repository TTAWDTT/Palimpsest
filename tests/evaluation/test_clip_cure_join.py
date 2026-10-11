"""Known full-width union path with raw unbounded coordinates and lineage gates."""

import numpy as np
import pytest

from experiments.origin_detection.clip_cure_complement.run_iteration import audited_join


def fixture_rows():
    inventory = [dict(filename='a.png', src='source', label='FAKE', role='fit',
                      domain='toy', scene='scene', condition='original', sha256='known')]
    return inventory, [{**inventory[0], 'variant': 'raw', 'clip': .25}], [{**inventory[0], 'variant': 'raw', 'cure': -7.}]


def test_known_unbounded_union_and_order():
    inventory, left, right = fixture_rows()
    joined = audited_join(left, right, inventory, first_names=('clip',), second_names=('cure',), variants=('raw',))
    assert joined == [{**left[0], 'cure': -7.}]
    np.testing.assert_array_equal([joined[0][n] for n in ('clip', 'cure')], [.25, -7.])


def test_actual_union_refuses_corrupted_lineage_and_coverage():
    inventory, left, right = fixture_rows()
    for a, b in ((left*2, right), (left, right*2), (left, []),
                 (left, [{**right[0], 'sha256': 'wrong'}]), (left, [{**right[0], 'cure': float('nan')}])):
        with pytest.raises(ValueError):
            audited_join(a, b, inventory, first_names=('clip',), second_names=('cure',), variants=('raw',))
    with pytest.raises(ValueError):
        audited_join(left, right, inventory, first_names=('clip',), second_names=('clip',), variants=('raw',))
