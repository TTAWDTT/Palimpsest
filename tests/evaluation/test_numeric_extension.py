"""A signed feature prefix cannot silently change when adding new coordinates."""

from types import SimpleNamespace

import numpy as np
import pytest

from palimpsest.evaluation.numeric_extension import check_prefix


def test_extension_prefix_identity_and_corruption_refused():
    reference=np.array([1.,-2.]);feature=SimpleNamespace(values=np.array([1.,-2.,.5]),probability_fake=.7)
    check_prefix(feature,reference,.7)
    feature.values[1]+=.001
    with pytest.raises(ValueError,match='prefix'):check_prefix(feature,reference,.7)
    feature.values[1]=-2.
    with pytest.raises(ValueError,match='probability'):check_prefix(feature,reference,.8)
