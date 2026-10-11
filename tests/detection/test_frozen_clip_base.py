"""Known storage coordinates for the separately pinned512-dimensional encoder."""

import numpy as np
import pytest

from palimpsest.detection.representations.frozen_clip_base import FEATURE_NAMES,shifted_base_unit


def test_base_unit_contract_and_invalid_inputs():
    x=np.zeros(512);x[3]=2
    value=shifted_base_unit(x)
    assert len(FEATURE_NAMES)==512 and value[3]==1 and np.all(np.delete(value,3)==.5)
    assert np.array_equal(value,shifted_base_unit(x*3))
    for bad in (np.zeros(512),np.ones(768),np.full(512,np.nan)):
        with pytest.raises(ValueError):shifted_base_unit(bad)
