"""Typed numeric-row storage and metadata-only overrides without vector copies."""

from collections import ChainMap

import numpy as np
import pytest

from palimpsest.evaluation.numeric_features import numeric_rows


def test_numeric_rows_keep_values_and_metadata():
    a=np.array([[1.,2.],[3.,4.]])
    rows=numeric_rows([{'label':'REAL'},{'label':'FAKE'}],a,('x','y'))
    assert rows[0]['x']==1. and rows[1]['label']=='FAKE'
    assert np.shares_memory(rows[0].vector,a)
    override=ChainMap({'label':'FAKE'},rows[0]);assert override['label']=='FAKE' and override['y']==2.
    assert dict(rows[0])=={'label':'REAL','x':1.,'y':2.}
    with pytest.raises(ValueError):numeric_rows([{'x':1}],a[:1],('x','y'))
    with pytest.raises(ValueError):numeric_rows([{}],np.array([[np.nan]]),('x',))
    with pytest.raises(ValueError):numeric_rows([{}],a[:1],('x','x'))
