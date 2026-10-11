"""Lazy metadata/NumPy feature rows, avoiding one Python object per coordinate."""

from collections.abc import Mapping
from itertools import chain

import numpy as np


class NumericFeatureRow(Mapping):
    def __init__(self,metadata,values,names,index):
        self.metadata=metadata;self.vector=values;self.names=names;self.index=index

    def __getitem__(self,key):
        if key in self.metadata:return self.metadata[key]
        try:return self.vector[self.index[key]]
        except KeyError:raise KeyError(key) from None

    def __iter__(self):return chain(self.metadata,self.names)

    def __len__(self):return len(self.metadata)+len(self.names)


def numeric_rows(metadata,values,names):
    names=tuple(names);a=np.asarray(values)
    if (a.ndim!=2 or a.shape!=(len(metadata),len(names)) or len(set(names))!=len(names)
            or not names or a.dtype.kind!='f' or not np.isfinite(a).all()):
        raise ValueError('Invalid numeric feature matrix/schema')
    name_set=set(names)
    if any(not name_set.isdisjoint(r) for r in metadata):raise ValueError('Numeric feature name shadows metadata')
    index={name:i for i,name in enumerate(names)}
    return [NumericFeatureRow(row,a[i],names,index) for i,row in enumerate(metadata)]
