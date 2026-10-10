"""Deterministic source-disjoint stage partition inside a supplied fit role."""

from collections import defaultdict
import hashlib
import json

import numpy as np


def source_half_split(records, labels, *, seed=20261008):
    y = np.asarray(labels)
    if y.shape != (len(records),) or not len(records) or not set(y.tolist()) <= {0, 1}:
        raise ValueError('Invalid split labels')
    identities, strata = {}, defaultdict(list)
    for row, label in zip(records, y):
        if row['role'] != 'fit':
            raise ValueError('Source split requires fit-only records')
        source = row['domain'], row['src']
        stratum = row['domain'], row['scene'], int(label)
        if source in identities and identities[source] != stratum:
            raise ValueError('Conflicting split source/class/scene')
        identities[source] = stratum
    if len(identities) % 2:
        raise ValueError('Even source count required')
    for source, stratum in sorted(identities.items()):
        strata[stratum].append(source)
    basis = set(); round_up = False
    def rank(source):
        payload = json.dumps([seed, *source], ensure_ascii=False, separators=(',', ':'))
        return hashlib.sha256(payload.encode('utf-8')).hexdigest(), source
    for _, sources in sorted(strata.items()):
        if len(sources) < 2:
            raise ValueError('Split stratum lacks two sources')
        count = len(sources)//2
        if len(sources) % 2:
            count += int(round_up); round_up = not round_up
        basis.update(sorted(sources, key=rank)[:count])
    if len(basis)*2 != len(identities):
        raise ValueError('Unbalanced source-stage budget')
    return np.array([(row['domain'], row['src']) in basis for row in records], dtype=bool)
