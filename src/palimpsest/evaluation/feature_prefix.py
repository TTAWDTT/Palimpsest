"""Exact shared feature-prefix/identity controls for extended pixel caches."""

import numpy as np

from .features import IDENTITY_FIELDS


def validate_cached_prefix(rows, reference, names, *, full_coverage):
    lookup = {(r['filename'], r['variant']): r for r in reference}
    keys = [(r['filename'], r['variant']) for r in rows]
    if (len(lookup) != len(reference) or len(set(keys)) != len(keys)
            or not set(keys) <= set(lookup) or (full_coverage and set(keys) != set(lookup))):
        raise ValueError('Shared feature-prefix coverage differs')
    for row in rows:
        old = lookup[(row['filename'], row['variant'])]
        if any(row[k] != old[k] for k in IDENTITY_FIELDS):
            raise ValueError('Shared feature-prefix identity differs')
        actual = np.array([float(row[n]) for n in names])
        expected = np.array([float(old[n]) for n in names])
        if not np.isfinite(actual).all() or not np.array_equal(actual, expected):
            raise ValueError('Shared feature-prefix values differ')
    return {'records': len(rows), 'dimensions': len(names), 'exact_equal': True,
            'full_coverage': full_coverage}
