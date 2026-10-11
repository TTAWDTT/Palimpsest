"""Paired descriptor changes relative to an original-image class contrast.

Descriptive fit-only diagnostics, not classifier accuracy or invariance proofs.
"""

import numpy as np


def paired_descriptor_drift(before, after, labels, scale, *, scale_floor=.001):
    a, b, y, s = (np.asarray(before, float), np.asarray(after, float),
        np.asarray(labels), np.asarray(scale, float))
    if (a.ndim != 2 or not a.size or a.shape != b.shape or y.shape != (len(a),)
            or s.shape != (a.shape[1],) or set(y.tolist()) != {0, 1}
            or not np.isfinite(scale_floor) or scale_floor <= 0
            or not all(np.isfinite(v).all() for v in (a, b, s)) or np.any(s < 0)):
        raise ValueError('Invalid paired descriptor dimensions, labels or values')
    s = np.maximum(s, scale_floor)
    original = (a[y == 1].mean(axis=0)-a[y == 0].mean(axis=0))/s
    processed = (b[y == 1].mean(axis=0)-b[y == 0].mean(axis=0))/s
    delta = (b-a)/s
    mean_delta = delta.mean(axis=0)
    length = np.linalg.norm(original)*np.linalg.norm(processed)
    cosine = None if length == 0 else float(np.clip(original@processed/length, -1, 1))
    result = {'sources': len(a), 'real_sources': int(np.sum(y == 0)),
        'ai_sources': int(np.sum(y == 1)), 'dimensions': a.shape[1],
        'before_class_gap_rms': float(np.sqrt(np.mean(original**2))),
        'after_class_gap_rms': float(np.sqrt(np.mean(processed**2))),
        'paired_drift_rms': float(np.sqrt(np.mean(delta**2))),
        'mean_shift_rms': float(np.sqrt(np.mean(mean_delta**2))),
        'class_gap_cosine': cosine}
    if any(not np.isfinite(v) for k, v in result.items() if v is not None):
        raise ValueError('Nonfinite descriptor diagnostic')
    return result


def paired_indices(records, domain, before_condition, before_variant, after_condition, after_variant):
    """Match all sources in a domain; no silent intersection or label conflict."""
    lookup = {}
    identities = set()
    for index, row in enumerate(records):
        if row['role'] != 'fit' or row['label'] not in ('REAL', 'FAKE'):
            raise ValueError('Diagnostics require fit-only rows with declared labels')
        if row['domain'] != domain:
            continue
        key = row['src'], row['condition'], row['variant']
        if key in lookup:
            raise ValueError('Duplicate paired feature identity')
        lookup[key] = index
        identities.add(row['src'])
    if not identities:
        raise ValueError('No paired sources in domain')
    first, second, labels = [], [], []
    for source in sorted(identities):
        a, b = ((source, before_condition, before_variant), (source, after_condition, after_variant))
        if a not in lookup or b not in lookup:
            raise ValueError('Incomplete source-paired feature coverage')
        i, j = lookup[a], lookup[b]
        if records[i]['label'] != records[j]['label'] or records[i]['scene'] != records[j]['scene']:
            raise ValueError('Conflicting paired source metadata')
        first.append(i); second.append(j); labels.append(int(records[i]['label'] == 'FAKE'))
    return np.asarray(first), np.asarray(second), np.asarray(labels)
