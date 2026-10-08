"""Cosine drift under a query's own perturbation, without a source reference.

Inputs use the existing (unit vector + 1)/2 storage. Sensitivity is prior art
(RIGID/MINDER/intermediate representations), not an invariance guarantee.
"""

import numpy as np


def shifted_cosine_drift(query, perturbed):
    """Return 1-cosine, restoring affine storage before computing the angle."""
    a, b = np.asarray(query, dtype=np.float64), np.asarray(perturbed, dtype=np.float64)
    if (a.ndim != 1 or a.shape != b.shape or not len(a)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or min(a.min(), b.min()) < 0 or max(a.max(), b.max()) > 1):
        raise ValueError('Invalid shifted query embeddings')
    a, b = 2*a-1, 2*b-1
    denominator = np.linalg.norm(a)*np.linalg.norm(b)
    if denominator <= 1e-12:
        raise ValueError('Degenerate query embedding')
    return float(1-np.clip(np.dot(a, b)/denominator, -1, 1))
