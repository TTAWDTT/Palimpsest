"""Median-anchored radial winsorization and normalized patch spread.

These are fixed feature-bag statistics, not Huber M-estimator optimization or
image-space invariance. Attention can spread a local corruption to every token.
"""

import numpy as np


def clipped_token_statistics(tokens):
    x=np.asarray(tokens,np.float64)
    if x.ndim!=2 or len(x)<2 or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('Invalid token matrix')
    anchor=np.median(x,axis=0);delta=x-anchor
    distances=np.sqrt(np.sum(delta*delta,axis=1));radius=float(np.median(distances))
    if not np.isfinite(distances).all():raise ValueError('Nonfinite token distance')
    factors=np.ones(len(x));np.divide(radius,distances,out=factors,where=distances>0)
    factors=np.minimum(factors,1.)
    clipped=anchor+delta*factors[:,None]
    center=np.mean(clipped,axis=0)
    spread=np.sqrt(np.mean((clipped-center)**2,axis=0));norm=float(np.linalg.norm(spread))
    unit=np.zeros_like(spread) if norm==0 else spread/norm
    if not np.isfinite(center).all() or not np.isfinite(unit).all():raise ValueError('Nonfinite token descriptor')
    return center,unit,{'radius':radius,'clipped_fraction':float(np.mean(factors<1.)),'spread_norm':norm}
