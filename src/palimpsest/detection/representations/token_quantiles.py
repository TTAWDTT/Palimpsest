"""Finite standardized projection quantiles, not a full transport metric.

Bag permutation/positive scaling properties do not imply image-channel
invariance. Finite directions and quantile levels cannot identify every law.
"""

import numpy as np

LEVELS = (.05, .1, .2, .3, .4, .6, .7, .8, .9, .95)


def projected_quantiles(tokens, projection):
    x, p = np.asarray(tokens, np.float64), np.asarray(projection, np.float64)
    if (x.ndim != 2 or len(x) < 2 or not x.shape[1] or p.ndim != 2
            or p.shape[0] != x.shape[1] or not 1 <= p.shape[1] <= p.shape[0]
            or not np.isfinite(x).all() or not np.isfinite(p).all()
            or not np.allclose(p.T@p, np.eye(p.shape[1]), rtol=0, atol=1e-12)):
        raise ValueError('Invalid token quantile inputs')
    with np.errstate(over='ignore', invalid='ignore'):
        projected = (x-x[0])@p
    if not np.isfinite(projected).all():
        raise ValueError('Nonfinite projected tokens')
    delta = projected-np.median(projected, axis=0)
    mad = np.median(np.abs(delta), axis=0)
    peak = np.max(np.abs(delta), axis=0)
    scaled = np.divide(delta, peak, out=np.zeros_like(delta), where=peak > 0)
    rms = peak*np.sqrt(np.mean(scaled**2, axis=0))
    scale = np.where(mad > 0, mad, rms)
    standardized = np.divide(delta, scale, out=np.zeros_like(delta), where=scale > 0)
    result = np.quantile(standardized, LEVELS, axis=0, method='linear').T.reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite quantile descriptor')
    return result, {'quantile_mad_zero_directions': int(np.count_nonzero(mad == 0)),
        'quantile_constant_directions': int(np.count_nonzero(peak == 0)),
        'quantile_rms_fallback_directions': int(np.count_nonzero((mad == 0)&(peak > 0)))}
