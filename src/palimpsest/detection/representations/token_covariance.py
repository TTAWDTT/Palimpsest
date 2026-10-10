"""Fixed projected covariance square root of a token bag.

Retains some cross-coordinate information omitted by diagonal spread. Feature-bag
permutation/affine properties do not imply invariance to physical image channels.
"""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def fixed_projection(channels, components=32, seed=20261008):
    if (not isinstance(channels,int) or not isinstance(components,int)
            or not 1<=components<=channels):
        raise ValueError('Invalid covariance projection dimensions')
    matrix = np.random.default_rng(seed).normal(size=(channels,components))
    q,_ = np.linalg.qr(matrix,mode='reduced')
    for j in range(components):
        if q[np.argmax(np.abs(q[:,j])),j]<0:q[:,j]*=-1
    q.setflags(write=False)
    return q


def projected_covariance(tokens, projection):
    x,p = np.asarray(tokens,np.float64),np.asarray(projection,np.float64)
    if (x.ndim!=2 or len(x)<2 or p.ndim!=2 or p.shape[0]!=x.shape[1]
            or not p.shape[1] or p.shape[1]>p.shape[0]
            or not np.isfinite(x).all() or not np.isfinite(p).all()
            or not np.allclose(p.T@p,np.eye(p.shape[1]),rtol=0,atol=1e-12)):
        raise ValueError('Invalid projected covariance inputs')
    # Subtract an actual row first: identical token bags stay exactly zero even
    # when their common offset cannot be reproduced by a long mean reduction.
    projected = (x-x[0])@p
    centered = projected-projected.mean(axis=0)
    covariance = centered.T@centered/len(x)
    trace = float(np.trace(covariance));indices = np.triu_indices(p.shape[1])
    if not np.isfinite(covariance).all() or not np.isfinite(trace) or trace<0:
        raise ValueError('Nonfinite or negative covariance')
    if trace==0:
        return np.zeros(len(indices[0])),{'covariance_trace':0.,'covariance_rank':0}
    normalized = (covariance+covariance.T)/(2*trace)
    eigenvalues,vectors = np.linalg.eigh(normalized)
    if eigenvalues.min() < -1e-10:raise ValueError('Projected covariance is not PSD')
    eigenvalues = np.maximum(eigenvalues,0)
    root = (vectors*np.sqrt(eigenvalues))@vectors.T
    root = (root+root.T)/2
    output = root[indices].copy();output[indices[0]!=indices[1]]*=np.sqrt(2)
    if not np.isfinite(output).all() or abs(np.linalg.norm(output)-1)>1e-10:
        raise ValueError('Covariance square-root norm differs')
    return output,{'covariance_trace':trace,'covariance_rank':int(np.sum(eigenvalues>1e-8))}
