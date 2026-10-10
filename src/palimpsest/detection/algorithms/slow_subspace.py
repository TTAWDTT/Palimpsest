"""Fit-only linear slow subspaces of matched source views.

Finite paired differences replace time derivatives. Variance normalization
prevents constants; neither it nor a retained class-mean direction guarantees
classification after an unseen process. These are empirical linear adaptations.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space

from palimpsest.detection.algorithms.readouts.stable_rule import StableRule


@dataclass(frozen=True)
class SlowSubspace:
    center: np.ndarray
    basis: np.ndarray
    mode: str

    def transform(self, values):
        x=np.asarray(values,float)
        if x.ndim!=2 or x.shape[1]!=len(self.center) or not np.isfinite(x).all():
            raise ValueError('Invalid subspace input')
        # Identical reduction order for scalar and batch execution.
        return np.array([self.basis.T@(row-self.center) for row in x]).reshape((len(x),self.basis.shape[1]))


def fit_slow_subspace(values,labels,weights,sources,*,dimension,mode,eigen_floor=1e-6):
    x,y,w,s=(np.asarray(v) for v in (values,labels,weights,sources))
    if (x.ndim!=2 or not len(x) or y.shape!=(len(x),) or set(y.tolist())!={0,1}
            or w.shape!=(len(x),) or s.shape!=(len(x),) or s.dtype.kind not in 'iu'
            or not np.isfinite(x).all() or not np.isfinite(w).all() or np.min(w)<=0
            or not 0<eigen_floor<1 or dimension<1 or int(dimension)!=dimension
            or mode not in ('pca','slow','protected_slow')):
        raise ValueError('Invalid slow-subspace training')
    w=w/w.sum();center=np.sum(w[:,None]*x,axis=0);z=x-center
    covariance=z.T@(w[:,None]*z)
    difference=np.zeros_like(covariance)
    for source in np.unique(s):
        index=np.flatnonzero(s==source);count=len(index)
        if count<2 or len(np.unique(y[index]))!=1 or not np.allclose(w[index],w[index[0]],atol=1e-15,rtol=0):
            raise ValueError('Source requires equal-weight views and one class')
        views=x[index];local=views-views.mean(axis=0)
        # Mean outer product of all count*(count-1)/2 distinct differences.
        difference+=w[index].sum()*2/(count-1)*(local.T@local)
    eigen,vectors=np.linalg.eigh(covariance)
    keep=eigen>eigen[-1]*eigen_floor
    if np.count_nonzero(keep)<dimension:
        raise ValueError('Insufficient covariance rank')
    eigen=eigen[keep];vectors=vectors[:,keep];whitening=vectors/np.sqrt(eigen)
    slow=whitening.T@difference@whitening
    slow=(slow+slow.T)/2
    means=[np.sum(x[y==c]*w[y==c,None],axis=0)/w[y==c].sum() for c in (0,1)]
    class_difference=whitening.T@(means[1]-means[0])
    energy=float(class_difference@class_difference)
    if mode=='pca':
        directions=np.eye(len(eigen))[:,-dimension:]
    elif mode=='slow':
        _,directions=np.linalg.eigh(slow);directions=directions[:,:dimension]
    else:
        if energy<1e-12:raise ValueError('No fit class-mean direction to protect')
        protected=class_difference/np.sqrt(energy)
        complement=null_space(protected[None,:])
        if dimension==1:directions=protected[:,None]
        else:
            _,rest=np.linalg.eigh(complement.T@slow@complement)
            directions=np.c_[protected,complement@rest[:,:dimension-1]]
    basis=whitening@directions
    whitening_error=float(np.max(np.abs(basis.T@covariance@basis-np.eye(dimension))))
    if whitening_error>1e-8:raise ValueError('Subspace variance constraint failed')
    for a in (center,basis):a.flags.writeable=False
    return SlowSubspace(center,basis,mode),{
        'covariance_rank':len(eigen),'dimension':dimension,'mode':mode,'eigen_floor':eigen_floor,
        'whitening_max_error':whitening_error,'mean_paired_energy':float(np.trace(basis.T@difference@basis)/dimension),
        'fit_class_mean_energy':energy,
        'retained_class_mean_energy_fraction':float(np.sum((directions.T@class_difference)**2)/energy) if energy>1e-12 else None,
        'scope':'Fit-only finite source-view covariance;no temporal or unseen-process guarantee'}


def collapse_readout(subspace,head,*,feature_names):
    """Algebraically eliminate the fitted linear subspace at single-image inference."""
    if (len(head.feature_names)!=subspace.basis.shape[1] or len(feature_names)!=len(subspace.center)):
        raise ValueError('Subspace/head schema differs')
    v=np.asarray(head.weights)/np.asarray(head.scale)
    weight=subspace.basis@v
    bias=head.bias-float(np.asarray(head.center)@v)
    return StableRule(tuple(feature_names),tuple(subspace.center),tuple(np.ones(len(weight))),
                      tuple(weight),bias,head.strength,head.ridge,head.threshold,head.fit_manifest_sha256)
