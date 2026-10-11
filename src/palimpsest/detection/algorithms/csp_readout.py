"""Supervised CSP-style variance directions for fixed token covariances.

Class means and filters must be learned inside each training fold. This is a
conventional signal-processing adaptation, not a physical invariance guarantee.
"""

from dataclasses import asdict,dataclass,replace
import json

import numpy as np
from scipy.linalg import eigh

from palimpsest.detection.algorithms.readouts.stable_rule import StableRule
from palimpsest.detection.algorithms.readouts.consistent_source_risk import fit_consistent_source_risk


def root_matrix(vector,channels):
    values=np.asarray(vector,float);indices=np.triu_indices(channels)
    if values.shape!=(len(indices[0]),) or not np.isfinite(values).all():
        raise ValueError('Invalid covariance square-root vector')
    norm=float(np.linalg.norm(values))
    if norm!=0 and abs(norm-1)>1e-10:raise ValueError('Covariance root norm differs')
    upper=values.copy();upper[indices[0]!=indices[1]]/=np.sqrt(2)
    matrix=np.zeros((channels,channels));matrix[indices]=upper
    matrix[(indices[1],indices[0])]=upper
    return matrix


def positive_covariance(root,shrinkage):
    channels=len(root);covariance=root@root
    if np.trace(covariance)==0:return np.eye(channels)/channels
    return (1-shrinkage)*covariance+shrinkage*np.eye(channels)/channels


@dataclass(frozen=True)
class CSPMap:
    feature_names:tuple[str,...]
    channels:int
    filters:tuple[tuple[float,...],...]
    shrinkage:float=.05

    def __post_init__(self):
        filters=np.asarray(self.filters)
        if (self.channels<1 or len(set(self.feature_names))!=len(self.feature_names)
                or self.prefix_size<0 or filters.ndim!=2 or filters.shape[0]!=self.channels
                or not 1<=filters.shape[1]<=self.channels or not np.isfinite(filters).all()
                or not np.isfinite(self.shrinkage) or not 0<self.shrinkage<1
                or not np.allclose(np.linalg.norm(filters,axis=0),1,atol=1e-12,rtol=0)):
            raise ValueError('Invalid CSP mapper')

    @property
    def prefix_size(self):return len(self.feature_names)-self.channels*(self.channels+1)//2

    @property
    def output_names(self):
        return self.feature_names[:self.prefix_size]+tuple(f'csp/log_variance/{i}' for i in range(len(self.filters[0])))

    def transform(self,values):
        x=np.asarray(values,float);filters=np.asarray(self.filters)
        if x.ndim!=2 or x.shape[1]!=len(self.feature_names) or not np.isfinite(x).all():
            raise ValueError('Invalid CSP readout input')
        output=[]
        # Fixed per-row path preserves single/batch scores to the last bit.
        for row in x:
            root=root_matrix(row[self.prefix_size:],self.channels)
            if np.any(root):
                variances=(1-self.shrinkage)*np.sum((root@filters)**2,axis=0)+self.shrinkage/self.channels
            else:variances=np.full(filters.shape[1],1/self.channels)
            if np.any(variances<=0) or not np.isfinite(variances).all():
                raise ValueError('Invalid filtered covariance power')
            output.append(np.r_[row[:self.prefix_size],np.log(variances/variances.sum())])
        return np.asarray(output).reshape((len(x),len(self.output_names)))


def fit_csp_map(values,labels,weights,*,feature_names,channels=32,filter_count=16,shrinkage=.05):
    x,y,w=np.asarray(values,float),np.asarray(labels),np.asarray(weights,float)
    if (x.ndim!=2 or not len(x) or x.shape[1]!=len(feature_names) or y.shape!=(len(x),)
            or set(y.tolist())!={0,1} or w.shape!=y.shape or not np.isfinite(x).all()
            or not np.isfinite(w).all() or np.any(w<=0) or not 1<=filter_count<=channels
            or not 0<shrinkage<1):raise ValueError('Invalid CSP training inputs')
    prefix=len(feature_names)-channels*(channels+1)//2
    if prefix<0:raise ValueError('CSP covariance schema differs')
    covariances=np.array([positive_covariance(root_matrix(row[prefix:],channels),shrinkage) for row in x])
    means=[np.sum(covariances[y==label]*w[y==label,None,None],axis=0)/w[y==label].sum() for label in (0,1)]
    eigenvalues,vectors=eigh(means[1],means[0]+means[1])
    selected=np.argsort(-np.abs(eigenvalues-.5),kind='stable')[:filter_count]
    filters=vectors[:,selected].copy();filters/=np.linalg.norm(filters,axis=0)
    for j in range(filter_count):
        if filters[np.argmax(np.abs(filters[:,j])),j]<0:filters[:,j]*=-1
    residual=means[1]@filters-(means[0]+means[1])@filters*eigenvalues[selected]
    if np.max(np.abs(residual))>1e-10:raise ValueError('CSP generalized eigensystem residual differs')
    mapper=CSPMap(tuple(feature_names),channels,tuple(tuple(row) for row in filters),shrinkage)
    return mapper,{'filter_count':filter_count,'covariance_shrinkage':shrinkage,
        'selected_eigenvalues':eigenvalues[selected].tolist(),'maximum_eigen_residual':float(np.max(np.abs(residual))),
        'mapping_scope':'Class covariance means from supplied training fold only;not independent token assumptions'}


@dataclass(frozen=True)
class CSPRule:
    mapper:CSPMap
    readout:StableRule

    def __post_init__(self):
        if self.mapper.output_names!=self.readout.feature_names:raise ValueError('CSP head schema differs')

    @property
    def threshold(self):return self.readout.threshold

    def score(self,values):return self.readout.score(self.mapper.transform(values))

    def save(self,path):
        value={'schema':1,'kind':'csp_source_readout','mapper':asdict(self.mapper),'readout':self.readout.payload()}
        path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

    @classmethod
    def load(cls,path):
        value=json.loads(path.read_text())
        if (value['schema']!=1 or value['kind']!='csp_source_readout'
                or value['readout']['schema']!=1 or value['readout']['kind']!='paired_stability'):
            raise ValueError('Invalid CSP rule payload')
        params=value['mapper'];params['feature_names']=tuple(params['feature_names'])
        params['filters']=tuple(tuple(row) for row in params['filters'])
        head=value['readout']['rule']
        for key in ('feature_names','center','scale','weights'):head[key]=tuple(head[key])
        return cls(CSPMap(**params),StableRule(**head))


def fit_csp_source(values,labels,weights,sources,*,feature_names,strength,consistency_groups=None,
                   channels=32,filter_count=16,manifest_sha=''):
    mapper,mapping=fit_csp_map(values,labels,weights,feature_names=feature_names,channels=channels,
                              filter_count=filter_count,shrinkage=.05)
    head,diagnostic=fit_consistent_source_risk(mapper.transform(values),labels,weights,sources,
        strength=strength,consistency_groups=consistency_groups,feature_names=mapper.output_names,
        temperature=.1,ridge=.01,scale_floor=.001,maximum_iterations=2000,
        gradient_tolerance=1e-5,manifest_sha=manifest_sha)
    return CSPRule(mapper,head),{**diagnostic,'csp_mapping':mapping}


def calibrate_csp_rule(rule,views,calibrate):
    mapped={key:(records,rule.mapper.transform(values),labels)
            for key,(records,values,labels) in views.items()}
    head,diagnostic=calibrate(rule.readout,mapped)
    return replace(rule,readout=head),diagnostic
