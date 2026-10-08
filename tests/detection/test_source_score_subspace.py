"""Known covariance discrimination, cache-label isolation and portable rules."""

import numpy as np

from palimpsest.detection.algorithms.source_score_subspace import ScoreSubspaceFitter,ScoreSubspaceRule


def test_score_bank_reuses_only_identical_training_and_preserves_scores(tmp_path):
    a=np.array([0.,0.,.5,1/np.sqrt(2),.5]);b=a.copy();b[3]*=-1
    x=np.repeat([a,a,b,b],3,axis=0);y=np.repeat([0,0,1,1],3)
    w=np.ones(len(x));s=np.repeat(np.arange(4),3)
    names=('constant0','constant1','root0','root1','root2')
    fitter=ScoreSubspaceFitter(names,channels=2,filter_count=2)
    zero,_=fitter.fit(x,y,w,s,0);rule,_=fitter.fit(x,y,w,s,1)
    assert fitter.builds==1 and zero.bank is rule.bank
    assert np.array_equal(rule.score(x)>0,y.astype(bool))
    assert np.array_equal(rule.score(x),[rule.score([row])[0] for row in x])
    path=tmp_path/'rule.json';rule.save(path)
    assert np.array_equal(ScoreSubspaceRule.load(path).score(x),rule.score(x))
    flipped,_=fitter.fit(x,1-y,w,s,0)
    assert fitter.builds==2 and flipped.bank is not rule.bank
    assert np.array_equal(flipped.score(x)>0,(1-y).astype(bool))
