"""Real fold cardinality with synthetic features tests all three roles."""

from types import SimpleNamespace

import numpy as np
import pytest

from palimpsest.evaluation.calibrated_source_crossfit import calibrated_crossfit


def test_disjoint_fit_calibration_and_held_sources():
    records=[]
    for domain,scenes,count in (('rr',('all',),540),('chimera',('cat','church','horse'),720)):
        for i in range(count):
            scene=scenes[i%len(scenes)];label='FAKE' if (i//len(scenes))%2 else 'REAL'
            for condition in ('original','transfer','redigital'):
                for variant in ('raw','jpeg'):
                    records.append(dict(domain=domain,scene=scene,src=str(i),label=label,
                                        condition=condition,variant=variant,role='fit'))
    identities={key:i for i,key in enumerate(sorted({(r['domain'],r['src']) for r in records}))}
    x=np.array([[identities[r['domain'],r['src']],int(r['label']=='FAKE')] for r in records],float)
    y=x[:,1].astype(int);sources=x[:,0].astype(int);calls=[]

    def fit(values,labels,weights,groups,strength):
        trained=set(values[:,0]);assert len(trained)==756 and len(values)==4536
        state={'train':trained,'cal':set()};calls.append(state)
        def score(test):
            assert trained.isdisjoint(test[:,0]) and state['cal'].isdisjoint(test[:,0])
            return 2*test[:,1]-1
        return SimpleNamespace(score=score,threshold=0.),{'fit_records':len(values),'sources':756}

    def calibrate(rule,views):
        cal=set()
        for rows,values,labels in views.values():
            assert all(r['role']=='threshold' for r in rows)
            assert calls[-1]['train'].isdisjoint(values[:,0]);cal.update(values[:,0])
        assert len(cal)==252;calls[-1]['cal']=cal
        return rule,{'known_threshold':0}

    result,chosen,_=calibrated_crossfit(records,x,y,np.ones(len(x)),sources,(0,),('raw','jpeg'),
                                       ('source_id','truth'),fit,calibrate)
    assert len(calls)==5 and result['0']['minimum_all_scope_ba']=='1'
    assert result['0']['maximum_all_scope_drop']=='0' and chosen[0]=='0'
    assert all(r['calibration_fold']==(r['fold']+1)%5 for r in result['0']['scores'])
    # Merge every numeric source group: no fitted/calibrated/held partition may
    # proceed, regardless of distinct source strings in metadata.
    with pytest.raises(ValueError,match='leakage'):
        calibrated_crossfit(records,x,y,np.ones(len(x)),np.zeros(len(x),int),(0,),('raw','jpeg'),
                            ('source_id','truth'),fit,calibrate)
