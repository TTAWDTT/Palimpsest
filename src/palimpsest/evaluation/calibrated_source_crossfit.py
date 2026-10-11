"""Source CV that includes a disjoint training-role calibration fold.

Three source folds fit, one calibrates, one evaluates. These temporary subroles
never change the frozen inventory or use external threshold/selection sources.
"""

import numpy as np

from .source_crossfit import source_folds,crossfit_rates,choose_strength
from .numeric_features import numeric_rows
from .features import feature_views


def calibrated_crossfit(records,x,labels,weights,sources,strengths,variants,names,fit,calibrate,progress=None):
    x,y,w,s=map(np.asarray,(x,labels,weights,sources))
    if (x.ndim!=2 or len(x)!=len(records) or x.shape[1]!=len(names)
            or any(a.shape!=(len(records),) for a in (y,w,s))):
        raise ValueError('Invalid calibrated source CV arrays')
    folds=source_folds(records);ids=np.array([folds[r['domain'],r['src']] for r in records])
    results={}
    for strength in strengths:
        prediction=np.full(len(x),np.nan);diagnostics=[];cal_ids=np.empty(len(x),int)
        for held in range(5):
            cal_fold=(held+1)%5;train=(ids!=held)&(ids!=cal_fold);cal=ids==cal_fold;test=ids==held
            if (int(train.sum())!=4536 or int(cal.sum())!=1512 or int(test.sum())!=1512
                    or set(s[train])&set(s[cal]) or set(s[train])&set(s[test]) or set(s[cal])&set(s[test])):
                raise ValueError('Calibrated CV size/source leakage')
            _,groups=np.unique(s[train],return_inverse=True)
            rule,diagnostic=fit(x[train],y[train],w[train],groups,strength)
            temporary=[{**{k:r[k] for k in ('domain','scene','condition','variant','src')},
                        'role':'threshold','label':'FAKE' if label else 'REAL'}
                       for r,label,flag in zip(records,y,cal) if flag]
            rows=numeric_rows(temporary,x[cal],names)
            views={key+'/'+v:view for v in variants for p in (False,True)
                   for key,view in feature_views(rows,names,'threshold',processed=p,variant=v).items()}
            if len(views)!=24:raise ValueError('Inner calibration view count differs')
            fixed,calibration=calibrate(rule,views)
            scores=np.asarray(fixed.score(x[test]))-fixed.threshold
            if scores.shape!=(1512,) or not np.isfinite(scores).all():raise ValueError('Held calibrated scores invalid')
            prediction[test]=scores;cal_ids[test]=cal_fold
            diagnostic.update({'held_fold':held,'calibration_fold':cal_fold,'training_source_count':756,
                'calibration_source_count':252,'held_source_count':252,'calibration':calibration,
                'calibration_threshold':fixed.threshold,'calibration_records':1512})
            diagnostics.append(diagnostic)
        if not np.isfinite(prediction).all():raise ValueError('Missing calibrated OOF score')
        scored=[{**{k:r[k] for k in ('domain','scene','condition','variant','src','role')},
                 'label':'FAKE' if label else 'REAL','score':float(score),'fold':int(fold),
                 'calibration_fold':int(cal_fold)}
                for r,label,score,fold,cal_fold in zip(records,y,prediction,ids,cal_ids)]
        rates=crossfit_rates(scored,variants)
        results[str(strength)]={**rates,'fit_diagnostics':diagnostics,'scores':scored}
        if progress is not None:progress(strength,results[str(strength)])
    return results,choose_strength(results),folds
