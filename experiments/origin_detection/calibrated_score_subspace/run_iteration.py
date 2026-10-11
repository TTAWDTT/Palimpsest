"""Whole fit-role source CV includes the calibration policy before selection."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.models.frozen_features.cure.source_score_subspace import ScoreSubspaceFitter, ScoreSubspaceRule, calibrate_score_subspace
from palimpsest.evaluation.calibrated_source_crossfit import calibrated_crossfit
from palimpsest.evaluation.source_consistency_campaign import run_source_consistency
from palimpsest.evaluation.source_crossfit import source_folds
from palimpsest.evaluation.numeric_features import numeric_rows
from palimpsest.evaluation.features import feature_views
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR,REPO_ROOT
from experiments.origin_detection.source_score_subspace.run_iteration import (
    PARENT,NAMES,CONFIG,STRENGTHS,VARIANTS,inputs,training,code_pins as input_pins,
    class_threshold,score_rule,wrong_sources,null_intervals,write_json)

OUTPUT=WORK_DIR/'robust_statistics/calibrated_score_subspace'


def code_pins():
    pins=input_pins();paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md',
        REPO_ROOT/'src/palimpsest/evaluation/calibrated_source_crossfit.py',
        REPO_ROOT/'tests/evaluation/test_calibrated_source_crossfit.py']
    pins.update({str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in sorted(paths)})
    return pins


def inner_views(records,x,truth,mask):
    temporary=[{**{k:r[k] for k in ('domain','scene','condition','variant','src')},
                'role':'threshold','label':'FAKE' if label else 'REAL'}
               for r,label,flag in zip(records,truth,mask) if flag]
    rows=numeric_rows(temporary,x[mask],NAMES)
    return {key+'/'+v:view for v in VARIANTS[:2] for p in (False,True)
            for key,view in feature_views(rows,NAMES,'threshold',processed=p,variant=v).items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'iteration.json').exists() or (OUTPUT/'crossfit.json').exists():
        raise FileExistsError('Preserve calibrated source experiment')
    control=json.loads((OUTPUT/'software_controls.json').read_text())
    if not control['passed'] or control['code_pins']!=code_pins():raise ValueError('CalibratedCV controls changed')
    rows,parent=inputs();manifest=parent['inventory_sha256'];records,x,truth,weights,sources=training(rows)
    fitter=ScoreSubspaceFitter(NAMES,manifest_sha=manifest);folds=source_folds(records)
    ids=np.array([folds[r['domain'],r['src']] for r in records])
    if args.pilot:
        if (OUTPUT/'pilot.json').exists():raise FileExistsError('Preserve calibrated pilot')
        train=(ids!=0)&(ids!=1);cal=ids==1;held=ids==0;_,groups=np.unique(sources[train],return_inverse=True)
        views=inner_views(records,x,truth,cal)
        times=[];diagnostics=[];rule=None
        with threadpool_limits(limits=1):
            for strength in (10,1):
                start=perf_counter();rule,d=fitter.fit(x[train],truth[train],weights[train],groups,strength)
                rule,d['inner_calibration']=calibrate_score_subspace(rule,views,class_threshold)
                scores=rule.score(x[held])-rule.threshold
                if len(scores)!=1512 or not np.isfinite(scores).all():raise ValueError('Pilot held margin invalid')
                times.append(perf_counter()-start);diagnostics.append(d)
        if fitter.builds!=1:raise ValueError('Calibrated pilot cache rebuilt')
        path=OUTPUT/'pilot_fold_rule.json';rule.save(path)
        projected=10*times[0]+30*times[1]
        write_json(OUTPUT/'pilot.json',{'passed':max(times)<=120,'cold_s':times[0],'warm_s':times[1],
            'projected40cv_s':projected,'train_rows':4536,'cal_rows':1512,'held_rows':1512,
            'diagnostics':diagnostics,'code_pins':code_pins(),'features_receipt_sha256':file_sha256(PARENT/'features.json'),
            'rule_sha256':file_sha256(path),'scope':'Budget with fit-role held scoring;not outer selection or independent data',
            'assumption':'10 cold+30 warm same-role calculations;fullfit/evaluation and loading excluded'})
        print(json.dumps({'cold_s':times[0],'warm_s':times[1],'projected40_s':projected}),flush=True)
        if max(times)>120:raise ValueError('Calibrated pilot too costly')
        return
    pilot=json.loads((OUTPUT/'pilot.json').read_text())
    if (not pilot['passed'] or pilot['code_pins']!=code_pins()
            or pilot['features_receipt_sha256']!=file_sha256(PARENT/'features.json')
            or pilot['rule_sha256']!=file_sha256(OUTPUT/'pilot_fold_rule.json')):
        raise ValueError('Calibrated pilot changed/failed')
    assignment,null=balanced_source_null(records,seed=CONFIG['seed'])
    prior=json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if null['assignment_sha256']!=prior['assignment_sha256']:raise ValueError('Calibrated null changed')
    sham=np.array([assignment[r['domain'],r['src']] for r in records]);cv={};start=perf_counter()
    with threadpool_limits(limits=1):
        for mode,labels in (('truth',truth),('null',sham)):
            def progress(strength,result):
                write_json(OUTPUT/f'cv_{mode}_{strength}.json',result)
                print(json.dumps({'cv':mode,'strength':strength,'minimum_ba':result['minimum_all_scope_ba'],
                                  'maximum_drop':result['maximum_all_scope_drop']}),flush=True)
            cv[mode],_,actual=calibrated_crossfit(records,x,labels,weights,sources,STRENGTHS,VARIANTS[:2],NAMES,
                fitter.fit,lambda r,v:calibrate_score_subspace(r,v,class_threshold),progress)
            if actual!=folds:raise ValueError('Calibrated source folds changed')
    cv_elapsed=perf_counter()-start
    write_json(OUTPUT/'inner_roles.json',{'fold_sources':{'/'.join(k):v for k,v in folds.items()},
        'calibration_fold_rule':'(held+1)%5','training_folds':'Remaining3','train_sources':756,
        'cal_sources':252,'held_sources':252,'cv_elapsed_s':cv_elapsed,
        'null_inner_calibration':'Pseudo labels;final threshold still true labels','code_pins':code_pins()})
    zero=None

    def calibrate(rule,views):
        nonlocal zero
        fixed,d=calibrate_score_subspace(rule,views,class_threshold)
        if zero is None:zero=fixed
        return fixed,d

    def zero_control(margins):
        selected=[r for r in rows if r['role']=='selection']
        native=np.array([[float(r[n]) for n in NAMES] for r in selected])
        path=OUTPUT/'zero_portable_control.json';zero.save(path);restored=ScoreSubspaceRule.load(path)
        values=zero.score(native)
        if (len(values)!=5040 or not np.array_equal(values,zero.readout.score(zero.bank.transform(native)))
                or not np.array_equal(values,restored.score(native))
                or not np.array_equal(values-zero.threshold,[r['score'] for r in margins])):
            raise ValueError('Calibrated zero portable identity differs')
        return {'records':5040,'portable_mapped_loaded_margin_exact':True,'scope':'Same-rule software identity only'}

    run_source_consistency(rows=rows,names=NAMES,records=records,x=x,truth=truth,sham=sham,
        weights=weights,sources=sources,strengths=STRENGTHS,variants=VARIANTS[:2],output=OUTPUT,
        fit=fitter.fit,calibrate=calibrate,score=lambda a,b,c:score_rule(a,b,c,CONFIG),
        wrong_sources=wrong_sources(rows,sources),zero_reference=zero_control,null_intervals=null_intervals,
        pins=code_pins(),manifest_sha=manifest,assignment_sha=null['assignment_sha256'],
        features_sha=file_sha256(PARENT/'features.json'),prior=cv)
    if fitter.builds!=12:raise ValueError('Calibrated cache bank count differs')
    write_json(OUTPUT/'bank_cache_audit.json',{'passed':True,'banks':12,'base_fits':36,
        'cv_elapsed_s':cv_elapsed,'training_array_keys':sorted(fitter.bases),
        'scope':'PrecomputedCV comes from current calibrated role split;not old score reuse'})


if __name__=='__main__':main()
