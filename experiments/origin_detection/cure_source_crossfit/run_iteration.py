"""Fit-only source CV for consistency strength, then fixed outer evaluation."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.readouts.consistent_source_risk import fit_consistent_source_risk
from palimpsest.detection.representations.frozen_cure import FULL_NAMES
from palimpsest.evaluation.source_crossfit import source_folds,crossfit_rates,choose_strength
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.features import feature_views
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT,WORK_DIR
from experiments.origin_detection.semantic_gaussian.run_iteration import cure_inputs,null_intervals
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS
from experiments.origin_detection.cure_readout.prepare_features import OUTPUT as CURE
from experiments.origin_detection.paired_kernel.run_iteration import wrong_sources
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.phase_statistics.run_iteration import write_json

OUTPUT=WORK_DIR/'robust_statistics/cure_source_crossfit'
STRENGTHS=(0,.1,1,10)
CONFIG={'seed':20261006,'bootstrap_repetitions':2000,'provisional_final_ba':.8,'provisional_final_drop':.02}


def code_pins():
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md']
    paths+=[REPO_ROOT/p for p in ('src/palimpsest/evaluation/source_crossfit.py',
        'src/palimpsest/detection/algorithms/readouts/consistent_source_risk.py',
        'src/palimpsest/detection/algorithms/readouts/source_view_risk.py',
        'src/palimpsest/evaluation/source_training.py','src/palimpsest/evaluation/balanced_null.py',
        'src/palimpsest/evaluation/features.py','src/palimpsest/evaluation/robust_views.py',
        'experiments/origin_detection/semantic_gaussian/run_iteration.py',
        'experiments/origin_detection/paired_kernel/run_iteration.py',
        'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        'tests/evaluation/test_source_crossfit.py','tests/detection/test_consistent_source_risk.py')]
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in sorted(paths)}


def fit(x,y,w,s,strength,manifest,wrong=None):
    return fit_consistent_source_risk(x,y,w,s,strength=strength,consistency_groups=wrong,
        feature_names=FULL_NAMES,temperature=.1,ridge=.01,scale_floor=.001,
        maximum_iterations=500,gradient_tolerance=1e-5,manifest_sha=manifest)


def source_reference(margins):
    path=CURE/'selection_scores.json';receipt=json.loads((CURE/'iteration.json').read_text())
    if file_sha256(path)!=receipt['selection_scores_sha256']:raise ValueError('CuRe source reference changed')
    old=json.loads(path.read_text())['full/source'];key=lambda r:tuple(r[k] for k in ('domain','src','condition','variant'))
    left,right={key(r):r for r in margins},{key(r):r for r in old}
    if len(left)!=5040 or set(left)!=set(right):raise ValueError('CuRe source reference coverage changed')
    count=sum((left[k]['score']>0)!=(right[k]['score']>0) for k in left)
    if count:raise ValueError('Zero consistency changed CuRe source decisions')
    return {'decisions':5040,'mismatches':count,'maximum_margin_discrepancy':max(abs(left[k]['score']-right[k]['score']) for k in left),
            'scores_sha256':file_sha256(path),'scope':'Same features/loss identity,not independent science'}


def main():
    if (OUTPUT/'iteration.json').exists() or (OUTPUT/'crossfit.json').exists():
        raise FileExistsError('Preserve source cross-fit campaign')
    controls=json.loads((OUTPUT/'software_controls.json').read_text())
    cost=json.loads((OUTPUT/'synthetic_cost.json').read_text())
    if not controls['passed'] or not cost['passed'] or controls['code_pins']!=code_pins() or cost['code_pins']!=code_pins():
        raise ValueError('Cross-fit controls/cost changed')
    rows,parent=cure_inputs();manifest=parent['inventory_sha256']
    records=sorted([r for r in rows if r['role']=='fit' and r['variant'] in VARIANTS[:2]],
                   key=lambda r:tuple(r[k] for k in ('domain','src','condition','variant')))
    x,truth,weights,sources=weighted_source_arrays(rows,FULL_NAMES,VARIANTS[:2],
        expected_source_counts={'rr':540,'chimera':720},seed=CONFIG['seed'])
    if len(x)!=7560 or not np.array_equal(x,[[float(r[n]) for n in FULL_NAMES] for r in records]):
        raise ValueError('Cross-fit traversal differs')
    folds=source_folds(records);fold_id=np.array([folds[r['domain'],r['src']] for r in records])
    assignment,null=balanced_source_null(records,seed=CONFIG['seed'])
    prior=json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if prior['assignment_sha256']!=null['assignment_sha256']:raise ValueError('Cross-fit null assignment changed')
    sham=np.array([assignment[r['domain'],r['src']] for r in records])
    cv,chosen={},{};start=perf_counter()
    with threadpool_limits(limits=1):
        for mode,labels in (('truth',truth),('null',sham)):
            cv[mode]={}
            for strength in STRENGTHS:
                predictions=np.full(len(x),np.nan);diagnostics=[]
                for fold in range(5):
                    train,test=fold_id!=fold,fold_id==fold
                    if sum(test)!=1512 or sum(train)!=6048:raise ValueError('Cross-fit fold size changed')
                    if set(sources[train])&set(sources[test]):raise ValueError('Cross-fit source leakage')
                    _,train_sources=np.unique(sources[train],return_inverse=True)
                    rule,diagnostic=fit(x[train],labels[train],weights[train],train_sources,strength,manifest)
                    predictions[test]=rule.score(x[test]);diagnostics.append(diagnostic)
                if not np.isfinite(predictions).all():raise ValueError('Incomplete OOF predictions')
                scored=[{**{k:r[k] for k in ('domain','scene','condition','variant','src','role')},
                         'label':'FAKE' if y else 'REAL','score':float(score),'fold':int(fold)}
                        for r,y,score,fold in zip(records,labels,predictions,fold_id)]
                rate=crossfit_rates(scored,VARIANTS[:2])
                if len(rate['metrics'])!=30 or len(rate['paired_drops'])!=35:raise ValueError('OOF view/pair count changed')
                cv[mode][str(strength)]={**rate,'fit_diagnostics':diagnostics,'scores':scored}
                print(json.dumps({'cv':mode,'strength':strength,'minimum_ba':rate['minimum_all_scope_ba'],
                                  'maximum_drop':rate['maximum_all_scope_drop']}),flush=True)
            chosen[mode]=choose_strength(cv[mode])
        write_json(OUTPUT/'crossfit.json',{'results':cv,'chosen':chosen,'fold_seed':20261007,
            'fold_sources':{'/'.join(k):v for k,v in folds.items()},'code_pins':code_pins(),'inventory_sha256':manifest,
            'scope':'Fit-only OOF zero-threshold weak views;not independent validation'})
        views={k+'/'+v:view for v in VARIANTS[:2] for p in (False,True)
               for k,view in feature_views(rows,FULL_NAMES,'threshold',processed=p,variant=v).items()}
        if len(views)!=24:raise ValueError('Outer threshold view count differs')
        results,scores,artifacts={}, {}, {};wrong=wrong_sources(rows,sources)
        for key,labels,strength,wrong_group in (('zero',truth,0,None),
            ('selected',truth,float(chosen['truth'][0]),None),
            ('selected_wrong_source',truth,float(chosen['truth'][0]),wrong),
            ('selected_null',sham,float(chosen['null'][0]),None)):
            rule,diagnostic=fit(x,labels,weights,sources,strength,manifest,wrong_group)
            rule,diagnostic['calibration']=class_threshold(rule,views)
            margins,result=score_rule(rows,FULL_NAMES,rule,CONFIG)
            if key=='zero':reference=source_reference(margins)
            result.update({'fit_diagnostics':diagnostic,'threshold':rule.threshold,'chosen_strength':strength})
            f=OUTPUT/(key+'_rule.json');rule.save(f)
            results[key],scores[key],artifacts[key]=result,margins,file_sha256(f)
            print(json.dumps({'candidate':key,'strength':strength,'minimum_ba':result['minimum_domain_ba'],
                             'minimum_scene_ba':result['minimum_scene_ba'],'maximum_drop':result['maximum_any_scene_drop']}),flush=True)
    write_json(OUTPUT/'selection_scores.json',scores)
    write_json(OUTPUT/'iteration.json',{'candidates':results,'chosen_strengths':chosen,'code_pins':code_pins(),
        'zero_reference_gate':reference,'crossfit_sha256':file_sha256(OUTPUT/'crossfit.json'),
        'inventory_sha256':manifest,'balanced_assignment_sha256':null['assignment_sha256'],'elapsed_s':perf_counter()-start,
        'null_raw_processed_auc_ci95':null_intervals(scores['selected_null']),
        'selection_scores_sha256':file_sha256(OUTPUT/'selection_scores.json'),'rule_files':artifacts,
        'goal_achieved':False,'scope':'Conventional source-variance CV;repeated exposed development;no external validation'})


if __name__=='__main__':main()
