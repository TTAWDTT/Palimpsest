"""Six existing convex objectives on the signed full CuRe representation."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.group_margin import fit_group_margin
from palimpsest.detection.representations.frozen_cure import FULL_NAMES
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.features import feature_views
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.semantic_gaussian.run_iteration import cure_inputs, null_intervals
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS, audit_receipt
from experiments.origin_detection.paired_stability.fit_rules import paired_deltas
from experiments.origin_detection.semantic_group_risk.run_iteration import wrong_weak_pairs
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.cure_readout.prepare_features import OUTPUT as CURE

OUTPUT=WORK_DIR/'robust_statistics/cure_group_risk'
CONFIG={'seed':20261006,'bootstrap_repetitions':2000,'provisional_final_ba':.8,'provisional_final_drop':.02}


def code_pins():
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md']
    paths+=[REPO_ROOT/p for p in ('src/palimpsest/detection/algorithms/group_margin.py',
        'src/palimpsest/detection/algorithms/paired_stability.py',
        'src/palimpsest/evaluation/source_training.py','src/palimpsest/evaluation/balanced_null.py',
        'src/palimpsest/evaluation/features.py','src/palimpsest/evaluation/robust_views.py',
        'experiments/origin_detection/semantic_gaussian/run_iteration.py',
        'experiments/origin_detection/paired_stability/fit_rules.py',
        'experiments/origin_detection/semantic_group_risk/run_iteration.py',
        'experiments/origin_detection/group_margin/run_iteration.py',
        'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        'tests/detection/test_group_margin.py','tests/evaluation/test_semantic_group_protocol.py')]
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in sorted(paths)}


def mean_reference(margins):
    receipt=audit_receipt(CURE/'iteration.json');path=CURE/'selection_scores.json'
    if file_sha256(path)!=receipt['selection_scores_sha256']:raise ValueError('CuRe mean reference changed')
    old=json.loads(path.read_text())['full/mean']
    key=lambda r:tuple(r[k] for k in ('domain','src','condition','variant'))
    left,right={key(r):r for r in margins},{key(r):r for r in old}
    if len(left)!=5040 or set(left)!=set(right):raise ValueError('CuRe mean reference coverage differs')
    count=sum((left[k]['score']>0)!=(right[k]['score']>0) for k in left)
    if count:raise ValueError('Equivalent CuRe mean solver changed decisions')
    return {'decisions':5040,'mismatches':count,'maximum_margin_discrepancy':max(abs(left[k]['score']-right[k]['score']) for k in left),
            'reference_sha256':file_sha256(path),'scope':'Same features/loss;arithmetic control,not independent science'}


def main():
    if (OUTPUT/'iteration.json').exists():raise FileExistsError('Preserve CuRe group campaign')
    controls=json.loads((OUTPUT/'software_controls.json').read_text())
    if not controls['passed'] or controls['code_pins']!=code_pins():raise ValueError('CuRe group controls changed')
    rows,parent=cure_inputs()
    assignment,null=balanced_source_null([r for r in rows if r['role']=='fit'],seed=CONFIG['seed'])
    prior=json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if prior['assignment_sha256']!=null['assignment_sha256']:raise ValueError('CuRe group assignment changed')
    views={k+'/'+v:view for v in VARIANTS[:2] for p in (False,True)
           for k,view in feature_views(rows,FULL_NAMES,'threshold',processed=p,variant=v).items()}
    if len(views)!=24:raise ValueError('CuRe group threshold coverage changed')
    results,scores,artifacts={}, {}, {};start=perf_counter()
    with threadpool_limits(limits=1):
        for mode in ('mean','group','pair','both','wrong_pair_both','balanced_null_both'):
            selected=[{**r,'label':'FAKE' if assignment[r['domain'],r['src']] else 'REAL'}
                      if r['role']=='fit' else r for r in rows] if mode=='balanced_null_both' else rows
            records=sorted([r for r in selected if r['role']=='fit' and r['variant'] in VARIANTS[:2]],
                           key=lambda r:tuple(r[k] for k in ('domain','src','condition','variant')))
            x,y,w,_=weighted_source_arrays(selected,FULL_NAMES,VARIANTS[:2],
                expected_source_counts={'rr':540,'chimera':720},seed=CONFIG['seed'])
            if not np.array_equal(x,[[float(r[n]) for n in FULL_NAMES] for r in records]) or not np.array_equal(y,[int(r['label']=='FAKE') for r in records]):
                raise ValueError('CuRe fit traversal and group metadata differ')
            keys=[tuple(r[k] for k in ('domain','scene','condition','variant','label')) for r in records]
            unique=sorted(set(keys));mapping={k:i for i,k in enumerate(unique)}
            if len(unique)!=48 or len(x)!=7560:raise ValueError('CuRe48groups7560fit coverage differs')
            groups=np.array([mapping[k] for k in keys])
            if mode=='wrong_pair_both':d,pw,pair_control=wrong_weak_pairs(selected,FULL_NAMES,CONFIG['seed'])
            else:
                d,pw=paired_deltas(selected,'fit',VARIANTS[:2],FULL_NAMES)
                pair_control={'pairs':len(d),'same_source_pairs':len(d)}
            if len(d)!=6300:raise ValueError('CuRe6300pair coverage differs')
            temperature=.1 if mode in ('group','both','wrong_pair_both','balanced_null_both') else 0.
            strength=.1 if mode in ('pair','both','wrong_pair_both','balanced_null_both') else 0.
            rule,diagnostic=fit_group_margin(x,y,w,groups,d,pw,feature_names=FULL_NAMES,ridge=.01,
                temperature=temperature,strength=strength,manifest_sha=parent['inventory_sha256'])
            rule,diagnostic['calibration']=class_threshold(rule,views)
            margins,result=score_rule(rows,FULL_NAMES,rule,CONFIG)
            if mode=='mean':reference=mean_reference(margins)
            diagnostic.update({'group_keys':unique,'pair_control':pair_control})
            result.update({'threshold':rule.threshold,'fit_diagnostics':diagnostic})
            f=OUTPUT/(mode+'_rule.json');rule.save(f)
            results[mode],scores[mode],artifacts[mode]=result,margins,file_sha256(f)
            print(json.dumps({'candidate':mode,'minimum_ba':result['minimum_domain_ba'],
                             'minimum_scene_ba':result['minimum_scene_ba'],'maximum_drop':result['maximum_any_scene_drop']}),flush=True)
    write_json(OUTPUT/'selection_scores.json',scores)
    write_json(OUTPUT/'iteration.json',{'candidates':results,'code_pins':code_pins(),'mean_reference_gate':reference,
        'inventory_sha256':parent['inventory_sha256'],'balanced_assignment_sha256':null['assignment_sha256'],
        'null_raw_processed_auc_ci95':null_intervals(scores['balanced_null_both']),'elapsed_s':perf_counter()-start,
        'selection_scores_sha256':file_sha256(OUTPUT/'selection_scores.json'),'rule_files':artifacts,
        'goal_achieved':False,'scope':'Established objectives on frozen CuRe;exposed development with similarity flag'})


if __name__=='__main__':main()
