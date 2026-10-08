"""Five fixed posterior-consistency controls on immutable CuRe vectors."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.probability_consistency import fit_probability_consistency
from palimpsest.detection.algorithms.paired_stability import StableRule
from palimpsest.detection.representations.frozen_cure import FULL_NAMES
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.features import feature_views
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT,WORK_DIR
from experiments.origin_detection.semantic_gaussian.run_iteration import cure_inputs,null_intervals
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS
from experiments.origin_detection.cure_source_crossfit.run_iteration import source_reference
from experiments.origin_detection.paired_kernel.run_iteration import wrong_sources
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.phase_statistics.run_iteration import write_json

OUTPUT=WORK_DIR/'robust_statistics/cure_probability_consistency'
CONFIG={'seed':20261006,'bootstrap_repetitions':2000,'provisional_final_ba':.8,'provisional_final_drop':.02}


def code_pins():
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md']
    paths+=[REPO_ROOT/p for p in ('src/palimpsest/detection/algorithms/probability_consistency.py',
        'src/palimpsest/detection/algorithms/source_view_risk.py','src/palimpsest/detection/algorithms/paired_stability.py',
        'src/palimpsest/evaluation/source_training.py','src/palimpsest/evaluation/balanced_null.py',
        'src/palimpsest/evaluation/features.py','src/palimpsest/evaluation/robust_views.py',
        'experiments/origin_detection/semantic_gaussian/run_iteration.py',
        'experiments/origin_detection/cure_source_crossfit/run_iteration.py',
        'experiments/origin_detection/paired_kernel/run_iteration.py',
        'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        'tests/detection/test_probability_consistency.py')]
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in sorted(paths)}


def main():
    if (OUTPUT/'iteration.json').exists():raise FileExistsError('Preserve posterior consistency campaign')
    controls=json.loads((OUTPUT/'software_controls.json').read_text());cost=json.loads((OUTPUT/'synthetic_cost.json').read_text())
    if not controls['passed'] or not cost['passed'] or controls['code_pins']!=code_pins() or cost['code_pins']!=code_pins():
        raise ValueError('Posterior controls or cost changed')
    rows,parent=cure_inputs();manifest=parent['inventory_sha256']
    records=sorted([r for r in rows if r['role']=='fit' and r['variant'] in VARIANTS[:2]],
                   key=lambda r:tuple(r[k] for k in ('domain','src','condition','variant')))
    x,truth,w,s=weighted_source_arrays(rows,FULL_NAMES,VARIANTS[:2],
        expected_source_counts={'rr':540,'chimera':720},seed=CONFIG['seed'])
    if len(x)!=7560 or not np.array_equal(x,[[float(r[n]) for n in FULL_NAMES] for r in records]):
        raise ValueError('Posterior fit coverage/traversal differs')
    assignment,null=balanced_source_null(records,seed=CONFIG['seed'])
    prior=json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if prior['assignment_sha256']!=null['assignment_sha256']:raise ValueError('Posterior balanced assignment differs')
    sham=np.array([assignment[r['domain'],r['src']] for r in records]);wrong=wrong_sources(rows,s)
    views={k+'/'+v:view for v in VARIANTS[:2] for p in (False,True)
           for k,view in feature_views(rows,FULL_NAMES,'threshold',processed=p,variant=v).items()}
    if len(views)!=24:raise ValueError('Posterior threshold coverage differs')
    results,scores,artifacts={}, {}, {};start=perf_counter()
    with threadpool_limits(limits=1):
        for key,strength,labels,grouping in (('zero',0,truth,None),('js1',1,truth,None),
            ('js12',12,truth,None),('js12_wrong_source',12,truth,wrong),('js12_source_null',12,sham,None)):
            rule,diagnostic=fit_probability_consistency(x,labels,w,s,strength=strength,consistency_groups=grouping,
                feature_names=FULL_NAMES,temperature=.1,ridge=.01,scale_floor=.001,
                maximum_iterations=500,gradient_tolerance=1e-5,manifest_sha=manifest)
            rule,diagnostic['calibration']=class_threshold(rule,views)
            f=OUTPUT/(key+'_rule.json');rule.save(f)
            if not np.array_equal(rule.score(x[:12]),StableRule.load(f).score(x[:12])):
                raise ValueError('Posterior rule load changed scores')
            margins,result=score_rule(rows,FULL_NAMES,rule,CONFIG)
            if key=='zero':reference=source_reference(margins)
            result.update({'fit_diagnostics':diagnostic,'threshold':rule.threshold})
            results[key],scores[key],artifacts[key]=result,margins,file_sha256(f)
            print(json.dumps({'candidate':key,'minimum_ba':result['minimum_domain_ba'],
                             'minimum_scene_ba':result['minimum_scene_ba'],'maximum_drop':result['maximum_any_scene_drop'],
                             'fit_js':diagnostic['probability_js']}),flush=True)
    write_json(OUTPUT/'selection_scores.json',scores)
    write_json(OUTPUT/'iteration.json',{'candidates':results,'code_pins':code_pins(),'zero_reference_gate':reference,
        'inventory_sha256':manifest,'balanced_assignment_sha256':null['assignment_sha256'],
        'null_raw_processed_auc_ci95':null_intervals(scores['js12_source_null']),'elapsed_s':perf_counter()-start,
        'selection_scores_sha256':file_sha256(OUTPUT/'selection_scores.json'),'rule_files':artifacts,
        'goal_achieved':False,'scope':'Known Bernoulli JS consistency;two stationary starts;exposed development,not independent verification'})


if __name__=='__main__':main()
