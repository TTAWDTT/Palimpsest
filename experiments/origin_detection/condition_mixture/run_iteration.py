"""Fixed feature-only condition gate and source-risk origin expert comparison."""

from dataclasses import replace
import json
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.source_view_risk import fit_source_risk
from palimpsest.detection.algorithms.condition_mixture import ConditionMixtureRule
from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES as CLIP_NAMES
from palimpsest.detection.representations.frozen_dinov2_small import FEATURE_NAMES as DINO_NAMES
from palimpsest.evaluation.signed_feature_inputs import load_signed_columns
from palimpsest.evaluation.pixel_features import join_features
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.evaluation.features import feature_views
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT,WORK_DIR
from experiments.origin_detection.source_view_risk.run_iteration import parent_data,VARIANTS,Q60
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals

OUTPUT = WORK_DIR/'robust_statistics/condition_mixture'
CONFIG = {'seed':20261006,'bootstrap_repetitions':2000,'provisional_final_ba':.8,'provisional_final_drop':.02}
FEATURE_NAMES = CLIP_NAMES+DINO_NAMES


def code_pins():
    paths = [REPO_ROOT/'src/palimpsest/detection/algorithms'/n for n in
             ('condition_mixture.py','source_view_risk.py','paired_stability.py')]
    paths += [REPO_ROOT/'src/palimpsest/evaluation'/n for n in
              ('signed_feature_inputs.py','source_training.py','pixel_features.py','robust_views.py')]
    paths += [REPO_ROOT/'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        REPO_ROOT/'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        REPO_ROOT/'tests/detection/test_condition_mixture.py']
    paths += [REPO_ROOT/'experiments/origin_detection/condition_mixture'/n for n in ('README.md','run_iteration.py')]
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in paths}


def fit_head(x,y,w,s,parent,temperature):
    return fit_source_risk(x,y,w,s,feature_names=FEATURE_NAMES,temperature=temperature,
        ridge=.01,scale_floor=.001,maximum_iterations=500,gradient_tolerance=1e-5,
        manifest_sha=parent['inventory_sha256'])


def main():
    destination = OUTPUT/'iteration.json'
    if destination.exists():
        raise FileExistsError('Preserve condition-mixture campaign')
    controls = json.loads((OUTPUT/'software_controls.json').read_text())
    if controls['returncode']!=0 or any(file_sha256(REPO_ROOT/k)!=v for k,v in controls['test_pins'].items()):
        raise ValueError('Mixture known-input controls changed')
    old,inventory,parent,_ = parent_data();del old
    parts,receipts = [],{}
    for slug,names in [('intermediate_encoder',CLIP_NAMES),('compact_frozen_encoder',DINO_NAMES)]:
        part,sha = load_signed_columns(WORK_DIR/'robust_statistics'/slug,names,inventory,
            repository_root=REPO_ROOT,inventory_sha=parent['inventory_sha256'],
            ordinary_variants=VARIANTS[:3],selection_variant=Q60)
        parts.append(part);receipts[slug]=sha
    rows = join_features(*parts,DINO_NAMES);del parts
    ordered = sorted([r for r in rows if r['role']=='fit' and r['variant'] in VARIANTS[:2]],
        key=lambda r:tuple(r[k] for k in ('domain','src','condition','variant')))
    processed = np.array([r['condition']!='original' for r in ordered])
    x,y,w,s = weighted_source_arrays(rows,FEATURE_NAMES,VARIANTS[:2],
        expected_source_counts={'rr':540,'chimera':720},seed=CONFIG['seed'])
    if len(x)!=7560 or len(ordered)!=len(x):
        raise ValueError('Expert mask coverage changed')
    for source in range(1260):
        if np.sum((s==source)&~processed)!=2 or np.sum((s==source)&processed)!=4:
            raise ValueError('Expert source views incomplete')
    gate_weights = w*np.where(processed,.75,1.5)
    if not np.allclose(np.bincount(processed.astype(int),weights=gate_weights),[.5,.5],rtol=0,atol=1e-12):
        raise ValueError('Gate balanced condition mass changed')
    start = perf_counter()
    diagnostics,rules,results,scores = {},{},{},{}
    with threadpool_limits(limits=1):
        gate,diagnostics['gate'] = fit_head(x,processed.astype(int),gate_weights,np.arange(len(x)),parent,0)
        for shuffled in (False,True):
            labels = y
            if shuffled:
                null_x,labels,null_w,null_s = weighted_source_arrays(rows,FEATURE_NAMES,VARIANTS[:2],
                    expected_source_counts={'rr':540,'chimera':720},seed=CONFIG['seed'],shuffled=True)
                if not np.array_equal(x,null_x) or not np.array_equal(w,null_w) or not np.array_equal(s,null_s):
                    raise ValueError('Null changed source features or weights')
                del null_x,null_w,null_s
            original,d0 = fit_head(x[~processed],labels[~processed],w[~processed],s[~processed],parent,.1)
            altered,d1 = fit_head(x[processed],labels[processed],w[processed],s[processed],parent,.1)
            diagnostics['null_experts' if shuffled else 'experts'] = {'original':d0,'processed':d1}
            mixture = ConditionMixtureRule(original,altered,gate)
            variants = [('soft_gate_null',mixture)] if shuffled else [
                ('soft_gate',mixture),('uniform_half',replace(mixture,uniform=True))]
            for key,rule in variants:
                views = {k+'/'+v:view for v in VARIANTS[:2] for p in (False,True)
                    for k,view in feature_views(rows,FEATURE_NAMES,'threshold',processed=p,variant=v).items()}
                if len(views)!=24:
                    raise ValueError('Mixture threshold views changed')
                rule,calibration = class_threshold(rule,views)
                margins,result = score_rule(rows,FEATURE_NAMES,rule,CONFIG)
                result.update({'threshold':rule.threshold,'calibration':calibration})
                rules[key],scores[key],results[key] = rule,margins,result
                print(json.dumps({'candidate':key,'minimum_ba':result['minimum_domain_ba'],
                    'minimum_scene_ba':result['minimum_scene_ba'],
                    'maximum_scene_drop':result['maximum_any_scene_drop']}),flush=True)
    selected = [r for r in rows if r['role']=='selection']
    values = np.array([[float(r[n]) for n in FEATURE_NAMES] for r in selected])
    gate_scores = gate.score(values)
    gate_controls = {}
    for domain in ('rr','chimera'):
        indices = [i for i,r in enumerate(selected) if r['domain']==domain and r['variant']=='raw']
        labels = np.array([selected[i]['condition']!='original' for i in indices],int)
        raw_scores = gate_scores[indices]
        gate_controls[domain] = {'images':len(indices),
            'original_accuracy':float(np.mean(raw_scores[labels==0]<=0)),
            'processed_accuracy':float(np.mean(raw_scores[labels==1]>0)),
            'scope':'Development processing cue only,not origin accuracy or wild posterior calibration'}
    null_intervals = {}
    for domain in ('rr','chimera'):
        for condition in sorted({r['condition'] for r in scores['soft_gate_null'] if r['domain']==domain and r['condition']!='original'}):
            records = sorted([r for r in scores['soft_gate_null'] if r['domain']==domain
                and r['condition']==condition and r['variant']=='raw'],key=lambda r:r['src'])
            name = domain+'/'+condition
            null_intervals[name] = auc_intervals({name:np.array([r['score'] for r in records])},
                np.array([int(r['label']=='FAKE') for r in records]),CONFIG)[name]
    refused = all(v[0]>.5 for v in null_intervals.values())
    chosen = None if refused else max(('soft_gate','uniform_half'),key=lambda k:(min(
        results[k]['minimum_domain_ba'],results[k]['minimum_scene_ba']),-results[k]['maximum_any_scene_drop']))
    for key,rule in rules.items():
        rule.save(OUTPUT/(key+'_rule.json'))
    write_json(OUTPUT/'selection_scores.json',scores)
    write_json(destination,{'candidates':results,'development_chosen':chosen,'interpretation_refused':refused,
        'null_raw_processed_auc_ci95':null_intervals,'gate_controls':gate_controls,'fit_diagnostics':diagnostics,
        'elapsed_s':perf_counter()-start,'parent_receipts':receipts,'code_pins':code_pins(),
        'inventory_sha256':parent['inventory_sha256'],'selection_scores_sha256':file_sha256(OUTPUT/'selection_scores.json'),
        'rule_files':{k:file_sha256(OUTPUT/(k+'_rule.json')) for k in rules},'goal_achieved':False,
        'scope':'Feature-only condition gate,conventional origin experts;exposed development,not invariant theorem'})


if __name__ == '__main__':
    main()
