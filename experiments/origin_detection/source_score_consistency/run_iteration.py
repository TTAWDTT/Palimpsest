"""Full-feature source risk with fixed consistency strengths and wrong pairs."""

import argparse
from collections import defaultdict
import json
import random
from time import perf_counter
import tomllib

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.readouts.consistent_source_risk import fit_consistent_source_risk
from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES
from palimpsest.evaluation.features import feature_views
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT,WORK_DIR
from experiments.origin_detection.semantic_kernel.run_iteration import inputs
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS,audit_receipt
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals
from experiments.origin_detection.phase_statistics.run_iteration import write_json

CONFIG=REPO_ROOT/'configs/evaluation/source_score_consistency.toml'
OUTPUT=WORK_DIR/'robust_statistics/source_score_consistency'
PARENT=WORK_DIR/'robust_statistics/source_view_risk'


def code_pins():
    files=[CONFIG,REPO_ROOT/'src/palimpsest/detection/algorithms/readouts/consistent_source_risk.py',
           REPO_ROOT/'src/palimpsest/detection/algorithms/readouts/source_view_risk.py',
           REPO_ROOT/'src/palimpsest/evaluation/source_training.py',
           REPO_ROOT/'src/palimpsest/evaluation/robust_views.py',
           REPO_ROOT/'experiments/origin_detection/semantic_kernel/run_iteration.py',
           REPO_ROOT/'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
           REPO_ROOT/'experiments/origin_detection/threshold_calibration/fit_threshold.py',
           REPO_ROOT/'tests/detection/test_consistent_source_risk.py']
    files.extend(REPO_ROOT/'experiments/origin_detection/source_score_consistency'/n for n in ('README.md','run_iteration.py'))
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in files}


def scrambled_pairs(rows,sources,seed):
    records=sorted([r for r in rows if r['role']=='fit' and r['variant'] in VARIANTS[:2]],
                   key=lambda r:tuple(r[k] for k in ('domain','src','condition','variant')))
    if len(records)!=len(sources):raise ValueError('Wrong-pair row alignment changed')
    strata=defaultdict(list)
    for i,r in enumerate(records):strata[tuple(r[k] for k in ('domain','label','condition','variant'))].append(i)
    rng=random.Random(seed);wrong=sources.copy()
    for key in sorted(strata):
        index=strata[key];values=sources[index].tolist();rng.shuffle(values);wrong[index]=values
    if np.array_equal(wrong,sources) or not np.array_equal(np.bincount(wrong),np.bincount(sources)):
        raise ValueError('Wrong-pair negative control did not alter groups')
    return wrong


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    destination=OUTPUT/('pilot.json' if args.pilot else 'iteration.json')
    if destination.exists():raise FileExistsError('Preserve consistency run')
    config=tomllib.loads(CONFIG.read_text());control_path=OUTPUT/'controls.json';control=json.loads(control_path.read_text())
    if control['returncode']!=0 or any(file_sha256(REPO_ROOT/k)!=v for k,v in control['test_pins'].items()):
        raise ValueError('Consistency software gates changed')
    rows,parent=inputs();reference=audit_receipt(PARENT/'iteration.json')
    reference_path=PARENT/'selection_scores.json'
    if file_sha256(reference_path)!=reference['selection_scores_sha256']:raise ValueError('Reference scores changed')
    old=json.loads(reference_path.read_text())['weak/source'];start=perf_counter()
    results={};rules={};scored_all={}
    candidates=[(s,'honest') for s in config['strengths']]+[(.1,'wrong_pairs'),(.1,'source_shuffled_null')]
    if args.pilot:candidates=[(.1,'honest')]
    with threadpool_limits(limits=1):
        for strength,control_kind in candidates:
            key=f'{strength:g}/{control_kind}'
            x,y,weights,sources=weighted_source_arrays(rows,FEATURE_NAMES,VARIANTS[:2],
               expected_source_counts={'rr':540,'chimera':720},seed=config['seed'],shuffled=control_kind=='source_shuffled_null')
            groups=scrambled_pairs(rows,sources,config['seed']) if control_kind=='wrong_pairs' else None
            rule,diagnostic=fit_consistent_source_risk(x,y,weights,sources,strength=strength,consistency_groups=groups,
                feature_names=FEATURE_NAMES,temperature=config['source_temperature'],ridge=config['ridge'],
                scale_floor=config['scale_floor'],maximum_iterations=config['maximum_iterations'],
                gradient_tolerance=config['gradient_tolerance'],manifest_sha=parent['inventory_sha256'])
            if args.pilot:
                write_json(destination,{'elapsed_s':perf_counter()-start,'code_pins':code_pins(),'fit_diagnostics':diagnostic,
                          'scope':'Fit-only costpilot;no threshold/selection scores'})
                print(json.dumps({'pilot_s':perf_counter()-start}),flush=True);return
            views={k+'/'+v:view for v in VARIANTS[:2] for processed in (False,True)
                   for k,view in feature_views(rows,FEATURE_NAMES,'threshold',processed=processed,variant=v).items()}
            if len(views)!=24:raise ValueError('Consistency threshold view count changed')
            rule,diagnostic['calibration']=class_threshold(rule,views)
            scored,result=score_rule(rows,FEATURE_NAMES,rule,config)
            if strength==0 and scored!=old:raise ValueError('Zero consistency changed stage17 reference')
            result.update({'fit_diagnostics':diagnostic,'threshold':rule.threshold,
                          'zero_reference_exact':scored==old if strength==0 else None})
            results[key]=result;rules[key]=rule;scored_all[key]=scored
            print(json.dumps({'candidate':key,'minimum_ba':result['minimum_domain_ba'],
                 'minimum_scene_ba':result['minimum_scene_ba'],'maxscene_drop':result['maximum_any_scene_drop'],
                 'fit_variance':diagnostic['source_score_variance']}),flush=True)
    null='0.1/source_shuffled_null';intervals={}
    for domain in ('rr','chimera'):
        groups=defaultdict(list)
        for r in scored_all[null]:
            if r['domain']==domain and r['variant']=='raw' and r['condition']!='original':groups[r['condition']].append(r)
        for condition,records in groups.items():
            records.sort(key=lambda r:r['src']);labels=np.array([int(r['label']=='FAKE') for r in records])
            intervals[domain+'/'+condition]=auc_intervals({condition:np.array([r['score'] for r in records])},labels,config)[condition]
    refused=all(v[0]>.5 for v in intervals.values())
    candidates=[k for k in results if k.endswith('/honest')]
    chosen=max(candidates,key=lambda k:(min(results[k]['minimum_domain_ba'],results[k]['minimum_scene_ba']),
                   -results[k]['maximum_any_scene_drop'],-float(k.split('/')[0]))) if not refused else None
    for k,rule in rules.items():rule.save(OUTPUT/(k.replace('/','_')+'_rule.json'))
    write_json(OUTPUT/'selection_scores.json',scored_all)
    write_json(destination,{'candidates':results,'development_chosen':chosen,'interpretation_refused':refused,
       'null_raw_processed_auc_ci95':intervals,'elapsed_s':perf_counter()-start,'code_pins':code_pins(),
       'inventory_sha256':parent['inventory_sha256'],'reference_iteration_sha256':file_sha256(PARENT/'iteration.json'),
       'controls_sha256':file_sha256(control_path),'selection_scores_sha256':file_sha256(OUTPUT/'selection_scores.json'),
       'rule_files':{k:file_sha256(OUTPUT/(k.replace('/','_')+'_rule.json')) for k in rules},
       'goal_achieved':False,'scope':'Frozen neuralCLIP plus conventional source-risk/consistency head;exposed development'})


if __name__=='__main__':main()
