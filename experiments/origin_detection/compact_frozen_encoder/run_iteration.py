"""One pinned small encoder; fixed descriptor x convex-head comparison."""

import argparse
from collections import defaultdict
import json
import random
from time import perf_counter
import tomllib

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.source_view_risk import fit_source_risk
from palimpsest.detection.algorithms.residual_statistics.features import resize256
from palimpsest.detection.representations.frozen_dinov2_small import FEATURE_NAMES,CLS_NAMES,FrozenDinoV2Small
from palimpsest.evaluation.features import feature_views
from palimpsest.evaluation.pixel_features import audit_variants,extract_inventory
from palimpsest.evaluation.robust_views import evaluate_views
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT,WORK_DIR,MODELS_ROOT
from experiments.origin_detection.source_view_risk.run_iteration import parent_data,Q60,VARIANTS
from experiments.origin_detection.phase_statistics.run_iteration import image_path,write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals

CONFIG=REPO_ROOT/'configs/evaluation/compact_frozen_encoder.toml'
OUTPUT=WORK_DIR/'robust_statistics/compact_frozen_encoder'
SOURCE_RECEIPT=WORK_DIR/'robust_statistics/compact_encoder_review/runtime_receipt.json'
MODES={'cls':CLS_NAMES,'cls_patch':FEATURE_NAMES}


def code_pins():
    files=[CONFIG,REPO_ROOT/'src/palimpsest/detection/representations/frozen_dinov2_small.py',
           REPO_ROOT/'src/palimpsest/detection/algorithms/source_view_risk.py',
           REPO_ROOT/'src/palimpsest/detection/algorithms/paired_stability.py',
           REPO_ROOT/'src/palimpsest/evaluation/source_training.py',
           REPO_ROOT/'src/palimpsest/evaluation/robust_views.py',
           REPO_ROOT/'src/palimpsest/evaluation/pixel_features.py',
           REPO_ROOT/'src/palimpsest/evaluation/features.py',
           REPO_ROOT/'src/palimpsest/evaluation/pairing.py',
           REPO_ROOT/'src/palimpsest/evaluation/classification.py',
           REPO_ROOT/'experiments/origin_detection/source_view_risk/run_iteration.py',
           REPO_ROOT/'experiments/origin_detection/threshold_calibration/fit_threshold.py',
           REPO_ROOT/'experiments/origin_detection/robust_statistics/rr/evaluate_features.py',
           REPO_ROOT/'tests/detection/test_frozen_dinov2_small.py',
           REPO_ROOT/'tests/evaluation/test_source_training.py']
    directory=REPO_ROOT/'experiments/origin_detection/compact_frozen_encoder'
    files.extend(directory/name for name in ('run_iteration.py','audit_runtime.py','README.md'))
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in sorted(files)}


def score_rule(rows,names,rule,config):
    selected=[r for r in rows if r['role']=='selection']
    matrix=np.array([[float(r[n]) for n in names] for r in selected])
    scores=rule.score(matrix)
    if not np.array_equal(scores,[rule.score([v])[0] for v in matrix]):raise ValueError('Single/batch scores differ')
    records=[{**{k:r[k] for k in ('domain','scene','condition','variant','src','role','label')},
              'score':float(score-rule.threshold)} for r,score in zip(selected,scores)]
    result=evaluate_views(records,variant_order=VARIANTS,minimum_ba=config['provisional_final_ba'],
                          maximum_drop=config['provisional_final_drop'])
    return records,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--pilot',type=int,default=0)
    group.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.pilot<0:parser.error('Pilot must be positive')
    config=tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    runtime_path=OUTPUT/'runtime_controls.json';runtime=json.loads(runtime_path.read_text())
    controls_path=OUTPUT/'controls.json';controls=json.loads(controls_path.read_text())
    if (not runtime['passed'] or controls['returncode']!=0
            or runtime['source_receipt_sha256']!=file_sha256(SOURCE_RECEIPT)
            or any(file_sha256(REPO_ROOT/k)!=v for k,v in controls['test_pins'].items())
            or runtime['script_sha256']!=file_sha256(REPO_ROOT/'experiments/origin_detection/compact_frozen_encoder/audit_runtime.py')):
        raise ValueError('Small encoder runtime/software controls failed or changed')
    old,inventory,parent,previous=parent_data();del old,previous
    cv2.setNumThreads(1)
    if args.prepare or args.pilot:
        directory=OUTPUT if args.prepare else OUTPUT.with_name('compact_frozen_encoder_pilot')
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/'features.json').exists():raise FileExistsError('Preserve small encoder extraction')
        encoder=FrozenDinoV2Small(MODELS_ROOT/'dinov2_small/dinov2_vits14_pretrain.pth',SOURCE_RECEIPT,device=config['device'])
        if args.pilot:
            random.Random(config['seed']).shuffle(inventory);inventory=inventory[:args.pilot]
        result=extract_inventory(inventory,directory/'features.csv',names=FEATURE_NAMES,extractor=encoder.extract,
          resolve_path=image_path,resize=resize256,ordinary_variants=VARIANTS[:3],selection_variant=Q60,
          jpeg_parameters={VARIANTS[1]:(90,0),VARIANTS[2]:(70,2),Q60:(60,2)})
        write_json(directory/'features.json',{**result,'code_pins':code_pins(),'encoder':encoder.provenance,
                   'inventory_sha256':parent['inventory_sha256'],'controls_sha256':file_sha256(controls_path),
                   'runtime_controls_sha256':file_sha256(runtime_path),
                   'parent_features_receipt_sha256':file_sha256(WORK_DIR/'robust_statistics/frozen_clip/features.json')})
        return
    if (OUTPUT/'iteration.json').exists():raise FileExistsError('Preserve small encoder evaluation')
    receipt=json.loads((OUTPUT/'features.json').read_text())
    if (receipt['code_pins']!=code_pins() or receipt['csv_sha256']!=file_sha256(OUTPUT/'features.csv')
            or receipt['inventory_sha256']!=parent['inventory_sha256']):raise ValueError('Small cache/code/inventory changed')
    rows=read_rows(OUTPUT/'features.csv')
    audit_variants(rows,inventory,FEATURE_NAMES,ordinary_variants=VARIANTS[:3],selection_variant=Q60)
    if len(rows)!=20160:raise ValueError('Small feature count differs')
    start=perf_counter();rules={};results={};scored_results={}
    candidates=[(mode,objective,False) for mode in config['representations'] for objective in config['objectives']]
    candidates.append(('cls_patch','source',True))
    with threadpool_limits(limits=1):
        for mode,objective,shuffled in candidates:
            key=mode+'/'+objective+('_source_shuffled_null' if shuffled else '')
            names=MODES[mode]
            x,y,weights,sources=weighted_source_arrays(rows,names,VARIANTS[:3],
                 expected_source_counts={'rr':540,'chimera':720},seed=config['seed'],shuffled=shuffled)
            rule,diagnostic=fit_source_risk(x,y,weights,sources,feature_names=names,
                 temperature=config['source_temperature'] if objective=='source' else 0,ridge=config['ridge'],
                 scale_floor=config['scale_floor'],maximum_iterations=config['maximum_iterations'],
                 gradient_tolerance=config['gradient_tolerance'],manifest_sha=parent['inventory_sha256'])
            views={key+'/'+v:view for v in VARIANTS[:2] for processed in (False,True)
                   for key,view in feature_views(rows,names,'threshold',processed=processed,variant=v).items()}
            if len(views)!=24:raise ValueError('Threshold views changed')
            rule,diagnostic['calibration']=class_threshold(rule,views)
            scored,result=score_rule(rows,names,rule,config)
            result.update({'fit_diagnostics':diagnostic,'threshold':rule.threshold})
            rules[key]=rule;results[key]=result;scored_results[key]=scored
            print(json.dumps({'candidate':key,'minimum_ba':result['minimum_domain_ba'],
                    'minimum_scene_ba':result['minimum_scene_ba'],'maximum_scene_drop':result['maximum_any_scene_drop']}),flush=True)
    null='cls_patch/source_source_shuffled_null';intervals={}
    for domain in ('rr','chimera'):
        groups=defaultdict(list)
        for r in scored_results[null]:
            if r['domain']==domain and r['variant']=='raw' and r['condition']!='original':groups[r['condition']].append(r)
        for condition,records in groups.items():
            records.sort(key=lambda r:r['src']);labels=np.array([int(r['label']=='FAKE') for r in records])
            intervals[domain+'/'+condition]=auc_intervals({condition:np.array([r['score'] for r in records])},labels,config)[condition]
    refused=all(v[0]>.5 for v in intervals.values())
    eligible=[k for k in results if k!=null]
    chosen=max(eligible,key=lambda k:(min(results[k]['minimum_domain_ba'],results[k]['minimum_scene_ba']),
                -results[k]['maximum_any_scene_drop'],k.startswith('cls/'),k.endswith('/mean'))) if not refused else None
    for key,rule in rules.items():rule.save(OUTPUT/(key.replace('/','_')+'_rule.json'))
    write_json(OUTPUT/'selection_scores.json',scored_results)
    write_json(OUTPUT/'iteration.json',{'candidates':results,'development_chosen':chosen,'interpretation_refused':refused,
       'null_raw_processed_auc_ci95':intervals,'elapsed_s':perf_counter()-start,'code_pins':code_pins(),
       'single_batch_scores_exact':True,'inventory_sha256':parent['inventory_sha256'],
       'features_receipt_sha256':file_sha256(OUTPUT/'features.json'),'controls_sha256':file_sha256(controls_path),
       'runtime_controls_sha256':file_sha256(runtime_path),'selection_scores_sha256':file_sha256(OUTPUT/'selection_scores.json'),
       'rule_files':{k:file_sha256(OUTPUT/(k.replace('/','_')+'_rule.json')) for k in rules},
       'goal_achieved':False,'scope':'Frozen pretrained neural representation+fitted conventional head;exposed development only'})


if __name__=='__main__':main()
