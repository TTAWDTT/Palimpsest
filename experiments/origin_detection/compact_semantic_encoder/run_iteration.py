"""One smaller semantic encoder, weak augmentation and fixed conventional heads."""

import argparse
from collections import defaultdict
import json
import random
from time import perf_counter
import tomllib

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.readouts.source_view_risk import fit_source_risk
from palimpsest.detection.representations.frozen_clip_base import FrozenClipBase,FEATURE_NAMES
from palimpsest.detection.algorithms.residual_statistics.features import resize256
from palimpsest.evaluation.features import feature_views
from palimpsest.evaluation.pixel_features import extract_inventory,audit_variants
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT,WORK_DIR,MODELS_ROOT
from experiments.origin_detection.source_view_risk.run_iteration import parent_data,VARIANTS,Q60
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import image_path,write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals

CONFIG=REPO_ROOT/'configs/evaluation/compact_semantic_encoder.toml'
OUTPUT=WORK_DIR/'robust_statistics/compact_semantic_encoder'


def code_pins():
    files=[CONFIG,REPO_ROOT/'src/palimpsest/detection/representations/frozen_clip_base.py',
           REPO_ROOT/'src/palimpsest/detection/representations/frozen_clip.py',
           REPO_ROOT/'src/palimpsest/detection/algorithms/readouts/source_view_risk.py',
           REPO_ROOT/'src/palimpsest/evaluation/source_training.py',
           REPO_ROOT/'src/palimpsest/evaluation/pixel_features.py',
           REPO_ROOT/'src/palimpsest/evaluation/robust_views.py',
           REPO_ROOT/'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
           REPO_ROOT/'tests/detection/test_frozen_clip_base.py',
           REPO_ROOT/'tests/detection/test_frozen_clip.py']
    files.extend(REPO_ROOT/'experiments/origin_detection/compact_semantic_encoder'/n for n in ('README.md','run_iteration.py','audit_runtime.py'))
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in files}


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group()
    group.add_argument('--pilot',type=int,default=0);group.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.pilot<0:parser.error('Pilot must be positive')
    config=tomllib.loads(CONFIG.read_text());control_path=OUTPUT/'controls.json';controls=json.loads(control_path.read_text())
    runtime_path=OUTPUT/'runtime_controls.json';runtime=json.loads(runtime_path.read_text())
    if (controls['returncode']!=0 or not runtime['passed']
        or any(file_sha256(REPO_ROOT/k)!=v for k,v in controls['test_pins'].items())
        or runtime['script_sha256']!=file_sha256(REPO_ROOT/'experiments/origin_detection/compact_semantic_encoder/audit_runtime.py')
        or runtime['representation_sha256']!=file_sha256(REPO_ROOT/'src/palimpsest/detection/representations/frozen_clip_base.py')
        or runtime['source_transform_sha256']!=file_sha256(WORK_DIR/'robust_statistics/compact_semantic_review/clip_clip.py')):
        raise ValueError('Base runtime/software gate changed')
    _,inventory,parent,_=parent_data();cv2.setNumThreads(1)
    if args.pilot or args.prepare:
        directory=OUTPUT.with_name('compact_semantic_encoder_pilot') if args.pilot else OUTPUT
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/'features.json').exists():raise FileExistsError('Preserve base feature receipt')
        if args.pilot:random.Random(config['seed']).shuffle(inventory);inventory=inventory[:args.pilot]
        encoder=FrozenClipBase(MODELS_ROOT/'clip_small/ViT-B-16.pt',device=config['device'])
        result=extract_inventory(inventory,directory/'features.csv',names=FEATURE_NAMES,extractor=encoder.extract,
                 resolve_path=image_path,resize=resize256,ordinary_variants=VARIANTS[:3],selection_variant=Q60,
                 jpeg_parameters={VARIANTS[1]:(90,0),VARIANTS[2]:(70,2),Q60:(60,2)})
        write_json(directory/'features.json',{**result,'code_pins':code_pins(),'encoder':encoder.provenance,
                    'inventory_sha256':parent['inventory_sha256'],'runtime_sha256':file_sha256(runtime_path),
                    'controls_sha256':file_sha256(control_path)})
        return
    destination=OUTPUT/'iteration.json'
    if destination.exists():raise FileExistsError('Preserve base fitting receipt')
    receipt=json.loads((OUTPUT/'features.json').read_text())
    if receipt['csv_sha256']!=file_sha256(OUTPUT/'features.csv') or receipt['inventory_sha256']!=parent['inventory_sha256']:
        raise ValueError('Base cache changed')
    for relative,sha in receipt['code_pins'].items():
        if file_sha256(REPO_ROOT/relative)!=sha:raise ValueError('Base extraction code changed')
    rows=read_rows(OUTPUT/'features.csv')
    audit_variants(rows,inventory,FEATURE_NAMES,ordinary_variants=VARIANTS[:3],selection_variant=Q60)
    start=perf_counter();results={};rules={};all_scores={}
    candidates=[(o,False) for o in config['objectives']]+[('source',True)]
    with threadpool_limits(limits=1):
        for objective,shuffled in candidates:
            key=objective+('_source_shuffled_null' if shuffled else '')
            x,y,w,s=weighted_source_arrays(rows,FEATURE_NAMES,VARIANTS[:2],
                  expected_source_counts={'rr':540,'chimera':720},seed=config['seed'],shuffled=shuffled)
            rule,diagnostic=fit_source_risk(x,y,w,s,feature_names=FEATURE_NAMES,
                  temperature=config['source_temperature'] if objective=='source' else 0,ridge=config['ridge'],
                  scale_floor=config['scale_floor'],maximum_iterations=config['maximum_iterations'],
                  gradient_tolerance=config['gradient_tolerance'],manifest_sha=parent['inventory_sha256'])
            views={k+'/'+v:view for v in VARIANTS[:2] for processed in (False,True)
                   for k,view in feature_views(rows,FEATURE_NAMES,'threshold',processed=processed,variant=v).items()}
            if len(views)!=24:raise ValueError('Base threshold count changed')
            rule,diagnostic['calibration']=class_threshold(rule,views)
            scored,result=score_rule(rows,FEATURE_NAMES,rule,config)
            result.update({'fit_diagnostics':diagnostic,'threshold':rule.threshold})
            results[key]=result;rules[key]=rule;all_scores[key]=scored
            print(json.dumps({'candidate':key,'minimum_ba':result['minimum_domain_ba'],
                             'minimum_scene_ba':result['minimum_scene_ba'],'maxscene_drop':result['maximum_any_scene_drop']}),flush=True)
    null='source_source_shuffled_null';intervals={}
    for domain in ('rr','chimera'):
        groups=defaultdict(list)
        for r in all_scores[null]:
            if r['domain']==domain and r['variant']=='raw' and r['condition']!='original':groups[r['condition']].append(r)
        for condition,records in groups.items():
            records.sort(key=lambda r:r['src']);labels=np.array([int(r['label']=='FAKE') for r in records])
            intervals[domain+'/'+condition]=auc_intervals({condition:np.array([r['score'] for r in records])},labels,config)[condition]
    refused=all(v[0]>.5 for v in intervals.values())
    chosen=max(config['objectives'],key=lambda k:(min(results[k]['minimum_domain_ba'],results[k]['minimum_scene_ba']),
              -results[k]['maximum_any_scene_drop'],k=='mean')) if not refused else None
    for key,rule in rules.items():rule.save(OUTPUT/(key+'_rule.json'))
    write_json(OUTPUT/'selection_scores.json',all_scores)
    write_json(destination,{'candidates':results,'development_chosen':chosen,'interpretation_refused':refused,
         'null_raw_processed_auc_ci95':intervals,'elapsed_s':perf_counter()-start,'code_pins':code_pins(),
         'features_receipt_sha256':file_sha256(OUTPUT/'features.json'),'inventory_sha256':parent['inventory_sha256'],
         'selection_scores_sha256':file_sha256(OUTPUT/'selection_scores.json'),
         'rule_files':{k:file_sha256(OUTPUT/(k+'_rule.json')) for k in rules},'goal_achieved':False,
         'scope':'Frozen pretrained neural semantic encoder plus conventional head;exposed development'})


if __name__=='__main__':main()
