"""Fixed augmentation x source-risk comparison; selection never enters fitting."""

import argparse
from collections import defaultdict
import json
import random
from time import perf_counter
import tomllib

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.paired_stability import StableRule
from palimpsest.detection.algorithms.source_view_risk import fit_source_risk
from palimpsest.detection.algorithms.residual_statistics.features import resize256
from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES, FrozenClip
from palimpsest.evaluation.features import feature_views
from palimpsest.evaluation.pixel_features import audit_variants, extract_inventory
from palimpsest.evaluation.robust_views import evaluate_views
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR, MODELS_ROOT
from experiments.origin_detection.paired_stability.run_iteration import INTERVENTION
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json
from experiments.origin_detection.residual_training.fit_rules import TRAINING_VARIANT
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals

CONFIG = REPO_ROOT/'configs/evaluation/source_view_risk.toml'
PARENT = WORK_DIR/'robust_statistics/frozen_clip'
OUTPUT = WORK_DIR/'robust_statistics/source_view_risk'
Q60 = 'jpeg60_420_after_resize256'
VARIANTS = ('raw', TRAINING_VARIANT, INTERVENTION, Q60)


def code_pins():
    files = [CONFIG, REPO_ROOT/'src/palimpsest/detection/algorithms/source_view_risk.py',
             REPO_ROOT/'src/palimpsest/detection/algorithms/paired_stability.py',
             REPO_ROOT/'src/palimpsest/detection/representations/frozen_clip.py',
             REPO_ROOT/'src/palimpsest/evaluation/robust_views.py',
             REPO_ROOT/'src/palimpsest/evaluation/pairing.py',
             REPO_ROOT/'src/palimpsest/evaluation/classification.py',
             REPO_ROOT/'src/palimpsest/evaluation/pixel_features.py',
             REPO_ROOT/'src/palimpsest/evaluation/features.py',
             REPO_ROOT/'experiments/origin_detection/threshold_calibration/fit_threshold.py',
             REPO_ROOT/'experiments/origin_detection/robust_statistics/rr/evaluate_features.py',
             REPO_ROOT/'tests/detection/test_source_view_risk.py',
             REPO_ROOT/'tests/evaluation/test_robust_views.py']
    directory = REPO_ROOT/'experiments/origin_detection/source_view_risk'
    files.extend(directory.glob('*.py')); files.append(directory/'README.md')
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(files)}


def audit_receipt(path):
    receipt = json.loads(path.read_text(encoding='utf-8'))
    for name, expected in receipt['code_pins'].items():
        if file_sha256(REPO_ROOT/name) != expected:
            raise ValueError(f'Parent code changed: {name}')
    return receipt


def parent_data():
    receipt = audit_receipt(PARENT/'features.json')
    iteration = audit_receipt(PARENT/'iteration.json')
    if (receipt['csv_sha256'] != file_sha256(PARENT/'features.csv')
            or iteration['features_receipt_sha256'] != file_sha256(PARENT/'features.json')
            or receipt['inventory_sha256'] != iteration['inventory_sha256']):
        raise ValueError('Parent feature/receipt/inventory changed')
    rows = read_rows(PARENT/'features.csv')
    removed = set(FEATURE_NAMES) | {'variant', 'preprocess_ms', 'statistics_ms'}
    inventory = [{k:v for k,v in r.items() if k not in removed} for r in rows if r['variant']=='raw']
    audit_variants(rows, inventory, FEATURE_NAMES, ordinary_variants=VARIANTS[:2], selection_variant=INTERVENTION)
    if len(inventory)!=6300 or len(rows)!=13860:
        raise ValueError('Frozen parent counts differ')
    return rows, inventory, receipt, iteration


def training_arrays(rows, variants, seed, *, shuffled=False):
    records = sorted([r for r in rows if r['role']=='fit' and r['variant'] in variants],
                     key=lambda r:tuple(r[k] for k in ('domain','src','condition','variant')))
    grouped = defaultdict(list)
    for r in records: grouped[(r['domain'],r['src'])].append(r)
    domains = defaultdict(list)
    for key, selected in sorted(grouped.items()):
        if len(selected)!=3*len(variants) or len({r['label'] for r in selected})!=1:
            raise ValueError('Incomplete/conflicting fit source views')
        domains[key[0]].append(key)
    if {k:len(v) for k,v in domains.items()}!={'rr':540,'chimera':720}:
        raise ValueError('Frozen fit source/domain counts differ')
    labels = {key:int(selected[0]['label']=='FAKE') for key,selected in grouped.items()}
    if shuffled:
        strata=defaultdict(list)
        for key, selected in sorted(grouped.items()): strata[(key[0],selected[0]['scene'])].append(key)
        rng=random.Random(seed)
        for keys in strata.values():
            values=[labels[k] for k in keys];rng.shuffle(values)
            labels.update(zip(keys,values))
    indices={key:i for i,key in enumerate(sorted(grouped))}
    x=np.array([[float(r[n]) for n in FEATURE_NAMES] for r in records])
    keys=[(r['domain'],r['src']) for r in records]
    y=np.array([labels[k] for k in keys])
    weights=np.array([.5/len(domains[k[0]])/len(grouped[k]) for k in keys])
    return x,y,weights,np.array([indices[k] for k in keys])


def evaluate_rule(rows, rule, config):
    selected=[r for r in rows if r['role']=='selection']
    x=np.array([[float(r[n]) for n in FEATURE_NAMES] for r in selected])
    scores=rule.score(x)
    if not np.array_equal(scores,[rule.score([v])[0] for v in x]):
        raise ValueError('Single/batch scores differ')
    scored=[{**{k:r[k] for k in ('domain','scene','condition','variant','src','role','label')},
             'score':float(s-rule.threshold)} for r,s in zip(selected,scores)]
    result=evaluate_views(scored,variant_order=VARIANTS,minimum_ba=config['provisional_final_ba'],
                          maximum_drop=config['provisional_final_drop'])
    return scored,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--pilot',type=int,default=0)
    group.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.pilot<0: parser.error('Pilot must be positive')
    config=tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    control_path=OUTPUT/'controls.json'
    control=json.loads(control_path.read_text(encoding='utf-8'))
    if control['returncode']!=0 or any(file_sha256(REPO_ROOT/k)!=v for k,v in control['test_pins'].items()):
        raise ValueError('Source-risk controls failed/changed')
    old,inventory,parent,previous=parent_data()
    cv2.setNumThreads(1)
    if args.prepare or args.pilot:
        directory=OUTPUT if args.prepare else OUTPUT.with_name('source_view_risk_pilot')
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/'features.json').exists(): raise FileExistsError('Preserve extraction receipt')
        encoder=FrozenClip(MODELS_ROOT/'d3/ViT-L-14.pt',device=config['device'])
        if args.pilot:
            random.Random(config['seed']).shuffle(inventory);inventory=inventory[:args.pilot]
        result=extract_inventory(inventory,directory/'features.csv',names=FEATURE_NAMES,extractor=encoder.extract,
                                 resolve_path=image_path,resize=resize256,ordinary_variants=(INTERVENTION,),
                                 selection_variant=Q60,jpeg_parameters={INTERVENTION:(70,2),Q60:(60,2)})
        extra=read_rows(directory/'features.csv')
        lookup={(r['filename'],r['variant']):r for r in old if r['variant']==INTERVENTION}
        repeated=[r for r in extra if r['variant']==INTERVENTION and r['role']=='selection']
        if any([float(r[n]) for n in FEATURE_NAMES]!=[float(lookup[r['filename'],INTERVENTION][n]) for n in FEATURE_NAMES] for r in repeated):
            raise ValueError('Regenerated selection Q70 differs from parent')
        write_json(directory/'features.json',{**result,'code_pins':code_pins(),'encoder':encoder.provenance,
                   'inventory_sha256':parent['inventory_sha256'],'controls_sha256':file_sha256(control_path),
                   'parent_features_receipt_sha256':file_sha256(PARENT/'features.json'),
                   'repeated_selection_q70_exact':True,'repeated_selection_q70_records':len(repeated)})
        return
    if (OUTPUT/'iteration.json').exists(): raise FileExistsError('Preserve source-risk evaluation')
    receipt=audit_receipt(OUTPUT/'features.json')
    if (receipt['code_pins']!=code_pins() or receipt['csv_sha256']!=file_sha256(OUTPUT/'features.csv')
            or receipt['parent_features_receipt_sha256']!=file_sha256(PARENT/'features.json')
            or not receipt['repeated_selection_q70_exact']):
        raise ValueError('Additional feature/code lineage changed')
    new=read_rows(OUTPUT/'features.csv')
    audit_variants(new,inventory,FEATURE_NAMES,ordinary_variants=(INTERVENTION,),selection_variant=Q60)
    rows=[r for r in old if r['variant']!=INTERVENTION]+new
    if len(new)!=7560 or len(rows)!=20160: raise ValueError('Additional/joined feature counts differ')
    start=perf_counter();results={};rules={};scored_results={}
    candidates=[('weak','mean',False),('weak','source',False),('strong','mean',False),('strong','source',False),('strong','source',True),('reference','both',False)]
    with threadpool_limits(limits=1):
        for training,objective,shuffled in candidates:
            key=training+'/'+objective+('_source_shuffled_null' if shuffled else '')
            if training=='reference' or (training=='weak' and objective=='mean'):
                parent_key='clip/'+('both' if training=='reference' else 'erm')
                rule_path=PARENT/(parent_key.replace('/','_')+'_rule.json')
                if file_sha256(rule_path)!=previous['rule_files'][parent_key]: raise ValueError('Reference rule changed')
                rule=StableRule.load(rule_path);diagnostic={'reused_parent_rule_sha256':file_sha256(rule_path)}
            else:
                variants=VARIANTS[:2] if training=='weak' else VARIANTS[:3]
                x,y,weights,sources=training_arrays(rows,variants,config['seed'],shuffled=shuffled)
                rule,diagnostic=fit_source_risk(x,y,weights,sources,feature_names=FEATURE_NAMES,
                  temperature=config['source_temperature'] if objective=='source' else 0,ridge=config['ridge'],
                  scale_floor=config['scale_floor'],maximum_iterations=config['maximum_iterations'],
                  gradient_tolerance=config['gradient_tolerance'],manifest_sha=parent['inventory_sha256'])
                views={key+'/'+v:view for v in VARIANTS[:2] for processed in (False,True)
                       for key,view in feature_views(rows,FEATURE_NAMES,'threshold',processed=processed,variant=v).items()}
                if len(views)!=24: raise ValueError('Threshold views changed')
                rule,diagnostic['calibration']=class_threshold(rule,views)
            scored,result=evaluate_rule(rows,rule,config)
            result['fit_diagnostics']=diagnostic
            result['threshold']=rule.threshold
            rules[key]=rule;results[key]=result;scored_results[key]=scored
            print(json.dumps({'candidate':key,'minimum_ba':result['minimum_domain_ba'],
                  'maximum_any_scene_drop':result['maximum_any_scene_drop'],'worst_pair':result['worst_pair']}),flush=True)
    null='strong/source_source_shuffled_null'
    null_intervals={}
    for domain in ('rr','chimera'):
        groups=defaultdict(list)
        for r in scored_results[null]:
            if r['domain']==domain and r['variant']=='raw' and r['condition']!='original':
                groups[r['condition']].append(r)
        for condition,records in groups.items():
            records.sort(key=lambda r:r['src'])
            labels=np.array([int(r['label']=='FAKE') for r in records])
            null_intervals[domain+'/'+condition]=auc_intervals({condition:np.array([r['score'] for r in records])},labels,config)[condition]
    refused=all(v[0]>.5 for v in null_intervals.values())
    for key,rule in rules.items(): rule.save(OUTPUT/(key.replace('/','_')+'_rule.json'))
    write_json(OUTPUT/'selection_scores.json',scored_results)
    write_json(OUTPUT/'iteration.json',{'candidates':results,'interpretation_refused':refused,'null_raw_processed_auc_ci95':null_intervals,
               'elapsed_s':perf_counter()-start,'code_pins':code_pins(),'single_batch_scores_exact':True,
               'inventory_sha256':parent['inventory_sha256'],'features_receipt_sha256':file_sha256(OUTPUT/'features.json'),
               'parent_iteration_sha256':file_sha256(PARENT/'iteration.json'),'controls_sha256':file_sha256(control_path),
               'selection_scores_sha256':file_sha256(OUTPUT/'selection_scores.json'),
               'rule_files':{k:file_sha256(OUTPUT/(k.replace('/','_')+'_rule.json')) for k in rules},
               'goal_achieved':False,'scope':'Exposed development; frozen neural encoder plus fitted conventional head; no independent certification'})


if __name__=='__main__': main()
