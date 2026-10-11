"""Signed sixth iteration: bounded RGB residuals, portable forests, no RR reserved."""

import argparse
import csv
import json
from io import BytesIO
import platform
import random
from time import perf_counter
import tomllib

import cv2
import numpy as np
from PIL import Image

from palimpsest.contracts import Origin, Prediction
from palimpsest.detection.algorithms.forest import ForestRule
from palimpsest.detection.algorithms.residual_statistics.features import FEATURE_NAMES, extract_features, resize256
from palimpsest.detection.algorithms.residual_statistics.detector import ResidualDetector
from palimpsest.evaluation.classification import evaluate
from palimpsest.evaluation.detection import DetectionObservation, evaluate_detection
from palimpsest.evaluation.features import validate_feature_cache
from palimpsest.evaluation.file_benchmark import benchmark_files
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.phase_statistics.run_iteration import inherited_inventory, image_path, write_json
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals
from .fit_rules import all_views, fit_rule

CONFIG = REPO_ROOT/'configs/evaluation/joint_residual_statistics.toml'
OUTPUT = WORK_DIR/'robust_statistics/joint_residual_statistics'
VARIANTS = ('raw','jpeg90_444_after_resize256')


def code_pins():
    files = [CONFIG,REPO_ROOT/'src/palimpsest/detection/algorithms/forest.py',
             REPO_ROOT/'src/palimpsest/detection/algorithms/ordinal_statistics/features.py',
             REPO_ROOT/'src/palimpsest/evaluation/features.py',REPO_ROOT/'src/palimpsest/evaluation/file_benchmark.py']
    for directory in (REPO_ROOT/'src/palimpsest/detection/algorithms/residual_statistics',
                      REPO_ROOT/'experiments/origin_detection/residual_statistics'):
        files.extend(directory.glob('*.py'));files.append(directory/'README.md')
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in sorted(files)}


def audit_cache(rows, inventory):
    validate_feature_cache(rows,inventory,FEATURE_NAMES,variants=VARIANTS,bounds=(0,1))
    for r in rows:
        values=np.array([float(r[n]) for n in FEATURE_NAMES])
        if any(not np.isclose(values[j:j+6].sum(),1,atol=1e-12,rtol=0) for j in range(0,28,7)):
            raise ValueError('Unnormalized ordinal histogram')
        if any(not np.isclose(values[j:j+39].sum(),1,atol=1e-12,rtol=0) for j in range(28,379,39)):
            raise ValueError('Unnormalized residual histogram')
    return rows


def extract_inventory(inventory, directory):
    start=perf_counter();records=[]
    for index,row in enumerate(inventory,1):
        path=image_path(row)
        if file_sha256(path)!=row['sha256']:
            raise ValueError('Residual input SHA changed')
        with Image.open(path) as image:
            if image.size!=(int(row['width']),int(row['height'])):
                raise ValueError('Residual input dimensions changed')
            rgb=np.asarray(image.convert('RGB'))
        for variant in VARIANTS:
            if variant=='raw': pixels=rgb
            else:
                stream=BytesIO();Image.fromarray(resize256(rgb)).save(stream,format='JPEG',quality=90,subsampling=0)
                stream.seek(0)
                with Image.open(stream) as image:pixels=np.asarray(image.convert('RGB'))
            f=extract_features(pixels)
            records.append({**row,'variant':variant,**dict(zip(FEATURE_NAMES,f.values.tolist())),
                            'preprocess_ms':f.preprocess_ms,'statistics_ms':f.statistics_ms})
        if index%60==0 or index==len(inventory):
            print(json.dumps({'images':index,'total':len(inventory),'elapsed_s':round(perf_counter()-start,3)}),flush=True)
    with (directory/'features.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    audit_cache(records,inventory)
    return {'images':len(inventory),'records':len(records),'elapsed_s':perf_counter()-start,
            'csv_sha256':file_sha256(directory/'features.csv'),'code_pins':code_pins()}


def signed_features():
    receipt=json.loads((OUTPUT/'features.json').read_text(encoding='utf-8'))
    if receipt['code_pins']!=code_pins() or receipt['csv_sha256']!=file_sha256(OUTPUT/'features.csv'):
        raise ValueError('Frozen residual code/cache changed')
    inventory,sha=inherited_inventory()
    if sha!=receipt['inventory_sha256']:
        raise ValueError('Inherited residual inventory changed')
    return audit_cache(read_rows(OUTPUT/'features.csv'),inventory),receipt


def aggregate(rows, names, rule, variant):
    output={}
    for domain in ('rr','chimera'):
        selected=[r for r in rows if r['role']=='selection' and r['domain']==domain and r['variant']==variant]
        scores=rule.score([[float(r[n]) for n in names] for r in selected])
        output[domain]=evaluate_detection([
            DetectionObservation(r['src'],r['condition'],Origin.AI if r['label']=='FAKE' else Origin.NATURAL,
                                 Prediction('residual-statistics',float(s),rule.threshold,'statistical_score'))
            for r,s in zip(selected,scores)])
    return output


def screen_rule(rows,names,rule,config):
    strata={}
    for key,(records,values,labels) in all_views(rows,names,'selection').items():
        scores=rule.score(values)
        strata[key]={**evaluate([{**r,'score':float(s-rule.threshold)} for r,s in zip(records,scores)],'score'),
                     'auc_ci95':auc_intervals({key:scores},labels,config)[key]}
    if len(strata)!=12:raise ValueError('Selection view count differs')
    raw,encoded=aggregate(rows,names,rule,'raw'),aggregate(rows,names,rule,VARIANTS[1])
    processed=[v for k,v in strata.items() if not k.endswith('/original')]
    changes=[(raw[d]['conditions'][c]['balanced_accuracy_at_zero'],v['balanced_accuracy_at_zero'])
             for d,z in encoded.items() for c,v in z['conditions'].items() if c!='original']
    criteria={
        'eight_auc_lower_bounds':all(v['auc_ci95'][0]>config['selection_auc_lower_gate'] for v in processed),
        'eight_both_class_accuracies':all(v[k]>=config['minimum_processed_class_accuracy'] for v in processed
                                        for k in ('real_accuracy_at_zero','fake_accuracy_at_zero')),
        'both_original_class_accuracies':all(z['conditions']['original'][k]>=config['minimum_original_class_accuracy']
                                             for z in raw.values() for k in ('real_accuracy_at_zero','fake_accuracy_at_zero')),
        'recompression_ba_floor':all(after>=config['minimum_reencoded_ba'] for _,after in changes),
        'recompression_ba_drop':all(before-after<=config['maximum_reencoded_ba_drop'] for before,after in changes)}
    return {'criteria':criteria,'gate_passed':all(criteria.values()),'threshold':rule.threshold,
            'selection_views':strata,'selection_aggregate':raw,'recompression_aggregate':encoded,
            'worst_processed_auc':min(v['auc'] for v in processed)}


def run_screen(rows,config,manifest_sha):
    output,rules={},{}
    for mode in config['candidate_modes']:
        names=FEATURE_NAMES[:28] if mode=='ordinal256' else FEATURE_NAMES
        rule=fit_rule(rows,names,seed=config['seed'],manifest_sha=manifest_sha)
        output[mode]=screen_rule(rows,names,rule,config);rules[mode]=rule
    null=fit_rule(rows,FEATURE_NAMES,seed=config['seed'],manifest_sha=manifest_sha,shuffled=True)
    output['hybrid_source_shuffled_null']=screen_rule(rows,FEATURE_NAMES,null,config)
    passing=[m for m in config['candidate_modes'] if output[m]['gate_passed']]
    chosen=max(passing,key=lambda m:output[m]['worst_processed_auc']) if passing else None
    if output['hybrid_source_shuffled_null']['gate_passed']:chosen=None
    return {'candidates':output,'chosen':chosen,'interpretation_refused':output['hybrid_source_shuffled_null']['gate_passed'],
            'scope':'Sixth RR/Chimera development iteration; repeated sources, no blind test'},rules


def benchmark(rows,rules):
    prior=WORK_DIR/'robust_statistics/rr_first_iteration'
    receipt=json.loads((prior/'benchmark.json').read_text(encoding='utf-8'))
    if file_sha256(prior/'benchmark.csv')!=receipt['csv_sha256']:raise ValueError('Benchmark selection changed')
    names=sorted({'rr/'+r['filename'] for r in read_rows(prior/'benchmark.csv')})
    by_name={r['filename']:r for r in rows if r['variant']=='raw'}
    selected=[by_name[n] for n in names]
    if len(selected)!=60:raise ValueError('Benchmark denominator differs')
    methods={}
    for mode,rule in rules.items():
        detector=ResidualDetector(rule)
        detector.predict(np.full((256,256,3),128,np.uint8))
        values=[[float(r[n]) for n in rule.feature_names] for r in selected]
        scores=rule.score(values)
        items=[{'filename':r['filename'],'path':image_path(r),'sha256':r['sha256'],'expected_score':float(s)}
               for r,s in zip(selected,scores)]
        methods[mode]=benchmark_files(detector,items,seed=20261006)
    return {'methods':methods,'processor':platform.processor(),'opencv_threads':cv2.getNumThreads(),
            'python_version':platform.python_version(),'prior_selection_csv_sha256':receipt['csv_sha256']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group();group.add_argument('--pilot',type=int,default=0)
    group.add_argument('--prepare',action='store_true');group.add_argument('--benchmark',action='store_true')
    args=parser.parse_args()
    if args.pilot<0:parser.error('Pilot must be positive')
    cv2.setNumThreads(1)
    with CONFIG.open('rb') as stream:config=tomllib.load(stream)
    if args.prepare or args.pilot:
        inventory,sha=inherited_inventory()
        directory=OUTPUT if args.prepare else OUTPUT.with_name('joint_residual_pilot')
        if directory.exists():raise FileExistsError('Preserve residual extraction results')
        if args.pilot:
            random.Random(config['seed']).shuffle(inventory);inventory=inventory[:args.pilot]
        directory.mkdir(parents=True)
        result=extract_inventory(inventory,directory);result['inventory_sha256']=sha
        write_json(directory/'features.json',result);return
    rows,receipt=signed_features()
    path=OUTPUT/('benchmark.json' if args.benchmark else 'iteration.json')
    if path.exists():raise FileExistsError('Preserve residual evaluation')
    if args.benchmark:
        previous=json.loads((OUTPUT/'iteration.json').read_text(encoding='utf-8'))
        if previous['features_receipt_sha256']!=file_sha256(OUTPUT/'features.json'):raise ValueError('Iteration cache link changed')
        rules={}
        for mode,sha in previous['rule_files'].items():
            rule_path=OUTPUT/f'{mode}_rule.json'
            if file_sha256(rule_path)!=sha:raise ValueError('Frozen residual rule changed')
            rules[mode]=ForestRule.load(rule_path)
        result=benchmark(rows,rules)
    else:
        result,rules=run_screen(rows,config,receipt['inventory_sha256'])
        for mode,rule in rules.items():rule.save(OUTPUT/f'{mode}_rule.json')
        result['rule_files']={m:file_sha256(OUTPUT/f'{m}_rule.json') for m in rules}
    result['features_receipt_sha256']=file_sha256(OUTPUT/'features.json');result['code_pins']=code_pins()
    write_json(path,result);print(json.dumps({'output':path.name,'chosen':result.get('chosen')}))


if __name__=='__main__':main()
