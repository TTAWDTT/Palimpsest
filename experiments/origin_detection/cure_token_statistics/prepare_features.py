"""Streaming numeric token cache;exact signed36 mean/probability controls."""

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image

from palimpsest.detection.representations.frozen_cure_tokens import FEATURE_NAMES,FrozenCureTokens
from palimpsest.detection.representations.frozen_cure import FEATURE_NAMES as OLD_NAMES,FULL_NAMES
from palimpsest.detection.algorithms.residual_statistics.features import resize256
from palimpsest.evaluation.numeric_features import numeric_rows
from palimpsest.evaluation.features import validate_feature_cache
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT,WORK_DIR,MODELS_ROOT
from experiments.origin_detection.cure_readout.prepare_features import OUTPUT as CURE
from experiments.origin_detection.cure_baseline.run_iteration import VENDOR,source_pins,runtime
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS,audit_receipt
from experiments.origin_detection.phase_statistics.run_iteration import image_path,write_json
from experiments.data_preparation.development_roles.audit_roles import audit_inventory

OUTPUT=WORK_DIR/'robust_statistics/cure_token_statistics'


def code_pins():
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md']
    paths+=[REPO_ROOT/p for p in ('src/palimpsest/detection/representations/frozen_cure_tokens.py',
        'src/palimpsest/detection/representations/token_statistics.py',
        'src/palimpsest/detection/representations/frozen_cure.py',
        'src/palimpsest/detection/baselines/cure.py','src/palimpsest/evaluation/numeric_features.py',
        'src/palimpsest/evaluation/features.py','src/palimpsest/evaluation/source_readout_campaign.py',
        'src/palimpsest/evaluation/source_training.py','src/palimpsest/evaluation/balanced_null.py',
        'src/palimpsest/evaluation/robust_views.py','src/palimpsest/detection/algorithms/readouts/source_view_risk.py',
        'experiments/origin_detection/cure_baseline/run_iteration.py',
        'experiments/origin_detection/cure_readout/prepare_features.py',
        'experiments/origin_detection/source_view_risk/run_iteration.py',
        'tests/detection/test_token_statistics.py','tests/evaluation/test_numeric_features.py')]
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in sorted(paths)}


def reference_data():
    receipt=audit_receipt(CURE/'features.json')
    if (file_sha256(CURE/'features.csv')!=receipt['csv_sha256'] or receipt['records']!=15120
            or not receipt['original_probability_exact']):raise ValueError('Signed CuRe source cache changed')
    old=audit_receipt(WORK_DIR/'robust_statistics/frozen_clip/features.json')
    if receipt['inventory_sha256']!=old['inventory_sha256']:raise ValueError('Token source inventory changed')
    means=np.empty((15120,1024),np.float32);probabilities=np.empty(15120,np.float64)
    index={};inventory=[]
    removed=set(OLD_NAMES)|{'variant','probability_fake','decode_preprocess_ms','encoder_ms'}
    with (CURE/'features.csv').open(newline='',encoding='utf-8') as stream:
        for i,r in enumerate(csv.DictReader(stream)):
            if i>=15120:raise ValueError('Extra parent reference row')
            key=r['filename'],r['variant']
            if key in index:raise ValueError('Duplicate token reference')
            index[key]=i;means[i]=[float(r[n]) for n in FULL_NAMES];probabilities[i]=float(r['probability_fake'])
            if r['variant']=='raw':
                meta={k:v for k,v in r.items() if k not in removed};inventory.append(meta)
    if len(index)!=15120 or len(inventory)!=6300 or not np.isfinite(means).all() or not np.isfinite(probabilities).all():
        raise ValueError('Parent token reference coverage/values invalid')
    structure=audit_inventory(inventory)
    if structure['sources']!=2100 or structure['source_role_counts']!={'fit':1260,'threshold':420,'selection':420}:
        raise ValueError('Token source roles changed')
    expected={(r['filename'],v) for r in inventory for v in (VARIANTS if r['role']=='selection' else VARIANTS[:2])}
    if set(index)!=expected:raise ValueError('Token reference variant coverage differs')
    return inventory,index,means,probabilities,receipt


def check_original(feature,mean,probability):
    if not np.array_equal(feature.values[:1024],mean) or feature.probability_fake!=probability:
        raise ValueError('Token original mean/probability changed')


def synthetic_gate(encoder,output):
    arrays={'flat':np.full((301,417,3),128,np.uint8),
            'channels':np.broadcast_to([32,128,224],(301,417,3)).astype(np.uint8),
            'spatial':np.random.default_rng(20261008).integers(0,256,(301,417,3),dtype=np.uint8)}
    result={}
    for key,pixels in arrays.items():
        path=output/(key+'.png')
        if path.exists():raise FileExistsError('Preserve token synthetic witness')
        Image.fromarray(pixels).save(path);a=encoder.extract_file(path);b=encoder.extract_file(path)
        if not np.array_equal(a.values,b.values) or a.probability_fake!=b.probability_fake:
            raise ValueError('Token synthetic repeated descriptor changed')
        direct=encoder.direct_features(path);check_original(a,direct.values[:1024],direct.probability_fake)
        corrupt=a.values.copy();corrupt[0]+=.001
        try:check_original(replace(a,values=corrupt),direct.values[:1024],direct.probability_fake)
        except ValueError:pass
        else:raise ValueError('Corrupt token mean accepted')
        result[key]={'repeat_exact':True,'direct_hook_disabled_mean_probability_exact':True,
                     'wrong_mean_rejected':True,'sha256':file_sha256(path)}
    return result


def audit_cache(metadata,matrix,inventory):
    rows=numeric_rows(metadata,matrix,FEATURE_NAMES)
    for role in ('fit','threshold','selection'):
        chosen=[r for r in inventory if r['role']==role]
        actual=[r for r in rows if r['role']==role]
        if chosen:validate_feature_cache(actual,chosen,FEATURE_NAMES,
            variants=VARIANTS if role=='selection' else VARIANTS[:2],bounds=(-np.inf,np.inf))
    spread=np.asarray(matrix[:,2048:])
    if np.any(spread<0) or np.any(spread>1+1e-12):raise ValueError('Token spread bounds differ')
    norms=np.sqrt(np.sum(spread*spread,axis=1))
    if np.any((norms!=0)&(np.abs(norms-1)>1e-12)):raise ValueError('Token spread unit norm differs')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--pilot',action='store_true');g.add_argument('--prepare',action='store_true');args=parser.parse_args()
    output=OUTPUT.with_name(OUTPUT.name+'_pilot') if args.pilot else OUTPUT;output.mkdir(exist_ok=True,parents=True)
    if any((output/n).exists() for n in ('features.json','vectors.npy','vectors.partial.npy','metadata.partial.csv')):
        raise FileExistsError('Preserve token cache or unfinished evidence')
    controls=json.loads((OUTPUT/'software_controls.json').read_text())
    if not controls['passed'] or controls['code_pins']!=code_pins():raise ValueError('Token software controls changed')
    inventory,index,means,probabilities,parent=reference_data()
    if args.pilot:
        p=CURE.with_name(CURE.name+'_pilot');old=json.loads((p/'features.json').read_text())
        if old['csv_sha256']!=file_sha256(p/'features.csv'):raise ValueError('Old exposed pilot changed')
        names={r['filename'] for r in read_rows(p/'features.csv')};inventory=[r for r in inventory if r['filename'] in names]
        if len(inventory)!=60 or any(r['role']!='selection' for r in inventory):raise ValueError('Expected exposed60pilot')
    cv2.setNumThreads(1);packages=runtime()
    if not args.pilot:
        p=OUTPUT.with_name(OUTPUT.name+'_pilot');pilot=json.loads((p/'features.json').read_text())
        if (pilot['code_pins']!=code_pins() or pilot['runtime']!=packages or not pilot['cost_passed']
                or not pilot['all_original_exact'] or pilot['repeated_raw_records']!=60
                or file_sha256(p/'vectors.npy')!=pilot['vectors_sha256']
                or file_sha256(p/'metadata.csv')!=pilot['metadata_sha256']):raise ValueError('Token pilot gate changed/failed')
    encoder=FrozenCureTokens(VENDOR,VENDOR/'weights/cure_adapter.pt',MODELS_ROOT/'cure/PE-Core-L14-336.pt',source_pins=source_pins())
    witnesses=synthetic_gate(encoder,output) if args.pilot else pilot['synthetic_witnesses']
    count=sum(len(VARIANTS if r['role']=='selection' else VARIANTS[:2]) for r in inventory)
    partial=output/'vectors.partial.npy';matrix=np.lib.format.open_memmap(partial,mode='w+',dtype='<f8',shape=(count,3072))
    matrix[:]=np.nan;metadata=[];durations=[];repeated=0;cursor=0;start=perf_counter();scratch=output/'query.jpg'
    if scratch.exists():raise FileExistsError('Unfinished token query requires audit')
    try:
        for i,r in enumerate(inventory,1):
            path=image_path(r)
            if file_sha256(path)!=r['sha256']:raise ValueError('Token native pixel bytes changed')
            with Image.open(path) as opened:
                if opened.size!=(int(r['width']),int(r['height'])):raise ValueError('Token native dimensions changed')
                pixels=np.asarray(opened.convert('RGB'))
            for variant in VARIANTS if r['role']=='selection' else VARIANTS[:2]:
                query=path
                if variant!='raw':
                    q,s=(90,0) if variant==VARIANTS[1] else (70,2) if variant==VARIANTS[2] else (60,2)
                    Image.fromarray(resize256(pixels)).save(scratch,format='JPEG',quality=q,subsampling=s);query=scratch
                feature=encoder.extract_file(query);j=index[r['filename'],variant]
                check_original(feature,means[j],probabilities[j])
                if args.pilot and variant=='raw':
                    repeat=encoder.extract_file(query)
                    if not np.array_equal(feature.values,repeat.values) or feature.probability_fake!=repeat.probability_fake:
                        raise ValueError('Real token repeated descriptor changed')
                    repeated+=1
                matrix[cursor]=feature.values;cursor+=1;durations.append(feature.elapsed_ms)
                metadata.append({**r,'variant':variant,'probability_fake':feature.probability_fake,
                    'elapsed_ms':feature.elapsed_ms,'statistics_ms':feature.statistics_ms,**feature.diagnostics})
            if i%60==0 or i==len(inventory):print(json.dumps({'images':i,'records':cursor,'total':len(inventory),
                'elapsed_s':round(perf_counter()-start,3)}),flush=True)
    finally:encoder.close()
    if cursor!=count:raise ValueError('Incomplete token numeric rows')
    audit_cache(metadata,matrix,inventory);matrix.flush();del matrix
    if scratch.exists():scratch.unlink()
    partial.rename(output/'vectors.npy')
    with (output/'metadata.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metadata[0]));writer.writeheader();writer.writerows(metadata)
    median=float(np.median(durations));cost_passed=median<=150 if args.pilot else None
    write_json(output/'features.json',{'images':len(inventory),'records':cursor,'feature_names':FEATURE_NAMES,
        'vectors_sha256':file_sha256(output/'vectors.npy'),'metadata_sha256':file_sha256(output/'metadata.csv'),
        'parent_mean_cache_sha256':parent['csv_sha256'],'inventory_sha256':parent['inventory_sha256'],
        'code_pins':code_pins(),'runtime':packages,'encoder':encoder.provenance,'synthetic_witnesses':witnesses,
        'all_original_exact':True,'repeated_raw_records':repeated,'cost_median_ms':median,'cost_passed':cost_passed,
        'elapsed_s':perf_counter()-start,'scope':'New token statistics on exposed pixels;not physical invariance or deployment speed'})
    if args.pilot and not cost_passed:raise ValueError('Token pilot engineering budget refused')


if __name__=='__main__':main()
