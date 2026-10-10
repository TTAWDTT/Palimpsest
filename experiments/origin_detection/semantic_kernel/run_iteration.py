"""Fixed nonlinear readout of the existing CLIP cache; no new pixel features."""

import argparse
from collections import defaultdict
import json
from time import perf_counter
import tomllib

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.kernel_readout import fit_fourier_map,KernelRule
from palimpsest.detection.algorithms.compiled_kernel import CompiledKernelRule
from palimpsest.detection.algorithms.readouts.source_view_risk import fit_source_risk
from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES
from palimpsest.evaluation.features import feature_views
from palimpsest.evaluation.pixel_features import audit_variants
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT,WORK_DIR
from experiments.origin_detection.source_view_risk.run_iteration import parent_data,VARIANTS,Q60,audit_receipt
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals

CONFIG=REPO_ROOT/'configs/evaluation/semantic_kernel.toml'
OUTPUT=WORK_DIR/'robust_statistics/semantic_kernel'
PARENT=WORK_DIR/'robust_statistics/source_view_risk'


def code_pins():
    files=[CONFIG,REPO_ROOT/'src/palimpsest/detection/algorithms/kernel_readout.py',
       REPO_ROOT/'src/palimpsest/detection/algorithms/compiled_kernel.py',
       REPO_ROOT/'tests/detection/test_compiled_kernel.py',
       REPO_ROOT/'src/palimpsest/detection/algorithms/readouts/source_view_risk.py',
       REPO_ROOT/'src/palimpsest/evaluation/source_training.py',
       REPO_ROOT/'src/palimpsest/evaluation/robust_views.py',
       REPO_ROOT/'experiments/origin_detection/source_view_risk/run_iteration.py',
       REPO_ROOT/'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
       REPO_ROOT/'tests/detection/test_kernel_readout.py',
       REPO_ROOT/'tests/detection/test_source_view_risk.py',
       REPO_ROOT/'experiments/origin_detection/threshold_calibration/fit_threshold.py']
    directory=REPO_ROOT/'experiments/origin_detection/semantic_kernel'
    files.extend(directory/name for name in ('run_iteration.py','README.md'))
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in sorted(files)}


def inputs():
    old,inventory,parent,_=parent_data()
    receipt=audit_receipt(PARENT/'features.json')
    if (receipt['csv_sha256']!=file_sha256(PARENT/'features.csv')
            or not receipt['repeated_selection_q70_exact'] or receipt['inventory_sha256']!=parent['inventory_sha256']):
        raise ValueError('Semantic source cache changed')
    new=read_rows(PARENT/'features.csv')
    audit_variants(new,inventory,FEATURE_NAMES,ordinary_variants=(VARIANTS[2],),selection_variant=Q60)
    return [r for r in old if r['variant']!=VARIANTS[2]]+new,parent


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    path=OUTPUT/('pilot.json' if args.pilot else 'iteration.json')
    if path.exists():raise FileExistsError('Preserve semantic nonlinear trial')
    config=tomllib.loads(CONFIG.read_text(encoding='utf-8'))
    controls_path=OUTPUT/'controls.json';controls=json.loads(controls_path.read_text())
    if controls['returncode']!=0 or any(file_sha256(REPO_ROOT/k)!=v for k,v in controls['test_pins'].items()):
        raise ValueError('Semantic nonlinear controls failed/changed')
    rows,parent=inputs();start=perf_counter()
    with threadpool_limits(limits=1):
        x,_,weights,_=weighted_source_arrays(rows,FEATURE_NAMES,VARIANTS[:2],
                         expected_source_counts={'rr':540,'chimera':720},seed=config['seed'])
        mapper=fit_fourier_map(x,weights,feature_names=FEATURE_NAMES,frequency_count=config['frequency_count'],
                              seed=config['seed'],scale_floor=config['scale_floor'])
        matrix=np.array([[float(r[n]) for n in FEATURE_NAMES] for r in rows])
        values=mapper.transform(matrix)
        mapped=[{**r,**dict(zip(mapper.output_names,v.tolist()))} for r,v in zip(rows,values)]
        mapping_s=perf_counter()-start
        if args.pilot:
            x,y,weights,sources=weighted_source_arrays(mapped,mapper.output_names,VARIANTS[:2],
                        expected_source_counts={'rr':540,'chimera':720},seed=config['seed'])
            _,diagnostic=fit_source_risk(x,y,weights,sources,feature_names=mapper.output_names,temperature=0,
               ridge=config['ridge'],scale_floor=config['scale_floor'],maximum_iterations=config['maximum_iterations'],
               gradient_tolerance=config['gradient_tolerance'],manifest_sha=parent['inventory_sha256'])
            write_json(path,{'code_pins':code_pins(),'fit_diagnostics':diagnostic,'elapsed_s':perf_counter()-start,
                             'mapping_s':mapping_s,'scope':'Fit-role only costpilot;no threshold/selection evaluation'})
            print(json.dumps({'pilot_s':perf_counter()-start}),flush=True);return
        results={};rules={};scored_results={}
        candidates=[(mode,obj,False) for mode in config['representations'] for obj in config['objectives']]
        candidates.append(('hybrid_fourier','source',True))
        for mode,objective,shuffled in candidates:
            names=(FEATURE_NAMES if mode=='hybrid_fourier' else ())+mapper.output_names
            key=mode+'/'+objective+('_source_shuffled_null' if shuffled else '')
            x,y,weights,sources=weighted_source_arrays(mapped,names,VARIANTS[:2],
                   expected_source_counts={'rr':540,'chimera':720},seed=config['seed'],shuffled=shuffled)
            head,diagnostic=fit_source_risk(x,y,weights,sources,feature_names=names,
                   temperature=config['source_temperature'] if objective=='source' else 0,
                   ridge=config['ridge'],scale_floor=config['scale_floor'],maximum_iterations=config['maximum_iterations'],
                   gradient_tolerance=config['gradient_tolerance'],manifest_sha=parent['inventory_sha256'])
            views={key+'/'+v:view for v in VARIANTS[:2] for processed in (False,True)
                   for key,view in feature_views(mapped,names,'threshold',processed=processed,variant=v).items()}
            if len(views)!=24:raise ValueError('Semantic threshold views changed')
            head,diagnostic['calibration']=class_threshold(head,views)
            kernel=KernelRule(mapper,head,mode=='hybrid_fourier')
            scored,result=score_rule(mapped,names,head,config)
            native=np.array([[float(r[n]) for n in FEATURE_NAMES] for r in rows if r['role']=='selection'])
            expected=np.array([r['score']+head.threshold for r in scored])
            direct=kernel.score(native);compiled=CompiledKernelRule(kernel)
            # Re-adding a threshold may round,so compare head directly as well.
            actual_head=head.score(np.array([[float(r[n]) for n in names] for r in mapped if r['role']=='selection']))
            if (not np.array_equal(direct,actual_head) or not np.array_equal(direct,compiled.score(native))
                    or not np.array_equal(direct,[compiled.score([v])[0] for v in native])):
                raise ValueError('Portable kernel mapped/single/batch differs')
            if not np.allclose(direct,expected,atol=1e-14,rtol=0):raise ValueError('Scored margins differ')
            result.update({'fit_diagnostics':diagnostic,'threshold':head.threshold,'portable_mapped_single_batch_exact':True})
            rules[key]=kernel;results[key]=result;scored_results[key]=scored
            print(json.dumps({'candidate':key,'minimum_ba':result['minimum_domain_ba'],
                      'minimum_scene_ba':result['minimum_scene_ba'],'maxscene_drop':result['maximum_any_scene_drop']}),flush=True)
    null='hybrid_fourier/source_source_shuffled_null';intervals={}
    for domain in ('rr','chimera'):
        groups=defaultdict(list)
        for r in scored_results[null]:
            if r['domain']==domain and r['variant']=='raw' and r['condition']!='original':groups[r['condition']].append(r)
        for condition,records in groups.items():
            records.sort(key=lambda r:r['src']);labels=np.array([int(r['label']=='FAKE') for r in records])
            intervals[domain+'/'+condition]=auc_intervals({condition:np.array([r['score'] for r in records])},labels,config)[condition]
    refused=all(v[0]>.5 for v in intervals.values())
    candidates=[k for k in results if k!=null]
    chosen=max(candidates,key=lambda k:(min(results[k]['minimum_domain_ba'],results[k]['minimum_scene_ba']),
                      -results[k]['maximum_any_scene_drop'],k.startswith('fourier/'),k.endswith('/mean'))) if not refused else None
    for key,rule in rules.items():rule.save(OUTPUT/(key.replace('/','_')+'_rule.json'))
    write_json(OUTPUT/'selection_scores.json',scored_results)
    write_json(path,{'candidates':results,'development_chosen':chosen,'interpretation_refused':refused,
       'null_raw_processed_auc_ci95':intervals,'elapsed_s':perf_counter()-start,'mapping_s':mapping_s,
       'code_pins':code_pins(),'portable_mapped_single_batch_exact':True,'inventory_sha256':parent['inventory_sha256'],
       'parent_features_sha256':file_sha256(PARENT/'features.json'),'controls_sha256':file_sha256(controls_path),
       'selection_scores_sha256':file_sha256(OUTPUT/'selection_scores.json'),
       'rule_files':{k:file_sha256(OUTPUT/(k.replace('/','_')+'_rule.json')) for k in rules},
       'goal_achieved':False,'scope':'Frozen neural CLIP+fixedRFF+fitted conventional head on exposed development'})


if __name__=='__main__':main()
