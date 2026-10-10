"""Paired slow/PCA/protected subspaces with collapsed native linear inference."""

import argparse
from collections import defaultdict
import json
from time import perf_counter
import tomllib

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.slow_subspace import fit_slow_subspace,collapse_readout
from palimpsest.detection.algorithms.readouts.source_view_risk import fit_source_risk
from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES
from palimpsest.evaluation.features import feature_views
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT,WORK_DIR
from experiments.origin_detection.semantic_kernel.run_iteration import inputs
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals
from experiments.origin_detection.phase_statistics.run_iteration import write_json

CONFIG=REPO_ROOT/'configs/evaluation/slow_subspace.toml'
OUTPUT=WORK_DIR/'robust_statistics/slow_subspace'


def code_pins():
    files=[CONFIG,REPO_ROOT/'src/palimpsest/detection/algorithms/slow_subspace.py',
          REPO_ROOT/'src/palimpsest/detection/algorithms/paired_stability.py', REPO_ROOT/'src/palimpsest/detection/algorithms/readouts/stable_rule.py',
          REPO_ROOT/'src/palimpsest/detection/algorithms/readouts/source_view_risk.py',
          REPO_ROOT/'src/palimpsest/evaluation/source_training.py',
          REPO_ROOT/'src/palimpsest/evaluation/robust_views.py',
          REPO_ROOT/'experiments/origin_detection/semantic_kernel/run_iteration.py',
          REPO_ROOT/'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
          REPO_ROOT/'experiments/origin_detection/threshold_calibration/fit_threshold.py',
          REPO_ROOT/'tests/detection/test_slow_subspace.py']
    files.extend(REPO_ROOT/'experiments/origin_detection/slow_subspace'/n for n in ('README.md','run_iteration.py'))
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in files}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    destination=OUTPUT/('pilot.json' if args.pilot else 'iteration.json')
    if destination.exists():raise FileExistsError('Preserve slow-subspace evidence')
    config=tomllib.loads(CONFIG.read_text());control_path=OUTPUT/'controls.json'
    control=json.loads(control_path.read_text())
    if control['returncode']!=0 or any(file_sha256(REPO_ROOT/k)!=v for k,v in control['test_pins'].items()):
        raise ValueError('Subspace controls changed/failed')
    rows,parent=inputs();start=perf_counter()
    names=tuple(f'paired_subspace/{i}' for i in range(config['dimension']))
    results={};scored_all={};rules={};maps={}
    with threadpool_limits(limits=1):
        matrix=np.array([[float(r[n]) for n in FEATURE_NAMES] for r in rows])
        selected=[r for r in rows if r['role']=='selection']
        native=np.array([[float(r[n]) for n in FEATURE_NAMES] for r in selected])
        candidates=[(m,o,False) for m in config['representations'] for o in config['objectives']]
        candidates.append(('protected_slow','source',True))
        if args.pilot:candidates=[('slow','mean',False)]
        for mode,objective,shuffled in candidates:
            key=mode+'/'+objective+('_source_shuffled_null' if shuffled else '')
            x,y,weights,sources=weighted_source_arrays(rows,FEATURE_NAMES,VARIANTS[:2],
                    expected_source_counts={'rr':540,'chimera':720},seed=config['seed'],shuffled=shuffled)
            mapping,diagnostics=fit_slow_subspace(x,y,weights,sources,dimension=config['dimension'],mode=mode,
                                                eigen_floor=config['eigen_floor'])
            fit=mapping.transform(x)
            head,fit_diagnostics=fit_source_risk(fit,y,weights,sources,feature_names=names,
                 temperature=config['source_temperature'] if objective=='source' else 0,ridge=config['ridge'],
                 scale_floor=config['scale_floor'],maximum_iterations=config['maximum_iterations'],
                 gradient_tolerance=config['gradient_tolerance'],manifest_sha=parent['inventory_sha256'])
            rule=collapse_readout(mapping,head,feature_names=FEATURE_NAMES)
            fit_difference=float(np.max(np.abs(rule.score(x)-head.score(fit))))
            if fit_difference>1e-10:raise ValueError('Collapsed fit scores differ')
            if args.pilot:
                write_json(destination,{'elapsed_s':perf_counter()-start,'code_pins':code_pins(),
                          'map_diagnostics':diagnostics,'fit_diagnostics':fit_diagnostics,
                          'maximum_collapse_difference':fit_difference,'scope':'Fit-only costpilot;no threshold/selection scores'})
                print(json.dumps({'pilot_s':perf_counter()-start}),flush=True);return
            views={k+'/'+v:view for v in VARIANTS[:2] for processed in (False,True)
                   for k,view in feature_views(rows,FEATURE_NAMES,'threshold',processed=processed,variant=v).items()}
            if len(views)!=24:raise ValueError('Subspace threshold view count changed')
            rule,fit_diagnostics['calibration']=class_threshold(rule,views)
            mapped_scores=head.score(mapping.transform(native))
            native_scores=rule.score(native)
            score_difference=float(np.max(np.abs(native_scores-mapped_scores)))
            if (score_difference>1e-10 or not np.array_equal(native_scores>rule.threshold,mapped_scores>rule.threshold)):
                raise ValueError('Collapsed selection scores/decisions changed')
            scored,result=score_rule(rows,FEATURE_NAMES,rule,config)
            result.update({'map_diagnostics':diagnostics,'fit_diagnostics':fit_diagnostics,'threshold':rule.threshold,
                          'maximum_collapse_difference':max(fit_difference,score_difference),
                          'collapse_selection_decisions_exact':True})
            maps[key]=mapping;rules[key]=rule;results[key]=result;scored_all[key]=scored
            print(json.dumps({'candidate':key,'minimum_ba':result['minimum_domain_ba'],
                  'minimum_scene_ba':result['minimum_scene_ba'],'maxscene_drop':result['maximum_any_scene_drop'],
                  'class_mean_energy_kept':diagnostics['retained_class_mean_energy_fraction']}),flush=True)
    null='protected_slow/source_source_shuffled_null';intervals={}
    for domain in ('rr','chimera'):
        groups=defaultdict(list)
        for r in scored_all[null]:
            if r['domain']==domain and r['variant']=='raw' and r['condition']!='original':groups[r['condition']].append(r)
        for condition,records in groups.items():
            records.sort(key=lambda r:r['src']);labels=np.array([int(r['label']=='FAKE') for r in records])
            intervals[domain+'/'+condition]=auc_intervals({condition:np.array([r['score'] for r in records])},labels,config)[condition]
    refused=all(v[0]>.5 for v in intervals.values())
    candidates=[k for k in results if k!=null]
    chosen=max(candidates,key=lambda k:(min(results[k]['minimum_domain_ba'],results[k]['minimum_scene_ba']),
                     -results[k]['maximum_any_scene_drop'],k.startswith('pca/'),k.endswith('/mean'))) if not refused else None
    for key,rule in rules.items():
        stem=key.replace('/','_');rule.save(OUTPUT/(stem+'_rule.json'))
        mapping=maps[key];np.savez(OUTPUT/(stem+'_map.npz'),center=mapping.center,basis=mapping.basis)
    write_json(OUTPUT/'selection_scores.json',scored_all)
    write_json(destination,{'candidates':results,'development_chosen':chosen,'interpretation_refused':refused,
             'null_raw_processed_auc_ci95':intervals,'elapsed_s':perf_counter()-start,'code_pins':code_pins(),
             'inventory_sha256':parent['inventory_sha256'],'controls_sha256':file_sha256(control_path),
             'selection_scores_sha256':file_sha256(OUTPUT/'selection_scores.json'),
             'rule_files':{k:file_sha256(OUTPUT/(k.replace('/','_')+'_rule.json')) for k in rules},
             'map_files':{k:file_sha256(OUTPUT/(k.replace('/','_')+'_map.npz')) for k in maps},
             'goal_achieved':False,'scope':'Frozen neural CLIP plus conventional paired subspace/head;exposed development'})


if __name__=='__main__':main()
