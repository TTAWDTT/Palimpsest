"""Explicit finite-view maximum hinge versus mean hinge; signed frozen features."""

import argparse
from collections import defaultdict
import json
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.readouts.source_hinge import fit_source_hinge, MarginFitRefused
from palimpsest.detection.algorithms.readouts.stable_rule import StableRule
from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.evaluation.features import feature_views
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.semantic_kernel.run_iteration import inputs
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals

OUTPUT = WORK_DIR/'robust_statistics/source_hinge'
CONFIG = {'seed':20261006, 'bootstrap_repetitions':2000,
          'provisional_final_ba':.8, 'provisional_final_drop':.02}


def code_pins():
    paths = [REPO_ROOT/'src/palimpsest/detection/algorithms'/name
             for name in ('source_hinge.py','paired_stability.py')]
    paths += [REPO_ROOT/'src/palimpsest/evaluation'/name
              for name in ('source_training.py','features.py','robust_views.py')]
    paths += [REPO_ROOT/'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        REPO_ROOT/'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        REPO_ROOT/'tests/detection/test_source_hinge.py']
    paths += [REPO_ROOT/'experiments/origin_detection/source_hinge'/name
              for name in ('README.md','run_iteration.py')]
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    destination = OUTPUT/('pilot.json' if args.pilot else 'iteration.json')
    if destination.exists():
        raise FileExistsError('Preserve finite hinge campaign')
    controls = json.loads((OUTPUT/'software_controls.json').read_text())
    if controls['returncode'] != 0 or any(file_sha256(REPO_ROOT/k) != v for k,v in controls['test_pins'].items()):
        raise ValueError('Hinge known-input controls failed or changed')
    rows, parent = inputs()
    start = perf_counter()
    results, rules, scores = {}, {}, {}
    candidates = [('max/source',True,False)] if args.pilot else [
        ('max/source',True,False),('mean/source',False,False),('max/source_null',True,True)]
    with threadpool_limits(limits=1):
        for key, maximum, shuffled in candidates:
            if key == 'max/source' and not args.pilot:
                pilot = json.loads((OUTPUT/'pilot.json').read_text())
                if (not pilot['passed'] or pilot['code_pins'] != code_pins()
                        or pilot['inventory_sha256'] != parent['inventory_sha256']
                        or pilot['rule_sha256'] != file_sha256(OUTPUT/'pilot_rule.json')):
                    raise ValueError('Hinge cost pilot changed')
                rule, diagnostic = StableRule.load(OUTPUT/'pilot_rule.json'), pilot['fit_diagnostics']
            else:
                x,y,w,s = weighted_source_arrays(rows,FEATURE_NAMES,VARIANTS[:2],
                    expected_source_counts={'rr':540,'chimera':720},seed=CONFIG['seed'],shuffled=shuffled)
                try:
                    rule, diagnostic = fit_source_hinge(x,y,w,s,feature_names=FEATURE_NAMES,
                        maximum_source=maximum,penalty=.001,scale_floor=.001,time_limit=120,
                        manifest_sha=parent['inventory_sha256'])
                except MarginFitRefused as error:
                    write_json(destination,{'passed':False,'candidate':key,'diagnostic':error.diagnostic,
                        'code_pins':code_pins(),'inventory_sha256':parent['inventory_sha256'],
                        'goal_achieved':False,'scope':'Refused solver execution,not detector failure evidence'})
                    raise
            if args.pilot:
                rule.save(OUTPUT/'pilot_rule.json')
                write_json(destination,{'passed':True,'fit_diagnostics':diagnostic,'code_pins':code_pins(),
                    'inventory_sha256':parent['inventory_sha256'],'rule_sha256':file_sha256(OUTPUT/'pilot_rule.json'),
                    'elapsed_s':perf_counter()-start,'scope':'Fit-only cost pilot,no threshold or selection scores'})
                print(json.dumps({'pilot_passed':True,'elapsed_s':perf_counter()-start}),flush=True)
                return
            views = {k+'/'+v:view for v in VARIANTS[:2] for processed in (False,True)
                for k,view in feature_views(rows,FEATURE_NAMES,'threshold',processed=processed,variant=v).items()}
            if len(views) != 24:
                raise ValueError('Hinge threshold views changed')
            rule, calibration = class_threshold(rule,views)
            scored, result = score_rule(rows,FEATURE_NAMES,rule,CONFIG)
            result.update({'fit_diagnostics':diagnostic,'calibration':calibration,'threshold':rule.threshold})
            results[key],rules[key],scores[key] = result,rule,scored
            print(json.dumps({'candidate':key,'minimum_ba':result['minimum_domain_ba'],
                'minimum_scene_ba':result['minimum_scene_ba'],'maximum_scene_drop':result['maximum_any_scene_drop']}),flush=True)
    groups = defaultdict(list)
    for row in scores['max/source_null']:
        if row['variant'] == 'raw' and row['condition'] != 'original':
            groups[row['domain']+'/'+row['condition']].append(row)
    null_intervals = {}
    for key,records in groups.items():
        records.sort(key=lambda r:r['src'])
        null_intervals[key] = auc_intervals({key:np.array([r['score'] for r in records])},
            np.array([int(r['label']=='FAKE') for r in records]),CONFIG)[key]
    refused = all(v[0] > .5 for v in null_intervals.values())
    keys = ['max/source','mean/source']
    chosen = None if refused else max(keys,key=lambda k:(min(results[k]['minimum_domain_ba'],
        results[k]['minimum_scene_ba']),-results[k]['maximum_any_scene_drop']))
    for key,rule in rules.items():
        rule.save(OUTPUT/(key.replace('/','_')+'_rule.json'))
    write_json(OUTPUT/'selection_scores.json',scores)
    write_json(destination,{'candidates':results,'development_chosen':chosen,'interpretation_refused':refused,
        'null_raw_processed_auc_ci95':null_intervals,'elapsed_s':perf_counter()-start,
        'code_pins':code_pins(),'inventory_sha256':parent['inventory_sha256'],
        'selection_scores_sha256':file_sha256(OUTPUT/'selection_scores.json'),
        'rule_files':{k:file_sha256(OUTPUT/(k.replace('/','_')+'_rule.json')) for k in rules},
        'goal_achieved':False,'scope':'Finite source maximum hinge,conventional readout;exposed development'})


if __name__ == '__main__':
    main()
