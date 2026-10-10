"""Fixed midpoint plus final frozen CLIP; source-risk readout and signed parity."""

import argparse
from collections import defaultdict
import json
import random
from time import perf_counter

import cv2
import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.readouts.source_view_risk import fit_source_risk
from palimpsest.detection.algorithms.residual_statistics.features import resize256
from palimpsest.detection.representations.intermediate_clip import IntermediateFrozenClip, FEATURE_NAMES, MID_NAMES
from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES as GLOBAL_NAMES
from palimpsest.evaluation.features import feature_views
from palimpsest.evaluation.pixel_features import extract_inventory, audit_variants
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR, MODELS_ROOT
from experiments.origin_detection.source_view_risk.run_iteration import parent_data, VARIANTS, Q60
from experiments.origin_detection.semantic_kernel.run_iteration import inputs
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals

OUTPUT = WORK_DIR / 'robust_statistics/intermediate_encoder'
CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}


def code_pins():
    paths = [REPO_ROOT / 'src/palimpsest/detection/representations/intermediate_clip.py',
        REPO_ROOT / 'src/palimpsest/detection/representations/frozen_clip.py',
        REPO_ROOT / 'src/palimpsest/detection/algorithms/readouts/source_view_risk.py',
        REPO_ROOT / 'src/palimpsest/evaluation/source_training.py',
        REPO_ROOT / 'src/palimpsest/evaluation/pixel_features.py',
        REPO_ROOT / 'src/palimpsest/evaluation/robust_views.py',
        REPO_ROOT / 'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        REPO_ROOT / 'tests/detection/test_intermediate_clip.py']
    paths.extend(REPO_ROOT / 'experiments/origin_detection/intermediate_encoder' / n
                 for n in ('README.md', 'run_iteration.py', 'audit_runtime.py'))
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in paths}


def exact_final_parity(rows, previous):
    lookup = {(r['filename'], r['variant']): r for r in previous}
    if len(lookup) != len(previous):
        raise ValueError('Duplicated prior final cache')
    for row in rows:
        old = lookup[row['filename'], row['variant']]
        actual = np.array([float(row[n]) for n in GLOBAL_NAMES])
        expected = np.array([float(old[n]) for n in GLOBAL_NAMES])
        if not np.array_equal(actual, expected):
            raise ValueError('Midpoint extraction changed signed final features')
    return {'passed': True, 'vectors': len(rows), 'maximum_difference': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pilot', type=int, default=0)
    group.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.pilot < 0:
        parser.error('Pilot count must be positive')
    gate_path = OUTPUT / 'runtime_controls.json'
    gate = json.loads(gate_path.read_text())
    software_path = OUTPUT / 'software_controls.json'
    software = json.loads(software_path.read_text())
    if (not gate['passed'] or software['returncode'] != 0
            or any(file_sha256(REPO_ROOT/k) != v for k,v in gate['code_pins'].items())
            or any(file_sha256(REPO_ROOT/k) != v for k,v in software['test_pins'].items())):
        raise ValueError('Midpoint computation control changed')
    _, inventory, parent, _ = parent_data()
    previous, _ = inputs()
    cv2.setNumThreads(1)
    if args.pilot or args.prepare:
        output = OUTPUT.with_name('intermediate_encoder_pilot') if args.pilot else OUTPUT
        output.mkdir(parents=True, exist_ok=True)
        if (output / 'features.csv').exists() or (output / 'features.json').exists():
            raise FileExistsError('Preserve midpoint extraction')
        if args.pilot:
            random.Random(CONFIG['seed']).shuffle(inventory)
            inventory = inventory[:args.pilot]
        else:
            pilot = json.loads((OUTPUT.with_name('intermediate_encoder_pilot') / 'features.json').read_text())
            if not pilot['final_parity']['passed'] or pilot['code_pins'] != code_pins():
                raise ValueError('Midpoint pilot gate changed')
        encoder = IntermediateFrozenClip(MODELS_ROOT / 'd3/ViT-L-14.pt')
        receipt = extract_inventory(inventory, output / 'features.csv', names=FEATURE_NAMES,
            extractor=encoder.extract, resolve_path=image_path, resize=resize256,
            ordinary_variants=VARIANTS[:3], selection_variant=Q60,
            jpeg_parameters=dict(zip(VARIANTS[1:], ((90,0),(70,2),(60,2)))))
        rows = read_rows(output / 'features.csv')
        try:
            parity = exact_final_parity(rows, previous)
        except ValueError as error:
            write_json(output / 'features.json', {**receipt, 'passed': False, 'error': str(error),
                'code_pins': code_pins(), 'scope': 'Final parity refused;no further computation'})
            raise
        write_json(output / 'features.json', {**receipt, 'passed': True, 'final_parity': parity,
            'code_pins': code_pins(), 'encoder': encoder.provenance,
            'inventory_sha256': parent['inventory_sha256'], 'runtime_sha256': file_sha256(gate_path),
            'software_sha256': file_sha256(software_path)})
        return
    destination = OUTPUT / 'iteration.json'
    if destination.exists():
        raise FileExistsError('Preserve midpoint fitting')
    receipt_path = OUTPUT / 'features.json'
    receipt = json.loads(receipt_path.read_text())
    if (not receipt['passed'] or receipt['csv_sha256'] != file_sha256(OUTPUT / 'features.csv')
            or receipt['inventory_sha256'] != parent['inventory_sha256']
            or receipt['code_pins'] != code_pins()):
        raise ValueError('Midpoint cache or code changed')
    rows = read_rows(OUTPUT / 'features.csv')
    audit_variants(rows, inventory, FEATURE_NAMES, ordinary_variants=VARIANTS[:3], selection_variant=Q60)
    parity = exact_final_parity(rows, previous)
    results, rules, all_scores = {}, {}, {}
    start = perf_counter()
    candidates = [('mid/source', MID_NAMES, .1, False), ('joint/source', FEATURE_NAMES, .1, False),
                  ('joint/mean', FEATURE_NAMES, 0, False), ('joint/source_null', FEATURE_NAMES, .1, True)]
    with threadpool_limits(limits=1):
        for key, names, temperature, shuffled in candidates:
            x,y,w,s = weighted_source_arrays(rows, names, VARIANTS[:2],
                expected_source_counts={'rr':540, 'chimera':720}, seed=CONFIG['seed'], shuffled=shuffled)
            rule, diagnostic = fit_source_risk(x,y,w,s, feature_names=names, temperature=temperature,
                ridge=.01, scale_floor=.001, maximum_iterations=500, gradient_tolerance=1e-5,
                manifest_sha=parent['inventory_sha256'])
            views = {k+'/'+v:view for v in VARIANTS[:2] for processed in (False,True)
                     for k,view in feature_views(rows,names,'threshold',processed=processed,variant=v).items()}
            if len(views) != 24:
                raise ValueError('Midpoint calibration views changed')
            rule, diagnostic['calibration'] = class_threshold(rule, views)
            scored, result = score_rule(rows, names, rule, CONFIG)
            result.update({'fit_diagnostics':diagnostic, 'threshold':rule.threshold})
            results[key], rules[key], all_scores[key] = result, rule, scored
            print(json.dumps({'candidate':key, 'minimum_ba':result['minimum_domain_ba'],
                'minimum_scene_ba':result['minimum_scene_ba'], 'maxscene_drop':result['maximum_any_scene_drop']}), flush=True)
    null_intervals = {}
    for domain in ('rr', 'chimera'):
        groups = defaultdict(list)
        for r in all_scores['joint/source_null']:
            if r['domain'] == domain and r['variant'] == 'raw' and r['condition'] != 'original':
                groups[r['condition']].append(r)
        for condition, records in groups.items():
            records.sort(key=lambda r:r['src'])
            labels = np.array([int(r['label']=='FAKE') for r in records])
            null_intervals[domain+'/'+condition] = auc_intervals(
                {condition:np.array([r['score'] for r in records])}, labels, CONFIG)[condition]
    refused = all(v[0] > .5 for v in null_intervals.values())
    keys = [k for k in results if k != 'joint/source_null']
    chosen = None if refused else max(keys, key=lambda k:(min(results[k]['minimum_domain_ba'],
        results[k]['minimum_scene_ba']), -results[k]['maximum_any_scene_drop'], k=='mid/source'))
    for key, rule in rules.items():
        rule.save(OUTPUT / (key.replace('/', '_')+'_rule.json'))
    write_json(OUTPUT / 'selection_scores.json', all_scores)
    write_json(destination, {'candidates':results, 'development_chosen':chosen, 'interpretation_refused':refused,
        'null_raw_processed_auc_ci95':null_intervals, 'elapsed_s':perf_counter()-start,
        'final_parity':parity, 'code_pins':code_pins(), 'features_receipt_sha256':file_sha256(receipt_path),
        'inventory_sha256':parent['inventory_sha256'],
        'selection_scores_sha256':file_sha256(OUTPUT / 'selection_scores.json'),
        'rule_files':{k:file_sha256(OUTPUT/(k.replace('/', '_')+'_rule.json')) for k in rules},
        'goal_achieved':False, 'scope':'Frozen neural midpoint+semantic descriptors,conventional head;exposed development'})


if __name__ == '__main__':
    main()
