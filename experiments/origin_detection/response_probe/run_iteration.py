"""Registered fixed-probe extraction and conventional source-risk evaluation."""

import argparse
from collections import defaultdict
import json
import random
from time import perf_counter

import cv2
import numpy as np

from palimpsest.detection.representations.response_clip import (
    ResponseFrozenClip, PROBE_NAMES, RESPONSE_NAMES, FEATURE_NAMES, SENSITIVITY_NAMES, response_feature)
from palimpsest.detection.representations.intermediate_clip import MID_NAMES, FEATURE_NAMES as PARENT_NAMES
from palimpsest.detection.algorithms.residual_statistics.features import resize256
from palimpsest.evaluation.pixel_features import extract_inventory, audit_variants, join_features
from palimpsest.evaluation.source_readout_campaign import fit_readouts
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR, MODELS_ROOT
from experiments.origin_detection.source_view_risk.run_iteration import parent_data, VARIANTS, Q60
from experiments.origin_detection.intermediate_encoder.run_iteration import code_pins as parent_pins
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import image_path, write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals

OUTPUT = WORK_DIR / 'robust_statistics/response_probe'
PARENT = WORK_DIR / 'robust_statistics/intermediate_encoder'
CONFIG = {'seed':20261006, 'bootstrap_repetitions':2000,
          'provisional_final_ba':.8, 'provisional_final_drop':.02}


def code_pins():
    paths = [REPO_ROOT/'src/palimpsest/detection/representations'/name
             for name in ('response_clip.py','intermediate_clip.py','frozen_clip.py')]
    paths += [REPO_ROOT/'src/palimpsest/evaluation'/name for name in
              ('source_readout_campaign.py','pixel_features.py','robust_views.py','source_training.py')]
    paths += [REPO_ROOT/'src/palimpsest/detection/algorithms/readouts/source_view_risk.py',
        REPO_ROOT/'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        REPO_ROOT/'tests/detection/test_response_clip.py']
    paths += [REPO_ROOT/'experiments/origin_detection/response_probe'/name
              for name in ('README.md','run_iteration.py','audit_runtime.py')]
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in paths}


def signed_parent(inventory, parent):
    receipt_path = PARENT/'features.json'
    receipt = json.loads(receipt_path.read_text())
    if (not receipt['passed'] or not receipt['final_parity']['passed']
            or receipt['csv_sha256'] != file_sha256(PARENT/'features.csv')
            or receipt['inventory_sha256'] != parent['inventory_sha256']
            or receipt['code_pins'] != parent_pins()):
        raise ValueError('Signed unprobed midpoint parent changed')
    rows = read_rows(PARENT/'features.csv')
    audit_variants(rows, inventory, PARENT_NAMES, ordinary_variants=VARIANTS[:3], selection_variant=Q60)
    return rows, file_sha256(receipt_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pilot', type=int, default=0)
    group.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.pilot < 0:
        parser.error('Pilot count must be positive')
    gate = json.loads((OUTPUT/'runtime_controls.json').read_text())
    software = json.loads((OUTPUT/'software_controls.json').read_text())
    if (not gate['passed'] or software['returncode'] != 0
            or any(file_sha256(REPO_ROOT/k) != v for k,v in gate['code_pins'].items())
            or any(file_sha256(REPO_ROOT/k) != v for k,v in software['test_pins'].items())):
        raise ValueError('Response known-input controls changed')
    _, inventory, parent, _ = parent_data()
    original, parent_sha = signed_parent(inventory, parent)
    cv2.setNumThreads(1)
    if args.pilot or args.prepare:
        output = OUTPUT.with_name('response_probe_pilot') if args.pilot else OUTPUT
        output.mkdir(parents=True, exist_ok=True)
        if (output/'features.csv').exists() or (output/'features.json').exists():
            raise FileExistsError('Preserve fixed response extraction')
        if args.pilot:
            random.Random(CONFIG['seed']).shuffle(inventory)
            inventory = inventory[:args.pilot]
        else:
            pilot = json.loads((OUTPUT.with_name('response_probe_pilot')/'features.json').read_text())
            if not pilot['passed'] or pilot['code_pins'] != code_pins() or pilot['parent_sha256'] != parent_sha:
                raise ValueError('Fixed response cost pilot changed')
        # Original target vectors are already signed; only new probe prefix runs.
        del original
        encoder = ResponseFrozenClip(MODELS_ROOT/'d3/ViT-L-14.pt')
        receipt = extract_inventory(inventory, output/'features.csv', names=PROBE_NAMES,
            extractor=encoder.extract, resolve_path=image_path, resize=resize256,
            ordinary_variants=VARIANTS[:3], selection_variant=Q60,
            jpeg_parameters=dict(zip(VARIANTS[1:], ((90,0),(70,2),(60,2)))))
        write_json(output/'features.json', {**receipt, 'passed':True, 'code_pins':code_pins(),
            'encoder':encoder.provenance, 'parent_sha256':parent_sha,
            'inventory_sha256':parent['inventory_sha256'],
            'runtime_sha256':file_sha256(OUTPUT/'runtime_controls.json'),
            'software_sha256':file_sha256(OUTPUT/'software_controls.json')})
        return
    destination = OUTPUT/'iteration.json'
    if destination.exists():
        raise FileExistsError('Preserve response fitting')
    receipt = json.loads((OUTPUT/'features.json').read_text())
    if (not receipt['passed'] or receipt['csv_sha256'] != file_sha256(OUTPUT/'features.csv')
            or receipt['inventory_sha256'] != parent['inventory_sha256']
            or receipt['code_pins'] != code_pins() or receipt['parent_sha256'] != parent_sha):
        raise ValueError('Response probe cache changed')
    probe = read_rows(OUTPUT/'features.csv')
    audit_variants(probe, inventory, PROBE_NAMES, ordinary_variants=VARIANTS[:3], selection_variant=Q60)
    rows = join_features(original, probe, PROBE_NAMES)
    del original, probe
    for row in rows:
        value = response_feature([float(row[n]) for n in MID_NAMES], [float(row[n]) for n in PROBE_NAMES])
        row.update(dict(zip(RESPONSE_NAMES, value.tolist())))
    audit_variants(rows, inventory, FEATURE_NAMES, ordinary_variants=VARIANTS[:3], selection_variant=Q60)
    start = perf_counter()
    candidates = [('sensitivity/source',SENSITIVITY_NAMES,.1,False),
        ('global_response/source',FEATURE_NAMES,.1,False),
        ('global_response/mean',FEATURE_NAMES,0,False),
        ('global_response/source_null',FEATURE_NAMES,.1,True)]
    results, rules, scores = fit_readouts(rows, candidates, fit_variants=VARIANTS[:2],
        manifest_sha=parent['inventory_sha256'], config=CONFIG, calibrate=class_threshold, score=score_rule)
    null_intervals = {}
    groups = defaultdict(list)
    for r in scores['global_response/source_null']:
        if r['variant'] == 'raw' and r['condition'] != 'original':
            groups[r['domain']+'/'+r['condition']].append(r)
    for name, records in groups.items():
        records.sort(key=lambda r:r['src'])
        null_intervals[name] = auc_intervals({name:np.array([r['score'] for r in records])},
            np.array([int(r['label']=='FAKE') for r in records]),CONFIG)[name]
    refused = all(v[0] > .5 for v in null_intervals.values())
    keys = [k for k in results if k != 'global_response/source_null']
    chosen = None if refused else max(keys, key=lambda k:(min(results[k]['minimum_domain_ba'],
        results[k]['minimum_scene_ba']), -results[k]['maximum_any_scene_drop']))
    for key, rule in rules.items():
        rule.save(OUTPUT/(key.replace('/','_')+'_rule.json'))
    write_json(OUTPUT/'selection_scores.json', scores)
    write_json(destination, {'candidates':results, 'development_chosen':chosen,
        'interpretation_refused':refused, 'null_raw_processed_auc_ci95':null_intervals,
        'elapsed_s':perf_counter()-start, 'code_pins':code_pins(), 'parent_sha256':parent_sha,
        'features_receipt_sha256':file_sha256(OUTPUT/'features.json'),
        'inventory_sha256':parent['inventory_sha256'],
        'selection_scores_sha256':file_sha256(OUTPUT/'selection_scores.json'),
        'rule_files':{k:file_sha256(OUTPUT/(k.replace('/','_')+'_rule.json')) for k in rules},
        'goal_achieved':False, 'scope':'Fixed frozen neural response,conventional head;exposed development'})


if __name__ == '__main__':
    main()
