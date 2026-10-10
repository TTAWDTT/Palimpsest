"""Fixed frozen CLIP/DINO complementarity, same-source conventional heads."""

from collections import defaultdict
import json
from time import perf_counter

import numpy as np

from palimpsest.detection.representations.frozen_clip import FEATURE_NAMES as CLIP_NAMES
from palimpsest.detection.representations.frozen_dinov2_small import FEATURE_NAMES as DINO_NAMES
from palimpsest.evaluation.pixel_features import join_features, audit_variants
from palimpsest.evaluation.source_readout_campaign import fit_readouts
from palimpsest.io.tables import read_rows
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.semantic_kernel.run_iteration import inputs
from experiments.origin_detection.source_view_risk.run_iteration import parent_data, VARIANTS, Q60
from experiments.origin_detection.compact_frozen_encoder.run_iteration import code_pins as dino_pins, score_rule
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals

OUTPUT = WORK_DIR/'robust_statistics/complementary_encoders'
DINO = WORK_DIR/'robust_statistics/compact_frozen_encoder'
CONFIG = {'seed':20261006,'bootstrap_repetitions':2000,'provisional_final_ba':.8,'provisional_final_drop':.02}
FEATURE_NAMES = CLIP_NAMES+DINO_NAMES


def code_pins():
    paths = [REPO_ROOT/'src/palimpsest/detection/representations'/name for name in
             ('feature_pair.py','frozen_clip.py','frozen_dinov2_small.py')]
    paths += [REPO_ROOT/'src/palimpsest/detection/algorithms/readouts/source_view_risk.py',
        REPO_ROOT/'src/palimpsest/evaluation/source_readout_campaign.py',
        REPO_ROOT/'src/palimpsest/evaluation/pixel_features.py',
        REPO_ROOT/'src/palimpsest/evaluation/source_training.py',
        REPO_ROOT/'src/palimpsest/evaluation/robust_views.py',
        REPO_ROOT/'tests/detection/test_feature_pair.py']
    paths += [REPO_ROOT/'experiments/origin_detection/complementary_encoders'/n
              for n in ('README.md','run_iteration.py')]
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in paths}


def main():
    path = OUTPUT/'iteration.json'
    if path.exists():
        raise FileExistsError('Preserve frozen complementarity campaign')
    controls = json.loads((OUTPUT/'software_controls.json').read_text())
    if controls['returncode'] != 0 or any(file_sha256(REPO_ROOT/k) != v for k,v in controls['test_pins'].items()):
        raise ValueError('Frozen pair known-input control changed')
    clip, parent = inputs()
    old,inventory,_,_ = parent_data();del old
    receipt = json.loads((DINO/'features.json').read_text())
    if (receipt['csv_sha256'] != file_sha256(DINO/'features.csv')
            or receipt['code_pins'] != dino_pins()
            or receipt['inventory_sha256'] != parent['inventory_sha256']):
        raise ValueError('Frozen DINO cache changed')
    dino = read_rows(DINO/'features.csv')
    audit_variants(dino,inventory,DINO_NAMES,ordinary_variants=VARIANTS[:3],selection_variant=Q60)
    rows = join_features(clip,dino,DINO_NAMES);del clip,dino
    audit_variants(rows,inventory,FEATURE_NAMES,ordinary_variants=VARIANTS[:3],selection_variant=Q60)
    start = perf_counter()
    candidates = [('weak_dino/source',DINO_NAMES,.1,False),('joint/source',FEATURE_NAMES,.1,False),
        ('joint/mean',FEATURE_NAMES,0,False),('joint/source_null',FEATURE_NAMES,.1,True)]
    results,rules,scores = fit_readouts(rows,candidates,fit_variants=VARIANTS[:2],
        manifest_sha=parent['inventory_sha256'],config=CONFIG,calibrate=class_threshold,score=score_rule)
    groups = defaultdict(list)
    for r in scores['joint/source_null']:
        if r['variant'] == 'raw' and r['condition'] != 'original':
            groups[r['domain']+'/'+r['condition']].append(r)
    null_intervals = {}
    for key,records in groups.items():
        records.sort(key=lambda r:r['src'])
        null_intervals[key] = auc_intervals({key:np.array([r['score'] for r in records])},
            np.array([int(r['label']=='FAKE') for r in records]),CONFIG)[key]
    refused = all(v[0] > .5 for v in null_intervals.values())
    keys = [k for k in results if k != 'joint/source_null']
    chosen = None if refused else max(keys,key=lambda k:(min(results[k]['minimum_domain_ba'],
        results[k]['minimum_scene_ba']),-results[k]['maximum_any_scene_drop']))
    for key,rule in rules.items():
        rule.save(OUTPUT/(key.replace('/','_')+'_rule.json'))
    write_json(OUTPUT/'selection_scores.json',scores)
    write_json(path,{'candidates':results,'development_chosen':chosen,'interpretation_refused':refused,
        'null_raw_processed_auc_ci95':null_intervals,'code_pins':code_pins(),
        'elapsed_s':perf_counter()-start,'inventory_sha256':parent['inventory_sha256'],
        'dino_features_sha256':receipt['csv_sha256'],
        'selection_scores_sha256':file_sha256(OUTPUT/'selection_scores.json'),
        'rule_files':{k:file_sha256(OUTPUT/(k.replace('/','_')+'_rule.json')) for k in rules},
        'goal_achieved':False,'scope':'Frozen neural feature complementarity,conventional readout;exposed development'})


if __name__ == '__main__':
    main()
