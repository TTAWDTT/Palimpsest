"""One fixed smaller semantic/self-supervised pair, signed cached-head comparison."""

import json
from time import perf_counter

from palimpsest.detection.representations.frozen_clip_base import FEATURE_NAMES as BASE_NAMES
from palimpsest.detection.representations.frozen_dinov2_small import FEATURE_NAMES as DINO_NAMES
from palimpsest.evaluation.cached_campaign import evaluate_cached_heads
from palimpsest.evaluation.pixel_features import join_features,audit_variants
from palimpsest.io.tables import read_rows
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT,WORK_DIR
from experiments.origin_detection.source_view_risk.run_iteration import parent_data,VARIANTS,Q60
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.phase_statistics.run_iteration import write_json
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.robust_statistics.rr.evaluate_features import auc_intervals

OUTPUT = WORK_DIR/'robust_statistics/compact_complement'
CONFIG = {'seed':20261006,'bootstrap_repetitions':2000,'provisional_final_ba':.8,'provisional_final_drop':.02}
FEATURE_NAMES = BASE_NAMES+DINO_NAMES


def code_pins():
    paths = [REPO_ROOT/'src/palimpsest/evaluation'/n for n in
        ('cached_campaign.py','source_readout_campaign.py','source_training.py','pixel_features.py','robust_views.py')]
    paths += [REPO_ROOT/'src/palimpsest/detection/algorithms/source_view_risk.py',
        REPO_ROOT/'src/palimpsest/detection/representations/feature_pair.py',
        REPO_ROOT/'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        REPO_ROOT/'tests/detection/test_feature_pair.py']
    paths += [REPO_ROOT/'experiments/origin_detection/compact_complement'/n for n in ('README.md','run_iteration.py')]
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in paths}


def main():
    path = OUTPUT/'iteration.json'
    if path.exists():
        raise FileExistsError('Preserve compact complement campaign')
    controls = json.loads((OUTPUT/'software_controls.json').read_text())
    if controls['returncode']!=0 or any(file_sha256(REPO_ROOT/k)!=v for k,v in controls['test_pins'].items()):
        raise ValueError('Known composition control changed')
    old,inventory,parent,_ = parent_data();del old
    parts, receipts = [], {}
    for slug,names in [('compact_semantic_encoder',BASE_NAMES),('compact_frozen_encoder',DINO_NAMES)]:
        directory = WORK_DIR/'robust_statistics'/slug
        receipt = json.loads((directory/'features.json').read_text())
        if (receipt['inventory_sha256']!=parent['inventory_sha256']
                or receipt['csv_sha256']!=file_sha256(directory/'features.csv')
                or any(file_sha256(REPO_ROOT/k)!=v for k,v in receipt['code_pins'].items())):
            raise ValueError('Compact component cache changed')
        rows = read_rows(directory/'features.csv')
        audit_variants(rows,inventory,names,ordinary_variants=VARIANTS[:3],selection_variant=Q60)
        parts.append(rows);receipts[slug]=file_sha256(directory/'features.json')
    rows = join_features(*parts,DINO_NAMES);del parts
    audit_variants(rows,inventory,FEATURE_NAMES,ordinary_variants=VARIANTS[:3],selection_variant=Q60)
    start = perf_counter()
    results,rules,scores,intervals,chosen,refused = evaluate_cached_heads(rows,
        [('joint/source',FEATURE_NAMES,.1,False),('joint/mean',FEATURE_NAMES,0,False),
         ('joint/source_null',FEATURE_NAMES,.1,True)],null_key='joint/source_null',fit_variants=VARIANTS[:2],
        manifest_sha=parent['inventory_sha256'],config=CONFIG,calibrate=class_threshold,
        score=score_rule,interval_function=auc_intervals)
    for key,rule in rules.items():
        rule.save(OUTPUT/(key.replace('/','_')+'_rule.json'))
    write_json(OUTPUT/'selection_scores.json',scores)
    write_json(path,{'candidates':results,'development_chosen':chosen,'interpretation_refused':refused,
        'null_raw_processed_auc_ci95':intervals,'code_pins':code_pins(),'parent_receipts':receipts,
        'elapsed_s':perf_counter()-start,'inventory_sha256':parent['inventory_sha256'],
        'selection_scores_sha256':file_sha256(OUTPUT/'selection_scores.json'),
        'rule_files':{k:file_sha256(OUTPUT/(k.replace('/','_')+'_rule.json')) for k in rules},
        'goal_achieved':False,'scope':'Compact frozen neural complementarity,conventional heads;exposed development'})


if __name__ == '__main__':
    main()
