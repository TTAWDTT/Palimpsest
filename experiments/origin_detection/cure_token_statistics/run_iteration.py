"""Five conventional readouts of numeric token statistics;frozen encoder."""

from collections import ChainMap
import json
from time import perf_counter

import numpy as np

from palimpsest.detection.representations.frozen_cure_tokens import FEATURE_NAMES,CLIPPED_NAMES,SPREAD_NAMES,MIXED_NAMES
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.source_readout_campaign import fit_readouts
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import WORK_DIR
from experiments.origin_detection.cure_token_statistics.prepare_features import OUTPUT,code_pins,reference_data,audit_cache
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS
from experiments.origin_detection.semantic_gaussian.run_iteration import null_intervals
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.phase_statistics.run_iteration import write_json

CONFIG={'seed':20261006,'bootstrap_repetitions':2000,'provisional_final_ba':.8,'provisional_final_drop':.02}


def main():
    if (OUTPUT/'iteration.json').exists():raise FileExistsError('Preserve token readout evaluation')
    receipt=json.loads((OUTPUT/'features.json').read_text())
    if (receipt['code_pins']!=code_pins() or not receipt['all_original_exact'] or receipt['records']!=15120
            or receipt['feature_names']!=list(FEATURE_NAMES)
            or file_sha256(OUTPUT/'vectors.npy')!=receipt['vectors_sha256']
            or file_sha256(OUTPUT/'metadata.csv')!=receipt['metadata_sha256']):
        raise ValueError('Signed token cache changed')
    inventory,_,_,_,parent=reference_data()
    if receipt['inventory_sha256']!=parent['inventory_sha256']:raise ValueError('Token inventory differs')
    matrix=np.load(OUTPUT/'vectors.npy',mmap_mode='r')
    if matrix.dtype!=np.dtype('<f8'):raise ValueError('Token storage dtype changed')
    rows=audit_cache(read_rows(OUTPUT/'metadata.csv'),matrix,inventory)
    assignment,control=balanced_source_null([r for r in rows if r['role']=='fit'],seed=CONFIG['seed'])
    prior=json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if control['assignment_sha256']!=prior['assignment_sha256']:raise ValueError('Token source null changed')
    start=perf_counter()
    candidates=[('clipped/source',CLIPPED_NAMES,.1,False),('spread/source',SPREAD_NAMES,.1,False),
                ('mixed/source',MIXED_NAMES,.1,False),('mixed/mean',MIXED_NAMES,0,False)]
    results,rules,scores=fit_readouts(rows,candidates,fit_variants=VARIANTS[:2],manifest_sha=parent['inventory_sha256'],
        config=CONFIG,calibrate=class_threshold,score=score_rule)
    sham=[ChainMap({'label':'FAKE' if assignment[r['domain'],r['src']] else 'REAL'},r)
          if r['role']=='fit' else r for r in rows]
    extra,rr,ss=fit_readouts(sham,[('mixed/source_null',MIXED_NAMES,.1,False)],fit_variants=VARIANTS[:2],
        manifest_sha=parent['inventory_sha256'],config=CONFIG,calibrate=class_threshold,score=score_rule)
    results.update(extra);rules.update(rr);scores.update(ss)
    artifacts={}
    for key,rule in rules.items():
        f=OUTPUT/(key.replace('/','_')+'_rule.json');rule.save(f);artifacts[key]=file_sha256(f)
    write_json(OUTPUT/'selection_scores.json',scores)
    write_json(OUTPUT/'iteration.json',{'candidates':results,'code_pins':code_pins(),
        'features_receipt_sha256':file_sha256(OUTPUT/'features.json'),'elapsed_s':perf_counter()-start,
        'inventory_sha256':parent['inventory_sha256'],'balanced_assignment_sha256':control['assignment_sha256'],
        'null_raw_processed_auc_ci95':null_intervals(scores['mixed/source_null']),
        'selection_scores_sha256':file_sha256(OUTPUT/'selection_scores.json'),'rule_files':artifacts,
        'goal_achieved':False,'scope':'Median anchored winsorized token statistics;single unchanged encoder;exposed development'})


if __name__=='__main__':main()
