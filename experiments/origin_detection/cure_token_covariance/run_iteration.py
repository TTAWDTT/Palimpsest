"""Five preregistered conventional readouts after signed covariance extraction."""

from collections import ChainMap
import json

import numpy as np

from palimpsest.detection.representations.frozen_cure_covariance import FEATURE_NAMES,COVARIANCE_NAMES
from palimpsest.detection.representations.frozen_cure_tokens import MIXED_NAMES
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.source_readout_campaign import fit_readouts
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import WORK_DIR
from experiments.origin_detection.cure_token_covariance.prepare_features import OUTPUT,code_pins,parent_data,audit_cache
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS
from experiments.origin_detection.token_consistency_crossfit.run_iteration import CONFIG,class_threshold,score_rule,null_intervals,write_json

COMBINED_NAMES=MIXED_NAMES+COVARIANCE_NAMES


def main():
    if (OUTPUT/'iteration.json').exists():raise FileExistsError('Preserve covariance readouts')
    receipt=json.loads((OUTPUT/'features.json').read_text())
    if (receipt['code_pins']!=code_pins() or receipt['records']!=15120 or not receipt['all_parent_prefix_exact']
            or file_sha256(OUTPUT/'vectors.npy')!=receipt['vectors_sha256']
            or file_sha256(OUTPUT/'metadata.csv')!=receipt['metadata_sha256']):
        raise ValueError('Signed covariance cache changed')
    inventory,_,_,parent=parent_data()
    rows=audit_cache(read_rows(OUTPUT/'metadata.csv'),np.load(OUTPUT/'vectors.npy',mmap_mode='r'),inventory)
    assignment,control=balanced_source_null([r for r in rows if r['role']=='fit'],seed=CONFIG['seed'])
    prior=json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if control['assignment_sha256']!=prior['assignment_sha256']:raise ValueError('Covariance null mapping differs')
    candidates=[('covariance/source',COVARIANCE_NAMES,.1,False),('mixed/source',COMBINED_NAMES,.1,False),
                ('mixed/mean',COMBINED_NAMES,0,False),('full/source',FEATURE_NAMES,.1,False)]
    results,rules,scores=fit_readouts(rows,candidates,fit_variants=VARIANTS[:2],manifest_sha=parent['inventory_sha256'],
        config=CONFIG,calibrate=class_threshold,score=score_rule)
    sham=[ChainMap({'label':'FAKE' if assignment[r['domain'],r['src']] else 'REAL'},r)
          if r['role']=='fit' else r for r in rows]
    extra,more_rules,more_scores=fit_readouts(sham,[('mixed/source_null',COMBINED_NAMES,.1,False)],
        fit_variants=VARIANTS[:2],manifest_sha=parent['inventory_sha256'],config=CONFIG,
        calibrate=class_threshold,score=score_rule)
    results.update(extra);rules.update(more_rules);scores.update(more_scores);artifacts={}
    for key,rule in rules.items():
        path=OUTPUT/(key.replace('/','_')+'_rule.json');rule.save(path);artifacts[key]=file_sha256(path)
    write_json(OUTPUT/'selection_scores.json',scores)
    write_json(OUTPUT/'iteration.json',{'candidates':results,'code_pins':code_pins(),
        'features_receipt_sha256':file_sha256(OUTPUT/'features.json'),'inventory_sha256':parent['inventory_sha256'],
        'balanced_assignment_sha256':control['assignment_sha256'],'rule_files':artifacts,
        'selection_scores_sha256':file_sha256(OUTPUT/'selection_scores.json'),
        'null_raw_processed_auc_ci95':null_intervals(scores['mixed/source_null']),
        'goal_achieved':False,'scope':'Fixed projection/token covariance on exposed development;no independent validation'})


if __name__=='__main__':main()
