"""Four scalar density controls on immutable CuRe36 readouts."""

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.conditional_score import fit_conditional_score, radial_coordinate, ConditionalScoreRule
from palimpsest.detection.algorithms.paired_stability import StableRule
from palimpsest.detection.representations.frozen_cure import FULL_NAMES
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.evaluation.features import feature_views
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.origin_detection.semantic_gaussian.run_iteration import cure_inputs, null_intervals
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS, audit_receipt
from experiments.origin_detection.cure_readout.prepare_features import OUTPUT as CURE
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.phase_statistics.run_iteration import write_json

OUTPUT = WORK_DIR/'robust_statistics/radial_score_calibration'
CONFIG = {'seed': 20261006, 'bootstrap_repetitions': 2000,
          'provisional_final_ba': .8, 'provisional_final_drop': .02}


def code_pins():
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent/'README.md']
    paths += [REPO_ROOT/p for p in (
        'src/palimpsest/detection/algorithms/conditional_score.py',
        'src/palimpsest/detection/algorithms/paired_stability.py',
        'src/palimpsest/evaluation/source_training.py', 'src/palimpsest/evaluation/balanced_null.py',
        'src/palimpsest/evaluation/features.py', 'src/palimpsest/evaluation/robust_views.py',
        'experiments/origin_detection/semantic_gaussian/run_iteration.py',
        'experiments/origin_detection/source_view_risk/run_iteration.py',
        'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        'tests/detection/test_conditional_score.py')]
    return {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in sorted(paths)}


def main():
    if (OUTPUT/'iteration.json').exists():
        raise FileExistsError('Preserve radial scalar campaign')
    controls = json.loads((OUTPUT/'software_controls.json').read_text())
    if not controls['passed'] or controls['code_pins'] != code_pins():
        raise ValueError('Radial scalar controls changed')
    rows, parent = cure_inputs()
    base_receipt = audit_receipt(CURE/'iteration.json')
    fit_rows = sorted([r for r in rows if r['role']=='fit' and r['variant'] in VARIANTS[:2]],
                     key=lambda r:tuple(r[k] for k in ('domain','src','condition','variant')))
    x, truth, weights, sources = weighted_source_arrays(rows, FULL_NAMES, VARIANTS[:2],
        expected_source_counts={'rr':540,'chimera':720},seed=CONFIG['seed'])
    if len(x)!=7560 or len(rows)!=15120 or not np.array_equal(x, [[float(r[n]) for n in FULL_NAMES] for r in fit_rows]):
        raise ValueError('Radial score coverage/traversal differs')
    assignment, control = balanced_source_null(fit_rows,seed=CONFIG['seed'])
    prior = json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if prior['assignment_sha256'] != control['assignment_sha256']:
        raise ValueError('Radial null assignment differs')
    sham = np.array([assignment[r['domain'],r['src']] for r in fit_rows])
    radius = radial_coordinate(x)
    wrong_radius = radius.copy()
    strata = defaultdict(list)
    for i,r in enumerate(fit_rows):
        strata[tuple(r[k] for k in ('domain','scene','label','condition','variant'))].append(i)
    rng = np.random.default_rng(CONFIG['seed'])
    for key,indices in sorted(strata.items()):
        wrong_radius[indices] = radius[rng.permutation(indices)]
    if np.array_equal(radius,wrong_radius):
        raise ValueError('Wrong radial association unchanged')
    views = {k+'/'+v:view for v in VARIANTS[:2] for processed in (False,True)
             for k,view in feature_views(rows,FULL_NAMES,'threshold',processed=processed,variant=v).items()}
    if len(views)!=24:
        raise ValueError('Radial threshold coverage differs')
    bases,base_pins = {},{}
    for name in ('source','source_null'):
        f=CURE/('full_'+name+'_rule.json')
        if file_sha256(f)!=base_receipt['rule_files']['full/'+name]:
            raise ValueError('Frozen base scalar rule changed')
        bases[name]=StableRule.load(f);base_pins[name]=file_sha256(f)
    results,scores,artifacts={}, {}, {}
    start=perf_counter()
    with threadpool_limits(limits=1):
        for key,conditional,base,labels,override in (
            ('constant_gaussian',False,bases['source'],truth,None),
            ('radial',True,bases['source'],truth,None),
            ('wrong_radius',True,bases['source'],truth,wrong_radius),
            ('source_null',True,bases['source_null'],sham,None)):
            rule,diagnostics=fit_conditional_score(x,labels,weights,base,conditional=conditional,radius_override=override)
            rule,calibration=class_threshold(rule,views)
            f=OUTPUT/(key+'_rule.json');rule.save(f)
            loaded=ConditionalScoreRule.load(f)
            if not np.array_equal(rule.score(x[:12]),loaded.score(x[:12])):
                raise ValueError('Radial load parity failed')
            margins,result=score_rule(rows,FULL_NAMES,rule,CONFIG)
            result.update({'threshold':rule.threshold,'calibration':calibration,'fit_diagnostics':diagnostics})
            results[key],scores[key],artifacts[key]=result,margins,file_sha256(f)
            print(json.dumps({'candidate':key,'minimum_domain_ba':result['minimum_domain_ba'],
                             'minimum_scene_ba':result['minimum_scene_ba'],
                             'maximum_drop':result['maximum_any_scene_drop']}),flush=True)
    write_json(OUTPUT/'selection_scores.json',scores)
    write_json(OUTPUT/'iteration.json',{'candidates':results,'code_pins':code_pins(),
        'inventory_sha256':parent['inventory_sha256'],'balanced_assignment_sha256':control['assignment_sha256'],
        'base_rule_sha256':base_pins,'wrong_radius_sha256':sha256(wrong_radius.tobytes()).hexdigest(),
        'elapsed_s':perf_counter()-start,'null_raw_processed_auc_ci95':null_intervals(scores['source_null']),
        'selection_scores_sha256':file_sha256(OUTPUT/'selection_scores.json'),'rule_files':artifacts,
        'goal_achieved':False,'scope':'Frozen scalar conditional likelihood;lognorm is not IQA;exposed development'})


if __name__=='__main__':main()
