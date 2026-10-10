"""Fit-only consistency selection on signed2048-dimensional token statistics."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.algorithms.readouts.consistent_source_risk import fit_consistent_source_risk
from palimpsest.detection.representations.frozen_cure_tokens import MIXED_NAMES, FEATURE_NAMES
from palimpsest.evaluation.consistency_crossfit import crossfit_strengths
from palimpsest.evaluation.source_crossfit import source_folds
from palimpsest.evaluation.source_training import weighted_source_arrays
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.evaluation.features import feature_views
from palimpsest.io.hashing import file_sha256
from palimpsest.io.tables import read_rows
from palimpsest.paths import REPO_ROOT, WORK_DIR
from experiments.data_preparation.development_roles.audit_roles import audit_inventory
from experiments.origin_detection.cure_token_statistics.prepare_features import OUTPUT as TOKENS, audit_cache
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS, audit_receipt
from experiments.origin_detection.semantic_gaussian.run_iteration import null_intervals
from experiments.origin_detection.paired_kernel.run_iteration import wrong_sources
from experiments.origin_detection.compact_frozen_encoder.run_iteration import score_rule
from experiments.origin_detection.threshold_calibration.fit_threshold import class_threshold
from experiments.origin_detection.phase_statistics.run_iteration import write_json

OUTPUT = WORK_DIR/'robust_statistics/token_consistency_crossfit'
STRENGTHS = (0, .1, 1, 10)
CONFIG = {'seed':20261006, 'bootstrap_repetitions':2000,
          'provisional_final_ba':.8, 'provisional_final_drop':.02}


def code_pins():
    paths = list(Path(__file__).parent.glob('*.py')) + [Path(__file__).parent/'README.md']
    paths += [REPO_ROOT/p for p in (
        'src/palimpsest/evaluation/consistency_crossfit.py', 'src/palimpsest/evaluation/source_crossfit.py',
        'src/palimpsest/evaluation/source_training.py', 'src/palimpsest/evaluation/balanced_null.py',
        'src/palimpsest/evaluation/features.py', 'src/palimpsest/evaluation/numeric_features.py',
        'src/palimpsest/evaluation/robust_views.py', 'src/palimpsest/detection/algorithms/readouts/consistent_source_risk.py',
        'src/palimpsest/detection/algorithms/readouts/source_view_risk.py',
        'src/palimpsest/detection/representations/frozen_cure_tokens.py',
        'experiments/origin_detection/cure_token_statistics/prepare_features.py',
        'experiments/origin_detection/source_view_risk/run_iteration.py',
        'experiments/data_preparation/development_roles/audit_roles.py',
        'experiments/origin_detection/semantic_gaussian/run_iteration.py',
        'experiments/origin_detection/paired_kernel/run_iteration.py',
        'experiments/origin_detection/compact_frozen_encoder/run_iteration.py',
        'experiments/origin_detection/threshold_calibration/fit_threshold.py',
        'tests/evaluation/test_consistency_crossfit.py', 'tests/evaluation/test_source_crossfit.py',
        'tests/detection/test_consistent_source_risk.py', 'tools/audit_source_crossfit.py')]
    return {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in sorted(set(paths))}


def inputs():
    receipt = audit_receipt(TOKENS/'features.json')
    if (receipt['records'] != 15120 or receipt['images'] != 6300
            or not receipt['all_original_exact'] or receipt['feature_names'] != list(FEATURE_NAMES)
            or file_sha256(TOKENS/'vectors.npy') != receipt['vectors_sha256']
            or file_sha256(TOKENS/'metadata.csv') != receipt['metadata_sha256']):
        raise ValueError('Signed token cache changed')
    metadata = read_rows(TOKENS/'metadata.csv')
    inventory = [r for r in metadata if r['variant'] == 'raw']
    structure = audit_inventory(inventory)
    if structure['source_role_counts'] != {'fit':1260, 'threshold':420, 'selection':420}:
        raise ValueError('Token source roles changed')
    values = np.load(TOKENS/'vectors.npy', mmap_mode='r')
    if values.dtype != np.dtype('<f8'):raise ValueError('Token cache dtype differs')
    return audit_cache(metadata, values, inventory), receipt


def fit(x, y, w, s, strength, manifest, wrong=None):
    return fit_consistent_source_risk(x, y, w, s, strength=strength, consistency_groups=wrong,
        feature_names=MIXED_NAMES, temperature=.1, ridge=.01, scale_floor=.001,
        maximum_iterations=500, gradient_tolerance=1e-5, manifest_sha=manifest)


def training(rows):
    records = sorted([r for r in rows if r['role']=='fit' and r['variant'] in VARIANTS[:2]],
                     key=lambda r:tuple(r[k] for k in ('domain','src','condition','variant')))
    x, y, weights, sources = weighted_source_arrays(rows, MIXED_NAMES, VARIANTS[:2],
        expected_source_counts={'rr':540,'chimera':720}, seed=CONFIG['seed'])
    if len(x)!=7560 or not np.array_equal(x, [[float(r[n]) for n in MIXED_NAMES] for r in records]):
        raise ValueError('Token training traversal differs')
    return records, x, y, weights, sources


def zero_reference(margins):
    receipt = audit_receipt(TOKENS/'iteration.json'); path = TOKENS/'selection_scores.json'
    if file_sha256(path)!=receipt['selection_scores_sha256']:raise ValueError('Token zero reference changed')
    old = json.loads(path.read_text())['mixed/source']
    key = lambda r:tuple(r[k] for k in ('domain','src','condition','variant'))
    left, right = {key(r):r for r in margins}, {key(r):r for r in old}
    if len(left)!=5040 or set(left)!=set(right):raise ValueError('Token zero coverage differs')
    mismatches = sum((left[k]['score']>0)!=(right[k]['score']>0) for k in left)
    difference = max(abs(left[k]['score']-right[k]['score']) for k in left)
    if mismatches or difference>1e-12:raise ValueError('Token zero decisions/margins changed')
    return {'records':5040, 'mismatches':mismatches, 'maximum_margin_discrepancy':difference,
            'scores_sha256':file_sha256(path), 'scope':'Loss/feature software identity only'}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args(); OUTPUT.mkdir(parents=True, exist_ok=True)
    if (OUTPUT/'iteration.json').exists() or (OUTPUT/'crossfit.json').exists():
        raise FileExistsError('Preserve token consistency calculation')
    controls = json.loads((OUTPUT/'software_controls.json').read_text())
    if not controls['passed'] or controls['code_pins']!=code_pins():raise ValueError('Controls changed')
    rows, parent = inputs(); manifest = parent['inventory_sha256']
    records, x, truth, weights, sources = training(rows)
    folds = source_folds(records)
    ids = np.array([folds[r['domain'],r['src']] for r in records])
    if args.pilot:
        if (OUTPUT/'pilot.json').exists():raise FileExistsError('Preserve pilot')
        selected = ids!=0; _, group = np.unique(sources[selected], return_inverse=True)
        if int(selected.sum())!=6048:raise ValueError('Pilot fold size changed')
        start = perf_counter()
        with threadpool_limits(limits=1):
            rule, diagnostic = fit(x[selected],truth[selected],weights[selected],group,1,manifest)
        elapsed = perf_counter()-start; path = OUTPUT/'pilot_fold_rule.json'; rule.save(path)
        write_json(OUTPUT/'pilot.json', {'passed':elapsed<=120, 'fit_s':elapsed,
            'projected_40_fit_s':40*elapsed, 'rows':6048, 'dimensions':2048, 'strength':1,
            'assumption':'Same train-fold sizes;linear fit count,conditional on strength/iterations',
            'code_pins':code_pins(), 'features_receipt_sha256':file_sha256(TOKENS/'features.json'),
            'rule_sha256':file_sha256(path), 'diagnostic':diagnostic,
            'scope':'Actual fit-fold timing only;held scores and selection not evaluated'})
        print(json.dumps({'pilot_fit_s':elapsed, 'projected_40_fit_s':40*elapsed}),flush=True)
        if elapsed>120:raise ValueError('Pilot too costly;restructure')
        return
    pilot = json.loads((OUTPUT/'pilot.json').read_text())
    if (not pilot['passed'] or pilot['code_pins']!=code_pins()
            or pilot['features_receipt_sha256']!=file_sha256(TOKENS/'features.json')
            or pilot['rule_sha256']!=file_sha256(OUTPUT/'pilot_fold_rule.json')):
        raise ValueError('Pilot gate changed/failed')
    assignment, null = balanced_source_null(records, seed=CONFIG['seed'])
    prior = json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if prior['assignment_sha256']!=null['assignment_sha256']:raise ValueError('Token null mapping changed')
    sham = np.array([assignment[r['domain'],r['src']] for r in records])
    cv, chosen = {}, {}; start = perf_counter()
    with threadpool_limits(limits=1):
        for mode, labels in (('truth',truth),('null',sham)):
            def progress(strength, result):
                write_json(OUTPUT/f'cv_{mode}_{strength}.json',result)
                print(json.dumps({'cv':mode,'strength':strength,'minimum_ba':result['minimum_all_scope_ba'],
                                  'maximum_drop':result['maximum_all_scope_drop']}),flush=True)
            cv[mode], chosen[mode], actual = crossfit_strengths(records,x,labels,weights,sources,STRENGTHS,
                VARIANTS[:2],lambda a,b,c,d,e:fit(a,b,c,d,e,manifest),progress=progress)
            if actual!=folds or any(len(v['metrics'])!=30 or len(v['paired_drops'])!=35 for v in cv[mode].values()):
                raise ValueError('OOF fold/metric coverage differs')
        write_json(OUTPUT/'crossfit.json', {'results':cv,'chosen':chosen,'fold_seed':20261007,
            'fold_sources':{'/'.join(k):v for k,v in folds.items()},'code_pins':code_pins(),
            'inventory_sha256':manifest,'scope':'Fit-only weak-view OOF;not independent validation'})
        views = {k+'/'+v:view for v in VARIANTS[:2] for p in (False,True)
                 for k,view in feature_views(rows,MIXED_NAMES,'threshold',processed=p,variant=v).items()}
        if len(views)!=24:raise ValueError('Outer threshold coverage differs')
        results, scores, artifacts = {}, {}, {}; wrong = wrong_sources(rows,sources)
        for key, labels, strength, wrong_group in (
            ('zero',truth,0,None), ('selected',truth,float(chosen['truth'][0]),None),
            ('selected_wrong_source',truth,float(chosen['truth'][0]),wrong),
            ('selected_null',sham,float(chosen['null'][0]),None)):
            rule, diagnostic = fit(x,labels,weights,sources,strength,manifest,wrong_group)
            rule, diagnostic['calibration'] = class_threshold(rule,views)
            margins, result = score_rule(rows,MIXED_NAMES,rule,CONFIG)
            if key=='zero':reference = zero_reference(margins)
            result.update({'fit_diagnostics':diagnostic,'threshold':rule.threshold,'chosen_strength':strength})
            path = OUTPUT/(key+'_rule.json'); rule.save(path)
            results[key], scores[key], artifacts[key] = result,margins,file_sha256(path)
            print(json.dumps({'candidate':key,'strength':strength,'minimum_ba':result['minimum_domain_ba'],
                'minimum_scene_ba':result['minimum_scene_ba'],'maximum_drop':result['maximum_any_scene_drop']}),flush=True)
    write_json(OUTPUT/'selection_scores.json',scores)
    write_json(OUTPUT/'iteration.json', {'candidates':results,'chosen_strengths':chosen,'code_pins':code_pins(),
        'features_receipt_sha256':file_sha256(TOKENS/'features.json'),'zero_reference_gate':reference,
        'crossfit_sha256':file_sha256(OUTPUT/'crossfit.json'),'inventory_sha256':manifest,
        'balanced_assignment_sha256':null['assignment_sha256'],'elapsed_s':perf_counter()-start,
        'null_raw_processed_auc_ci95':null_intervals(scores['selected_null']),
        'selection_scores_sha256':file_sha256(OUTPUT/'selection_scores.json'),'rule_files':artifacts,
        'goal_achieved':False,'scope':'Known consistency on clipped token descriptors;exposed development'})


if __name__=='__main__':main()
