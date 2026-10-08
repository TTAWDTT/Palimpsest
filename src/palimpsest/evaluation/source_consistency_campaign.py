"""Shared OOF selection and fixed outer controls for source consistency.

Previously converged OOF artifacts can be supplied by a caller that has already
verified their signed provenance. This module rechecks records and arithmetic;
it cannot independently certify how an earlier optimizer obtained the scores.
"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from .consistency_crossfit import crossfit_strengths
from .source_crossfit import source_folds, crossfit_rates, choose_strength
from .features import feature_views
from palimpsest.io.hashing import file_sha256


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def validate_prior_oof(result, records, labels, folds, variants):
    """Refuse changed saved labels/folds/identities or confusion arithmetic."""
    identities = {(r['domain'],r['src'],r['condition'],r['variant']) for r in records}
    expected = {(r['domain'],r['src']):int(y) for r,y in zip(records,labels)}
    saved = result['scores']
    if (len(saved)!=len(records)
            or {(r['domain'],r['src'],r['condition'],r['variant']) for r in saved}!=identities
            or any(r['fold']!=folds[r['domain'],r['src']]
                   or int(r['label']=='FAKE')!=expected[r['domain'],r['src']] for r in saved)):
        raise ValueError('Reused OOF source/label/fold identity differs')
    actual = crossfit_rates(saved, variants)
    if any(result[k]!=value for k,value in actual.items()):
        raise ValueError('Reused OOF arithmetic differs')
    return actual


def run_source_consistency(*, rows, names, records, x, truth, sham, weights, sources,
                           strengths, variants, output, fit, calibrate, score,
                           wrong_sources, zero_reference, null_intervals, pins,
                           manifest_sha, assignment_sha, features_sha, prior=None):
    """Freeze fit-only choice before four fixed threshold/selection evaluations."""
    output = Path(output); start = perf_counter(); prior = prior or {}
    folds = source_folds(records)
    cv, chosen = {}, {}
    with threadpool_limits(limits=1):
        for mode, labels in (('truth', truth), ('null', sham)):
            cv[mode] = dict(prior.get(mode, {}))
            if not set(cv[mode]) <= {str(k) for k in strengths}:
                raise ValueError('Undeclared reused consistency strength')
            for result in cv[mode].values():
                validate_prior_oof(result,records,labels,folds,variants)
            missing = tuple(k for k in strengths if str(k) not in cv[mode])

            def progress(strength, result):
                _write(output/f'cv_{mode}_{strength}.json', result)
                print(json.dumps({'cv':mode,'strength':strength,
                    'minimum_ba':result['minimum_all_scope_ba'],
                    'maximum_drop':result['maximum_all_scope_drop']}),flush=True)

            if missing:
                new, _, actual_folds = crossfit_strengths(records,x,labels,weights,sources,missing,
                                                         variants,fit,progress=progress)
                if actual_folds!=folds:raise ValueError('Campaign source folds changed')
                cv[mode].update(new)
            if any(len(v['metrics'])!=30 or len(v['paired_drops'])!=35 for v in cv[mode].values()):
                raise ValueError('Campaign OOF metric/pair coverage differs')
            chosen[mode] = choose_strength(cv[mode])
        _write(output/'crossfit.json', {'results':cv,'chosen':chosen,'fold_seed':20261007,
            'fold_sources':{'/'.join(k):v for k,v in folds.items()},'code_pins':pins,
            'inventory_sha256':manifest_sha,'scope':'Fit-only OOF;not independent validation'})
        views = {k+'/'+v:view for v in variants for processed in (False,True)
                 for k,view in feature_views(rows,names,'threshold',processed=processed,variant=v).items()}
        if len(views)!=24:raise ValueError('Threshold coverage differs')
        results, scores, artifacts = {}, {}, {}
        for key, labels, strength, wrong in (
            ('zero',truth,0,None), ('selected',truth,float(chosen['truth'][0]),None),
            ('selected_wrong_source',truth,float(chosen['truth'][0]),wrong_sources),
            ('selected_null',sham,float(chosen['null'][0]),None)):
            rule, diagnostic = fit(x,labels,weights,sources,strength,wrong)
            rule, diagnostic['calibration'] = calibrate(rule,views)
            margins, result = score(rows,names,rule)
            if key=='zero':reference = zero_reference(margins)
            result.update({'fit_diagnostics':diagnostic,'threshold':rule.threshold,'chosen_strength':strength})
            path = output/(key+'_rule.json'); rule.save(path)
            results[key], scores[key], artifacts[key] = result,margins,file_sha256(path)
            print(json.dumps({'candidate':key,'strength':strength,'minimum_ba':result['minimum_domain_ba'],
                'minimum_scene_ba':result['minimum_scene_ba'],'maximum_drop':result['maximum_any_scene_drop']}),flush=True)
    _write(output/'selection_scores.json', scores)
    _write(output/'iteration.json', {'candidates':results,'chosen_strengths':chosen,'code_pins':pins,
        'features_receipt_sha256':features_sha,'zero_reference_gate':reference,
        'crossfit_sha256':file_sha256(output/'crossfit.json'),'inventory_sha256':manifest_sha,
        'balanced_assignment_sha256':assignment_sha,'elapsed_s':perf_counter()-start,
        'null_raw_processed_auc_ci95':null_intervals(scores['selected_null']),
        'selection_scores_sha256':file_sha256(output/'selection_scores.json'),'rule_files':artifacts,
        'reused_cv_strengths':{k:sorted(v) for k,v in prior.items()},'goal_achieved':False,
        'scope':'Fixed conventional head;repeatedly exposed development;external validation absent'})
