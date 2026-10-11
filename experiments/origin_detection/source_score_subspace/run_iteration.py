"""Three fit-local discriminant scores with fixed final source-consistency CV."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from palimpsest.detection.models.frozen_features.cure.source_score_subspace import ScoreSubspaceFitter, ScoreSubspaceRule, calibrate_score_subspace
from palimpsest.evaluation.source_consistency_campaign import run_source_consistency
from palimpsest.evaluation.source_crossfit import source_folds
from palimpsest.evaluation.balanced_null import balanced_source_null
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR,REPO_ROOT
from experiments.origin_detection.token_csp_consistency.run_iteration import (
    PARENT,NAMES,CONFIG,STRENGTHS,VARIANTS,inputs,training,
    code_pins as input_pins,class_threshold,score_rule,wrong_sources,null_intervals,write_json)

OUTPUT=WORK_DIR/'robust_statistics/source_score_subspace'


def code_pins():
    pins=input_pins();paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent/'README.md',
        REPO_ROOT/'src/palimpsest/detection/models/frozen_features/cure/source_score_subspace.py',
        REPO_ROOT/'tests/detection/test_source_score_subspace.py']
    pins.update({str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in sorted(paths)})
    return pins


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'iteration.json').exists() or (OUTPUT/'crossfit.json').exists():
        raise FileExistsError('Preserve score subspace experiment')
    controls=json.loads((OUTPUT/'software_controls.json').read_text())
    if not controls['passed'] or controls['code_pins']!=code_pins():raise ValueError('Score subspace controls changed')
    rows,parent=inputs();manifest=parent['inventory_sha256'];records,x,truth,weights,sources=training(rows)
    fitter=ScoreSubspaceFitter(NAMES,manifest_sha=manifest)
    if args.pilot:
        if (OUTPUT/'pilot.json').exists():raise FileExistsError('Preserve score bank pilot')
        folds=source_folds(records);chosen=np.array([folds[r['domain'],r['src']]!=0 for r in records])
        _,groups=np.unique(sources[chosen],return_inverse=True)
        with threadpool_limits(limits=1):
            start=perf_counter();rule,cold=fitter.fit(x[chosen],truth[chosen],weights[chosen],groups,10)
            cold_s=perf_counter()-start;start=perf_counter()
            _,warm=fitter.fit(x[chosen],truth[chosen],weights[chosen],groups,1)
            warm_s=perf_counter()-start
        if fitter.builds!=1 or cold['training_arrays_sha256']!=warm['training_arrays_sha256']:
            raise ValueError('Same training bank rebuilt or changed')
        path=OUTPUT/'pilot_fold_rule.json';rule.save(path)
        projected=10*cold_s+30*warm_s
        write_json(OUTPUT/'pilot.json',{'passed':max(cold_s,warm_s)<=120,'cold_fit_s':cold_s,'warm_fit_s':warm_s,
            'projected_40_meta_cv_s':projected,'rows':int(chosen.sum()),'input_dimensions':2576,'meta_dimensions':3,
            'cold_diagnostic':cold,'warm_diagnostic':warm,'base_bank_builds':fitter.builds,'code_pins':code_pins(),
            'features_receipt_sha256':file_sha256(PARENT/'features.json'),'rule_sha256':file_sha256(path),
            'assumption':'10 same-size coldfold banks+30 reusedbank fits;strength-dependent cost;fullfit/scoring not included',
            'scope':'Actual train-fold fit budget;no held or outer scores'})
        print(json.dumps({'cold_s':cold_s,'warm_s':warm_s,'projected40_meta_s':projected}),flush=True)
        if max(cold_s,warm_s)>120:raise ValueError('Score bank pilot too costly')
        return
    pilot=json.loads((OUTPUT/'pilot.json').read_text())
    if (not pilot['passed'] or pilot['code_pins']!=code_pins()
            or pilot['features_receipt_sha256']!=file_sha256(PARENT/'features.json')
            or pilot['rule_sha256']!=file_sha256(OUTPUT/'pilot_fold_rule.json')):
        raise ValueError('Score subspace pilot changed/failed')
    assignment,null=balanced_source_null(records,seed=CONFIG['seed'])
    prior=json.loads((WORK_DIR/'robust_statistics/balanced_null/assignment.json').read_text())
    if null['assignment_sha256']!=prior['assignment_sha256']:raise ValueError('Score bank null mapping changed')
    sham=np.array([assignment[r['domain'],r['src']] for r in records]);zero=None

    def calibrate(rule,views):
        nonlocal zero
        fixed,diagnostic=calibrate_score_subspace(rule,views,class_threshold)
        if zero is None:zero=fixed
        return fixed,diagnostic

    def zero_control(margins):
        selected=[r for r in rows if r['role']=='selection']
        native=np.array([[float(r[n]) for n in NAMES] for r in selected])
        path=OUTPUT/'zero_portable_control.json';zero.save(path);restored=ScoreSubspaceRule.load(path)
        actual=zero.score(native);mapped=zero.readout.score(zero.bank.transform(native))
        if (len(actual)!=5040 or not np.array_equal(actual,mapped)
                or not np.array_equal(actual,restored.score(native))
                or not np.array_equal(actual-zero.threshold,[r['score'] for r in margins])):
            raise ValueError('Score subspace zero portable/mapped differs')
        return {'records':5040,'portable_mapped_loaded_margin_exact':True,'scope':'Shared-rule software identity only'}

    run_source_consistency(rows=rows,names=NAMES,records=records,x=x,truth=truth,sham=sham,
        weights=weights,sources=sources,strengths=STRENGTHS,variants=VARIANTS[:2],output=OUTPUT,
        fit=fitter.fit,calibrate=calibrate,score=lambda a,b,c:score_rule(a,b,c,CONFIG),
        wrong_sources=wrong_sources(rows,sources),zero_reference=zero_control,null_intervals=null_intervals,
        pins=code_pins(),manifest_sha=manifest,assignment_sha=null['assignment_sha256'],
        features_sha=file_sha256(PARENT/'features.json'))
    if fitter.builds!=12:raise ValueError('Expected10CV+2fullfit bank cache entries')
    write_json(OUTPUT/'bank_cache_audit.json',{'passed':True,'base_bank_builds':fitter.builds,
        'base_fits':3*fitter.builds,'training_array_keys':sorted(fitter.bases),
        'scope':'Memoization accounting,not actual source-member independence certification'})


if __name__=='__main__':main()
