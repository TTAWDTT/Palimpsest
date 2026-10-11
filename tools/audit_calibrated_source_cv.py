"""Recount saved calibration-aware OOF scores and recorded disjoint subroles.

This verifies arithmetic and saved role records, not optimizer training identity
or scientific independence. No model or image processing is executed.
"""

import argparse
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import runpy

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT


def verify_calibration(rows,fold_sources):
    for row in rows:
        key=row['domain']+'/'+row['src'];held=fold_sources[key]
        if row['fold']!=held or row['calibration_fold']!=(held+1)%5:
            raise ValueError('Recorded held/calibration fold differs')
    for held in range(5):
        cal=(held+1)%5
        train={k for k,v in fold_sources.items() if v not in (held,cal)}
        calibration={k for k,v in fold_sources.items() if v==cal}
        test={k for k,v in fold_sources.items() if v==held}
        if train&calibration or train&test or calibration&test:
            raise ValueError('Recorded source roles overlap')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();output=args.directory/'calibrated_oof_audit.json'
    if output.exists():raise FileExistsError('Preserve calibrated OOF audit')
    tools=runpy.run_path(str(REPO_ROOT/'tools/audit_source_crossfit.py'))
    toy=[{'domain':'rr','src':str(i),'fold':i,'calibration_fold':(i+1)%5} for i in range(5)]
    mapping={'rr/'+str(i):i for i in range(5)};verify_calibration(toy,mapping)
    bad=deepcopy(toy);bad[0]['calibration_fold']=0
    try:verify_calibration(bad,mapping)
    except ValueError:pass
    else:raise ValueError('Wrong calibration-fold control accepted')
    receipt=json.loads((args.directory/'iteration.json').read_text());path=args.directory/'crossfit.json'
    if file_sha256(path)!=receipt['crossfit_sha256']:raise ValueError('Calibrated crossfit bytes changed')
    for name,sha in receipt['code_pins'].items():
        if file_sha256(REPO_ROOT/name)!=sha:raise ValueError('Calibrated producer source changed')
    data=json.loads(path.read_text());fold_sources=data['fold_sources'];results={}
    for mode,candidates in data['results'].items():
        if set(candidates)!={'0','0.1','1','10'}:raise ValueError('Calibrated strength grid differs')
        counted={}
        for strength,record in candidates.items():
            if len(record['scores'])!=7560 or len(record['fit_diagnostics'])!=5:
                raise ValueError('Calibrated OOF count differs')
            rates=tools['verify_rates'](record['scores'],record)
            if len(rates['metrics'])!=30 or len(rates['paired_drops'])!=35:
                raise ValueError('Calibrated metric/pair coverage differs')
            sources=tools['verify_folds'](record['scores'],fold_sources)
            if len(sources)!=1260 or any(len(v)!=6 for v in sources.values()):
                raise ValueError('Calibrated source view coverage differs')
            verify_calibration(record['scores'],fold_sources)
            for held,diagnostic in enumerate(record['fit_diagnostics']):
                cal=(held+1)%5
                if (diagnostic['held_fold']!=held or diagnostic['calibration_fold']!=cal
                        or diagnostic['fit_records']!=4536 or diagnostic['sources']!=756
                        or diagnostic['calibration_records']!=1512
                        or diagnostic['calibration_source_count']!=252 or diagnostic['held_source_count']!=252
                        or diagnostic['maximum_absolute_gradient']>1e-5):
                    raise ValueError('Calibrated fitting diagnostics differ')
                if sum(v not in (held,cal) for v in fold_sources.values())!=756:
                    raise ValueError('Calibrated train-source count differs')
            counted[strength]=rates
        feasible=[k for k in counted if Fraction(counted[k]['minimum_all_scope_ba'])>=Fraction(4,5)]
        if feasible:
            chosen=min(feasible,key=lambda k:(Fraction(counted[k]['maximum_all_scope_drop']),
                       -Fraction(counted[k]['minimum_all_scope_ba']),Fraction(k)))
        else:
            chosen=min(counted,key=lambda k:(-Fraction(counted[k]['minimum_all_scope_ba']),
                       Fraction(counted[k]['maximum_all_scope_drop']),Fraction(k)))
        if chosen!=data['chosen'][mode][0]:raise ValueError('Calibrated strength selection differs')
        results[mode]={'strengths':4,'groups_each':30,'pairs_each':35,'selected':chosen,'passed':True}
    bad=deepcopy(data['results']['truth']['0'])
    changed=next(i for i,row in enumerate(bad['scores']) if row['score']!=0)
    bad['scores'][changed]['score']*=-1
    try:tools['verify_rates'](bad['scores'],bad)
    except ValueError:pass
    else:raise ValueError('Changed calibrated score accepted')
    bad=deepcopy(data['results']['truth']['0']['scores']);bad[0]['calibration_fold']=bad[0]['fold']
    try:verify_calibration(bad,fold_sources)
    except ValueError:pass
    else:raise ValueError('Changed real calibration-fold accepted')
    value={'passed':True,'results':results,'wrong_calibration_fold_rejected':True,'changed_real_score_rejected':True,
        'crossfit_sha256':file_sha256(path),'script_sha256':file_sha256(Path(__file__)),
        'scope':'Recorded role separation and independent saved-score integer arithmetic;not actual optimizer members or scientific independence'}
    output.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');print(json.dumps(value))


if __name__=='__main__':main()
