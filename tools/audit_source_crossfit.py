"""Recount saved fit-only OOF predictions without the producer's rate/selector code.

Audits arithmetic and recorded folds, not actual optimizer training memberships,
near-duplicate independence, or an external validation result.
"""

import argparse
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
from itertools import combinations
import json
import math
from pathlib import Path

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT


def recount(rows):
    groups=defaultdict(list);seen=set();metadata={}
    for r in rows:
        key=tuple(r[k] for k in ('domain','scene','condition','variant','src'))
        if key in seen or r['role']!='fit' or r['label'] not in ('REAL','FAKE') or not math.isfinite(r['score']):
            raise ValueError('Invalid OOF identity/score')
        seen.add(key);source=(r['domain'],r['src']);m=(r['scene'],r['label'])
        if source in metadata and metadata[source]!=m:raise ValueError('Conflicting OOF metadata')
        metadata[source]=m;groups[key[:4]].append(r)
        if r['scene']!='all':groups[(key[0],'all',*key[2:4])].append(r)
    rates={}
    for k,records in groups.items():
        value=Fraction(0)
        for label in ('REAL','FAKE'):
            selected=[r for r in records if r['label']==label]
            if not selected:raise ValueError('Missing OOF class')
            correct=sum((r['score']>0)==(label=='FAKE') for r in selected)
            value+=Fraction(correct,2*len(selected))
        rates['/'.join(k)]=str(value)
    drops={}
    def pair(a,b,name):
        if {r['src'] for r in groups[a]}!={r['src'] for r in groups[b]}:raise ValueError('OOF pair mismatch')
        drops['/'.join(name)]=str(Fraction(rates['/'.join(a)])-Fraction(rates['/'.join(b)]))
    for domain,scene,condition,variant in groups:
        if condition!='original':
            pair((domain,scene,'original',variant),(domain,scene,condition,variant),
                 (domain,scene,'original>'+condition,variant))
    for domain,scene,condition in sorted({k[:3] for k in groups}):
        for a,b in combinations(('raw','jpeg90_444_after_resize256'),2):
            pair((domain,scene,condition,a),(domain,scene,condition,b),(domain,scene,condition,a+'>'+b))
    return {'metrics':rates,'paired_drops':drops,'minimum_all_scope_ba':str(min(map(Fraction,rates.values()))),
            'maximum_all_scope_drop':str(max(map(Fraction,drops.values())))}


def verify_rates(rows,quoted):
    actual=recount(rows)
    if any(actual[k]!=quoted[k] for k in actual):raise ValueError('OOF confusion arithmetic differs')
    return actual


def verify_folds(rows,fold_sources):
    by_source=defaultdict(list)
    for r in rows:by_source[r['domain']+'/'+r['src']].append(r)
    if set(by_source)!=set(fold_sources):raise ValueError('OOF source-fold coverage differs')
    for source,views in by_source.items():
        if {r['fold'] for r in views}!={fold_sources[source]}:raise ValueError('OOF source split across recorded folds')
    return by_source


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args();out=args.directory/'oof_counts_audit.json'
    if out.exists():raise FileExistsError('Preserve OOF audit')
    toy=[{'domain':'rr','scene':'all','src':label,'label':label,'role':'fit','condition':c,'variant':v,
          'score':1. if label=='FAKE' else -1.,'fold':int(label=='FAKE')}
         for label in ('REAL','FAKE') for c in ('original','transfer','redigital')
         for v in ('raw','jpeg90_444_after_resize256')]
    known=recount(toy)
    if known['minimum_all_scope_ba']!='1' or known['maximum_all_scope_drop']!='0':raise ValueError('Known rate control failed')
    bad=deepcopy(toy);bad[6]['score']=-1.
    try:verify_rates(bad,known)
    except ValueError:pass
    else:raise ValueError('Planted wrong OOF score accepted')
    folds={'rr/REAL':0,'rr/FAKE':1};verify_folds(toy,folds)
    bad=deepcopy(toy);bad[0]['fold']=1
    try:verify_folds(bad,folds)
    except ValueError:pass
    else:raise ValueError('Planted split source accepted')
    receipt=json.loads((args.directory/'iteration.json').read_text())
    path=args.directory/'crossfit.json'
    if file_sha256(path)!=receipt['crossfit_sha256']:raise ValueError('Crossfit artifact changed')
    for k,v in receipt['code_pins'].items():
        if file_sha256(REPO_ROOT/k)!=v:raise ValueError('Crossfit producer code changed')
    data=json.loads(path.read_text());results={}
    for mode,candidates in data['results'].items():
        if set(candidates)!={'0','0.1','1','10'}:raise ValueError('Crossfit strength set differs')
        counted={}
        for key,c in candidates.items():
            if len(c['scores'])!=7560 or len(c['fit_diagnostics'])!=5:raise ValueError('OOF record/fold count differs')
            rates=verify_rates(c['scores'],c);sources=verify_folds(c['scores'],data['fold_sources'])
            if len(sources)!=1260 or any(len(v)!=6 for v in sources.values()):raise ValueError('OOF source/view count differs')
            for fold in range(5):
                if sum(data['fold_sources'][s]==fold for s in sources)!=252:raise ValueError('OOF fold source count differs')
            for d in c['fit_diagnostics']:
                if d['fit_records']!=6048 or d['sources']!=1008 or d['maximum_absolute_gradient']>1e-5:
                    raise ValueError('Recorded CV fit diagnostics differ')
            counted[key]=rates
        feasible=[k for k in counted if Fraction(counted[k]['minimum_all_scope_ba'])>=Fraction(4,5)]
        if feasible:
            expected=sorted(feasible,key=lambda k:(Fraction(counted[k]['maximum_all_scope_drop']),
                -Fraction(counted[k]['minimum_all_scope_ba']),Fraction(k)))[0]
        else:
            expected=sorted(counted,key=lambda k:(-Fraction(counted[k]['minimum_all_scope_ba']),
                Fraction(counted[k]['maximum_all_scope_drop']),Fraction(k)))[0]
        if expected!=data['chosen'][mode][0] or expected!=receipt['chosen_strengths'][mode][0]:
            raise ValueError('Recorded strength selector differs')
        results[mode]={'strengths':len(counted),'metric_groups_each':30,'paired_comparisons_each':35,
                       'selected_strength':expected,'passed':True}
    # Same gate must reject a changed real score in an otherwise real receipt.
    c=deepcopy(data['results']['truth']['0']);index=next(i for i,r in enumerate(c['scores']) if r['score']!=0)
    c['scores'][index]['score']*=-1
    try:verify_rates(c['scores'],c)
    except ValueError:pass
    else:raise ValueError('Changed real OOF score passed audit')
    value={'passed':True,'results':results,'wrong_toy_score_rejected':True,'wrong_fold_rejected':True,
        'changed_real_score_rejected':True,'crossfit_sha256':file_sha256(path),
        'script_sha256':file_sha256(Path(__file__)),'scope':'Saved score arithmetic and recorded folds;not actual training-membership or scientific independence certification'}
    out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
