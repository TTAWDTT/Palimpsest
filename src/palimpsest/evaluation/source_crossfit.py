"""Fit-only source folds and exact out-of-fold prediction summaries."""

from collections import defaultdict
from fractions import Fraction
from hashlib import sha256
from itertools import combinations
import math

from .robust_views import exact_ba


def source_folds(records,*,folds=5,seed=20261007):
    if folds<2:raise ValueError('Need multiple source folds')
    metadata={};seen=set();strata=defaultdict(list)
    for r in records:
        if r['role']!='fit':raise ValueError('Non-fit source fold input')
        key=r['domain'],r['src'];value=r['scene'],r['label']
        identity=key+(r['condition'],r['variant'])
        if identity in seen or (key in metadata and metadata[key]!=value):
            raise ValueError('Duplicate/conflicting source fold view')
        seen.add(identity);metadata[key]=value
    for key,(scene,label) in sorted(metadata.items()):
        if label not in ('REAL','FAKE'):raise ValueError('Unknown source fold class')
        strata[key[0],scene,label].append(key)
    if not metadata:raise ValueError('Empty source fold input')
    assignment={}
    for stratum,keys in sorted(strata.items()):
        if len(keys)<folds:raise ValueError('Insufficient source fold stratum')
        ordered=sorted(keys,key=lambda k:(sha256(f'{seed}/{k[0]}/{k[1]}'.encode()).hexdigest(),k))
        assignment.update((key,index%folds) for index,key in enumerate(ordered))
    return assignment


def crossfit_rates(rows,variants):
    groups=defaultdict(list);seen=set();metadata={}
    for r in rows:
        key=tuple(r[k] for k in ('domain','scene','condition','variant','src'))
        if (key in seen or r['role']!='fit' or r['label'] not in ('REAL','FAKE')
                or r['variant'] not in variants or not math.isfinite(r['score'])):
            raise ValueError('Invalid out-of-fold scored row')
        source=key[0],key[-1];value=r['scene'],r['label']
        if source in metadata and metadata[source]!=value:raise ValueError('Conflicting OOF source metadata')
        metadata[source]=value
        seen.add(key);groups[key[:4]].append(r)
        if r['scene']!='all':groups[(key[0],'all',*key[2:4])].append(r)
    metrics={key:exact_ba(records) for key,records in groups.items()};drops={}
    for key,records in groups.items():
        domain,scene,condition,variant=key
        if condition!='original':
            ref=(domain,scene,'original',variant)
            if ref not in groups or {r['src'] for r in records}!={r['src'] for r in groups[ref]}:
                raise ValueError('Incomplete out-of-fold physical pair')
            drops[(domain,scene,'original>'+condition,variant)]=metrics[ref]-metrics[key]
    for domain,scene,condition in sorted({k[:3] for k in groups}):
        for before,after in combinations(variants,2):
            a,b=(domain,scene,condition,before),(domain,scene,condition,after)
            if a not in groups or b not in groups or {r['src'] for r in groups[a]}!={r['src'] for r in groups[b]}:
                raise ValueError('Incomplete out-of-fold encoding pair')
            drops[(domain,scene,condition,before+'>'+after)]=metrics[a]-metrics[b]
    if not drops:raise ValueError('No out-of-fold pair')
    return {'minimum_all_scope_ba':str(min(metrics.values())),
            'maximum_all_scope_drop':str(max(drops.values())),
            'metrics':{'/'.join(k):str(v) for k,v in metrics.items()},
            'paired_drops':{'/'.join(k):str(v) for k,v in drops.items()}}


def choose_strength(results):
    feasible={k:v for k,v in results.items() if Fraction(v['minimum_all_scope_ba'])>=Fraction(4,5)}
    if feasible:
        chosen=min(feasible,key=lambda k:(Fraction(feasible[k]['maximum_all_scope_drop']),
                   -Fraction(feasible[k]['minimum_all_scope_ba']),float(k)))
        reason='Among OOF BA>=80%,smallest maximum paired decline;then BA;then smaller strength'
    else:
        chosen=max(results,key=lambda k:(Fraction(results[k]['minimum_all_scope_ba']),
                   -Fraction(results[k]['maximum_all_scope_drop']),-float(k)))
        reason='No OOF BA>=80%;highest minimum BA fallback;not goal achievement'
    return chosen,reason
