"""Scored development views: exact BA boundaries and complete source pairs.

No fitting, threshold selection or independence certification happens here.
Scores supplied by the caller are margins (threshold already subtracted).
"""

from collections import defaultdict
from fractions import Fraction
from itertools import combinations
import math

from .classification import evaluate
from .pairing import paired_change


def exact_ba(records):
    rates=[]
    for label in ('REAL','FAKE'):
        selected=[r for r in records if r['label']==label]
        if not selected: raise ValueError('Missing class in scored view')
        rates.append(Fraction(sum((float(r['score'])>0)==(label=='FAKE') for r in selected),len(selected)))
    return sum(rates)/2


def evaluate_views(rows, *, variant_order, minimum_ba=.8, maximum_drop=.02, reference='original'):
    if not rows or not variant_order or len(set(variant_order))!=len(variant_order): raise ValueError('Invalid view set')
    source_labels={}; seen=set(); groups=defaultdict(list)
    for r in rows:
        if r['role']!='selection' or r['label'] not in ('REAL','FAKE') or not math.isfinite(float(r['score'])):
            raise ValueError('Invalid scored selection role/label/margin')
        identity=tuple(r[k] for k in ('domain','scene','condition','variant','src'))
        if identity in seen or r['variant'] not in variant_order: raise ValueError('Duplicate/unregistered scored view')
        seen.add(identity)
        source=(r['domain'],r['src'])
        if source in source_labels and source_labels[source]!=(r['label'],r['scene']): raise ValueError('Conflicting source metadata')
        source_labels[source]=(r['label'],r['scene'])
        groups[identity[:4]].append(r)
        if r['scene']!='all': groups[(r['domain'],'all',r['condition'],r['variant'])].append(r)
    bases={key[:3] for key in groups}
    if set(groups)!={(*base,v) for base in bases for v in variant_order}: raise ValueError('Incomplete encoding views')
    metrics={'/'.join(key):evaluate(records,'score') for key,records in groups.items()}
    pairs={}
    def add_pair(name, before, after):
        first={r['src']:r for r in before};second={r['src']:r for r in after}
        if set(first)!=set(second): raise ValueError('Incomplete same-source pair')
        result=paired_change(first,second)
        result['exact_ba_drop']=str(exact_ba(before)-exact_ba(after));pairs[name]=result
    for key,records in groups.items():
        domain,scene,condition,variant=key
        if condition!=reference:
            ref=(domain,scene,reference,variant)
            if ref not in groups: raise ValueError('Missing reference view')
            add_pair('/'.join((domain,scene,reference+'>'+condition,variant)),groups[ref],records)
    for domain,scene,condition in sorted(bases):
        for before,after in combinations(variant_order,2):
            add_pair('/'.join((domain,scene,condition,before+'>'+after)),groups[(domain,scene,condition,before)],groups[(domain,scene,condition,after)])
    if not pairs: raise ValueError('No robustness comparisons')
    all_ba=[exact_ba(r) for k,r in groups.items() if k[1]=='all']
    scene_ba=[exact_ba(r) for k,r in groups.items() if k[1]!='all']
    all_drop=[Fraction(v['exact_ba_drop']) for k,v in pairs.items() if k.split('/')[1]=='all']
    worst=max(pairs,key=lambda k:Fraction(pairs[k]['exact_ba_drop']))
    maxdrop=Fraction(pairs[worst]['exact_ba_drop'])
    return {'metrics':metrics,'pairs':pairs,'minimum_domain_ba':float(min(all_ba)),
            'minimum_scene_ba':float(min(scene_ba)) if scene_ba else None,
            'maximum_domain_drop':float(max(all_drop)),'maximum_any_scene_drop':float(maxdrop),
            'worst_pair':worst,'development_absolute_ba':min(all_ba)>=Fraction(str(minimum_ba)),
            'development_all_scope_drop':maxdrop<=Fraction(str(maximum_drop)),
            'independent_real_validation':False,'scope':'Scored exposed development,not independent validation'}
