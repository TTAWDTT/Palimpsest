"""Fit-role matrices with equal domains and equal matched sources.

The caller fixes variants and expected counts. No selection/threshold rows enter
returned matrices; a source has one label, with optional within-scene null.
"""

from collections import defaultdict
import random
import numpy as np


def weighted_source_arrays(rows,names,variants,*,expected_source_counts,seed,shuffled=False):
    records=sorted([r for r in rows if r['role']=='fit' and r['variant'] in variants],
                   key=lambda r:tuple(r[k] for k in ('domain','src','condition','variant')))
    sources=defaultdict(list)
    for r in records: sources[(r['domain'],r['src'])].append(r)
    domains=defaultdict(list)
    for key,selected in sorted(sources.items()):
        metadata={(r['label'],r['scene']) for r in selected}
        views={(r['condition'],r['variant']) for r in selected}
        conditions={r['condition'] for r in selected}
        if (len(metadata)!=1 or selected[0]['label'] not in ('REAL','FAKE')
                or len(views)!=len(selected) or views!={(c,v) for c in conditions for v in variants}):
            raise ValueError('Incomplete/conflicting fit source views')
        domains[key[0]].append(key)
    if {k:len(v) for k,v in domains.items()}!=expected_source_counts:
        raise ValueError('Frozen fit source/domain counts differ')
    labels={key:int(selected[0]['label']=='FAKE') for key,selected in sources.items()}
    if shuffled:
        strata=defaultdict(list)
        for key,selected in sorted(sources.items()): strata[(key[0],selected[0]['scene'])].append(key)
        rng=random.Random(seed)
        for keys in strata.values():
            values=[labels[k] for k in keys];rng.shuffle(values);labels.update(zip(keys,values))
    keys=[(r['domain'],r['src']) for r in records]
    indices={key:i for i,key in enumerate(sorted(sources))}
    x=np.array([[float(r[n]) for n in names] for r in records])
    if not np.isfinite(x).all(): raise ValueError('Nonfinite fit features')
    y=np.array([labels[k] for k in keys])
    weights=np.array([1/len(domains)/len(domains[k[0]])/len(sources[k]) for k in keys])
    return x,y,weights,np.array([indices[k] for k in keys])
