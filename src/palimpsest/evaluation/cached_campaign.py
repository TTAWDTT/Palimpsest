"""Reusable cached-readout selection and null intervals; no pixel selection."""

from collections import defaultdict

import numpy as np

from .source_readout_campaign import fit_readouts


def evaluate_cached_heads(rows, candidates, *, null_key, fit_variants, manifest_sha,
                          config, calibrate, score, interval_function):
    results,rules,scores = fit_readouts(rows,candidates,fit_variants=fit_variants,
        manifest_sha=manifest_sha,config=config,calibrate=calibrate,score=score)
    groups = defaultdict(list)
    for row in scores[null_key]:
        if row['variant']=='raw' and row['condition']!='original':
            groups[row['domain']+'/'+row['condition']].append(row)
    intervals = {}
    for key,records in groups.items():
        records.sort(key=lambda r:r['src'])
        intervals[key] = interval_function({key:np.array([r['score'] for r in records])},
            np.array([int(r['label']=='FAKE') for r in records]),config)[key]
    refused = all(v[0] > .5 for v in intervals.values())
    keys = [k for k in results if k!=null_key]
    chosen = None if refused else max(keys,key=lambda k:(min(results[k]['minimum_domain_ba'],
        results[k]['minimum_scene_ba']),-results[k]['maximum_any_scene_drop']))
    # Preserve every domain's null result, including anomalies that don't meet
    # the uniform-direction refusal rule. That rule is not an all-clear certificate.
    return results,rules,scores,intervals,chosen,refused
