"""Independent integer arithmetic for the registered three-encoding OOF panel.

No inference, fitting, producer rate function, or bootstrap is called. Saved LP
states and fold records are checked, not independently re-solved or certified.
"""

import argparse
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
from itertools import combinations
import json
import math
from pathlib import Path
import runpy

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT


def recount(rows, variants):
    groups = defaultdict(list); seen = set(); metadata = {}
    for row in rows:
        key = tuple(row[k] for k in ('domain', 'scene', 'condition', 'variant', 'src'))
        if (key in seen or row['role'] != 'fit' or row['label'] not in ('REAL', 'FAKE')
                or row['variant'] not in variants or not math.isfinite(row['score'])):
            raise ValueError('Invalid panel OOF identity/score')
        seen.add(key); source = key[0], key[-1]; label = row['scene'], row['label']
        if source in metadata and metadata[source] != label: raise ValueError('Conflicting panel source metadata')
        metadata[source] = label; groups[key[:4]].append(row)
        if row['scene'] != 'all': groups[(key[0], 'all', *key[2:4])].append(row)
    metrics = {}
    for key, records in groups.items():
        rates = []
        for label in ('REAL', 'FAKE'):
            selected = [r for r in records if r['label'] == label]
            if not selected: raise ValueError('Missing panel class')
            rates.append(Fraction(sum((r['score'] > 0) == (label == 'FAKE') for r in selected), len(selected)))
        metrics['/'.join(key)] = str(sum(rates)/2)
    pairs = {}
    def pair(a, b, name):
        if {r['src'] for r in groups[a]} != {r['src'] for r in groups[b]}: raise ValueError('Panel pair mismatch')
        pairs['/'.join(name)] = str(Fraction(metrics['/'.join(a)])-Fraction(metrics['/'.join(b)]))
    for domain, scene, condition, variant in groups:
        if condition != 'original':
            pair((domain, scene, 'original', variant), (domain, scene, condition, variant),
                 (domain, scene, 'original>'+condition, variant))
    for domain, scene, condition in sorted({k[:3] for k in groups}):
        for before, after in combinations(variants, 2):
            pair((domain, scene, condition, before), (domain, scene, condition, after),
                 (domain, scene, condition, before+'>'+after))
    return {'metrics': metrics, 'paired_drops': pairs, 'minimum_all_scope_ba': str(min(map(Fraction, metrics.values()))),
            'maximum_all_scope_drop': str(max(map(Fraction, pairs.values())))}


def verify(rows, quoted, variants):
    actual = recount(rows, variants)
    if any(actual[k] != quoted[k] for k in actual): raise ValueError('Panel OOF integer arithmetic differs')
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args(); output = args.directory/'calibrated_oof_audit.json'
    if output.exists(): raise FileExistsError('Preserve panel audit')
    variants = ('raw', 'jpeg90_444_after_resize256', 'jpeg60_420_after_resize256')
    toy = [dict(domain='rr', scene='all', src=label, label=label, role='fit', condition=c, variant=v,
                score=1. if label == 'FAKE' else -1.)
           for label in ('REAL', 'FAKE') for c in ('original', 'transfer', 'redigital') for v in variants]
    expected = recount(toy, variants)
    if expected['minimum_all_scope_ba'] != '1' or expected['maximum_all_scope_drop'] != '0':
        raise ValueError('Known panel count plant failed')
    bad = deepcopy(toy); bad[-1]['score'] *= -1
    try: verify(bad, expected, variants)
    except ValueError: pass
    else: raise ValueError('Known changed strong-view score accepted')
    receipt = json.loads((args.directory/'iteration.json').read_text()); path = args.directory/'crossfit.json'
    if file_sha256(path) != receipt['crossfit_sha256']: raise ValueError('Panel OOF bytes changed')
    for name, sha in receipt['code_pins'].items():
        if file_sha256(REPO_ROOT/name) != sha: raise ValueError('Panel producer changed')
    data = json.loads(path.read_text()); panel = data['source_panel']
    if tuple(panel['variants']) != variants or panel['sources'] != 1260 or panel['conditions'] != 3 or panel['folds'] != 5:
        raise ValueError('Audit only covers registered three-encoding panel')
    fold_tools = runpy.run_path(str(REPO_ROOT/'tools/audit_source_crossfit.py'))
    role_tools = runpy.run_path(str(REPO_ROOT/'tools/audit_calibrated_source_cv.py'))
    results = {}
    for mode, candidates in data['results'].items():
        if set(candidates) != {'0.0001', '0.001', '0.01', '0.1'}: raise ValueError('Panel parameter grid changed')
        counted = {}
        for parameter, record in candidates.items():
            if len(record['scores']) != 11340 or len(record['fit_diagnostics']) != 5: raise ValueError('Panel OOF count')
            rates = verify(record['scores'], record, variants)
            if len(rates['metrics']) != 45 or len(rates['paired_drops']) != 75: raise ValueError('Panel metric/pair count')
            sources = fold_tools['verify_folds'](record['scores'], data['fold_sources'])
            if len(sources) != 1260 or any(len(v) != 9 for v in sources.values()): raise ValueError('Panel source views')
            role_tools['verify_calibration'](record['scores'], data['fold_sources'])
            for held, d in enumerate(record['fit_diagnostics']):
                if (d['held_fold'] != held or d['calibration_fold'] != (held+1) % 5
                        or d['fit_records'] != 6804 or d['sources'] != 756 or d['calibration_records'] != 2268
                        or d['calibration_source_count'] != 252 or d['held_source_count'] != 252
                        or d['status'] != 0 or not d['certificate']['passed'] or not d['maximum_source']
                        or d['penalty_l1'] != float(parameter)):
                    raise ValueError('Recorded panel fit/cal/held or LP diagnostic differs')
            counted[parameter] = rates
        feasible = [k for k in counted if Fraction(counted[k]['minimum_all_scope_ba']) >= Fraction(4, 5)]
        chosen = min(feasible, key=lambda k: (Fraction(counted[k]['maximum_all_scope_drop']),
            -Fraction(counted[k]['minimum_all_scope_ba']), Fraction(k))) if feasible else min(counted,
            key=lambda k: (-Fraction(counted[k]['minimum_all_scope_ba']), Fraction(counted[k]['maximum_all_scope_drop']), Fraction(k)))
        if chosen != data['chosen'][mode][0]: raise ValueError('Panel parameter selection differs')
        results[mode] = {'passed': True, 'parameters': 4, 'groups_each': 45, 'pairs_each': 75, 'selected': chosen}
    bad = deepcopy(data['results']['truth']['0.001'])
    changed = next(i for i, r in enumerate(bad['scores']) if r['score'] != 0); bad['scores'][changed]['score'] *= -1
    try: verify(bad['scores'], bad, variants)
    except ValueError: pass
    else: raise ValueError('Changed real panel score accepted')
    bad = deepcopy(data['results']['truth']['0.001']['scores']); bad[0]['calibration_fold'] = bad[0]['fold']
    try: role_tools['verify_calibration'](bad, data['fold_sources'])
    except ValueError: pass
    else: raise ValueError('Wrong real panel cal fold accepted')
    value = {'passed': True, 'results': results, 'changed_score_rejected': True, 'wrong_calibration_fold_rejected': True,
        'crossfit_sha256': file_sha256(path), 'script_sha256': file_sha256(Path(__file__)),
        'scope': 'Independent saved-score integer arithmetic and recorded roles/LP status;not optimizer members or external science'}
    with output.open('x', encoding='utf-8') as stream: json.dump(value, stream, indent=2); stream.write('\n')
    print(json.dumps({k: v for k, v in value.items() if k != 'results'}))


if __name__ == '__main__': main()
