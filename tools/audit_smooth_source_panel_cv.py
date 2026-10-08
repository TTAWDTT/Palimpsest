"""Recount smooth-readout OOF scores; inspect recorded role/gradient gates.

Uses the separate integer recounter,never producer metrics or model fitting.
Gradient records are not an independent optimizer or physical certificate.
"""

import argparse
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import runpy

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args(); output = args.directory/'calibrated_oof_audit.json'
    if output.exists(): raise FileExistsError('Preserve smooth panel audit')
    tools = runpy.run_path(str(REPO_ROOT/'tools/audit_source_panel_cv.py'))
    fold_tools = runpy.run_path(str(REPO_ROOT/'tools/audit_source_crossfit.py'))
    role_tools = runpy.run_path(str(REPO_ROOT/'tools/audit_calibrated_source_cv.py'))
    variants = ('raw', 'jpeg90_444_after_resize256', 'jpeg60_420_after_resize256')
    toy = [dict(domain='rr', scene='all', src=label, label=label, role='fit', condition=c, variant=v,
                score=1. if label == 'FAKE' else -1.)
           for label in ('REAL', 'FAKE') for c in ('original', 'transfer', 'redigital') for v in variants]
    known = tools['recount'](toy, variants)
    if known['minimum_all_scope_ba'] != '1' or known['maximum_all_scope_drop'] != '0': raise ValueError('Known plant failed')
    bad = deepcopy(toy); bad[-1]['score'] *= -1
    try: tools['verify'](bad, known, variants)
    except ValueError: pass
    else: raise ValueError('Known changed score accepted')
    receipt = json.loads((args.directory/'iteration.json').read_text()); path = args.directory/'crossfit.json'
    if file_sha256(path) != receipt['crossfit_sha256']: raise ValueError('Smooth OOF bytes changed')
    for name, sha in receipt['code_pins'].items():
        if file_sha256(REPO_ROOT/name) != sha: raise ValueError('Smooth producer source changed')
    data = json.loads(path.read_text()); p = data['source_panel']
    if tuple(p['variants']) != variants or p['sources'] != 1260 or p['conditions'] != 3 or p['folds'] != 5:
        raise ValueError('Audit only covers registered strong panel')
    results = {}
    for mode, candidates in data['results'].items():
        if set(candidates) != {'0', '0.1', '1', '10'}: raise ValueError('Smooth strength grid differs')
        counted = {}
        for strength, record in candidates.items():
            if len(record['scores']) != 11340 or len(record['fit_diagnostics']) != 5: raise ValueError('Smooth OOF count')
            rates = tools['verify'](record['scores'], record, variants)
            if len(rates['metrics']) != 45 or len(rates['paired_drops']) != 75: raise ValueError('Smooth coverage')
            sources = fold_tools['verify_folds'](record['scores'], data['fold_sources'])
            if len(sources) != 1260 or any(len(v) != 9 for v in sources.values()): raise ValueError('Smooth source views')
            role_tools['verify_calibration'](record['scores'], data['fold_sources'])
            for held, d in enumerate(record['fit_diagnostics']):
                if (d['held_fold'] != held or d['calibration_fold'] != (held+1) % 5
                        or d['fit_records'] != 6804 or d['sources'] != 756 or d['calibration_records'] != 2268
                        or d['calibration_source_count'] != 252 or d['held_source_count'] != 252
                        or d['maximum_absolute_gradient'] > 1e-5 or d['consistency_strength'] != float(strength)
                        or d['mapped_dimensions'] != 14 or d['readout_method'] != 'L2 entropy-smoothed source logistic plus score variance'):
                    raise ValueError('Recorded smooth role/convergence differs')
            counted[strength] = rates
        feasible = [k for k in counted if Fraction(counted[k]['minimum_all_scope_ba']) >= Fraction(4, 5)]
        chosen = min(feasible, key=lambda k: (Fraction(counted[k]['maximum_all_scope_drop']),
            -Fraction(counted[k]['minimum_all_scope_ba']), Fraction(k))) if feasible else min(counted,
            key=lambda k: (-Fraction(counted[k]['minimum_all_scope_ba']), Fraction(counted[k]['maximum_all_scope_drop']), Fraction(k)))
        if chosen != data['chosen'][mode][0]: raise ValueError('Smooth strength selection differs')
        results[mode] = {'passed': True, 'strengths': 4, 'groups_each': 45, 'pairs_each': 75, 'selected': chosen}
    bad = deepcopy(data['results']['truth']['0'])
    changed = next(i for i, row in enumerate(bad['scores']) if row['score'] != 0); bad['scores'][changed]['score'] *= -1
    try: tools['verify'](bad['scores'], bad, variants)
    except ValueError: pass
    else: raise ValueError('Changed real smooth score accepted')
    bad = deepcopy(data['results']['truth']['0']['scores']); bad[0]['calibration_fold'] = bad[0]['fold']
    try: role_tools['verify_calibration'](bad, data['fold_sources'])
    except ValueError: pass
    else: raise ValueError('Wrong real cal fold accepted')
    value = {'passed': True, 'results': results, 'changed_score_rejected': True, 'wrong_calibration_fold_rejected': True,
        'crossfit_sha256': file_sha256(path), 'script_sha256': file_sha256(Path(__file__)),
        'scope': 'Saved-score arithmetic/roles/gradient diagnostics;not optimizer replay or scientific independence'}
    with output.open('x', encoding='utf-8') as stream: json.dump(value, stream, indent=2); stream.write('\n')
    print(json.dumps({k: v for k, v in value.items() if k != 'results'}))


if __name__ == '__main__': main()
