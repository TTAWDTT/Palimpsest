"""Recount compact LP OOF arithmetic and recorded fit/cal/held roles.

The saved LP diagnostics are checked, not independently replayed. Same-score
integer arithmetic and source records cannot certify unseen-image performance.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True); args = parser.parse_args()
    output = args.directory/'calibrated_oof_audit.json'
    if output.exists(): raise FileExistsError('Preserve calibrated margin audit')
    rates_tool = runpy.run_path(str(REPO_ROOT/'tools/audit_source_crossfit.py'))
    roles_tool = runpy.run_path(str(REPO_ROOT/'tools/audit_calibrated_source_cv.py'))
    receipt = json.loads((args.directory/'iteration.json').read_text()); path = args.directory/'crossfit.json'
    if file_sha256(path) != receipt['crossfit_sha256']: raise ValueError('Margin OOF bytes changed')
    for name, sha in receipt['code_pins'].items():
        if file_sha256(REPO_ROOT/name) != sha: raise ValueError('Margin source changed')
    data = json.loads(path.read_text()); fold_sources = data['fold_sources']; results = {}
    for mode, candidates in data['results'].items():
        if set(candidates) != {'0.0001', '0.001', '0.01', '0.1'}: raise ValueError('Margin grid differs')
        counted = {}
        for penalty, record in candidates.items():
            if len(record['scores']) != 7560 or len(record['fit_diagnostics']) != 5: raise ValueError('OOF size differs')
            rates = rates_tool['verify_rates'](record['scores'], record)
            if len(rates['metrics']) != 30 or len(rates['paired_drops']) != 35: raise ValueError('OOF coverage differs')
            sources = rates_tool['verify_folds'](record['scores'], fold_sources)
            if len(sources) != 1260 or any(len(v) != 6 for v in sources.values()): raise ValueError('OOF source coverage')
            roles_tool['verify_calibration'](record['scores'], fold_sources)
            for held, d in enumerate(record['fit_diagnostics']):
                if (d['held_fold'] != held or d['calibration_fold'] != (held+1)%5
                        or d['fit_records'] != 4536 or d['sources'] != 756
                        or d['calibration_records'] != 1512 or d['calibration_source_count'] != 252
                        or d['held_source_count'] != 252 or not d['certificate']['passed']
                        or d['status'] != 0 or not d['maximum_source'] or d['penalty_l1'] != float(penalty)):
                    raise ValueError('Recorded LP role/solver diagnostic differs')
            counted[penalty] = rates
        feasible = [k for k in counted if Fraction(counted[k]['minimum_all_scope_ba']) >= Fraction(4, 5)]
        chosen = min(feasible, key=lambda k: (Fraction(counted[k]['maximum_all_scope_drop']),
                     -Fraction(counted[k]['minimum_all_scope_ba']), Fraction(k))) if feasible else min(counted,
            key=lambda k: (-Fraction(counted[k]['minimum_all_scope_ba']), Fraction(counted[k]['maximum_all_scope_drop']), Fraction(k)))
        if chosen != data['chosen'][mode][0]: raise ValueError('Margin penalty choice differs')
        results[mode] = {'passed': True, 'penalties': 4, 'groups_each': 30, 'pairs_each': 35, 'selected': chosen}
    bad = deepcopy(data['results']['truth']['0.001'])
    changed = next(i for i, r in enumerate(bad['scores']) if r['score'] != 0); bad['scores'][changed]['score'] *= -1
    try: rates_tool['verify_rates'](bad['scores'], bad)
    except ValueError: pass
    else: raise ValueError('Changed score accepted')
    bad = deepcopy(data['results']['truth']['0.001']['scores']); bad[0]['calibration_fold'] = bad[0]['fold']
    try: roles_tool['verify_calibration'](bad, fold_sources)
    except ValueError: pass
    else: raise ValueError('Wrong calibration fold accepted')
    value = {'passed': True, 'results': results, 'changed_score_rejected': True, 'wrong_calibration_fold_rejected': True,
        'crossfit_sha256': file_sha256(path), 'script_sha256': file_sha256(Path(__file__)),
        'scope': 'Saved roles/LP diagnostics and independent integer arithmetic;not optimizer members or scientific independence'}
    with output.open('x', encoding='utf-8') as stream: json.dump(value, stream, indent=2); stream.write('\n')
    print(json.dumps({k: v for k, v in value.items() if k != 'results'}))


if __name__ == '__main__': main()
