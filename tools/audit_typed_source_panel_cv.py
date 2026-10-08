"""Saved-score/role audit with explicit representation dimensions and basis count.

The old fourteen-term auditor stays immutable. Arithmetic uses its existing
independent integer/fold checkers; no producer evaluator/optimizer is replayed.
"""

import argparse
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import runpy

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--mapped-dimensions', type=int, required=True)
    parser.add_argument('--basis-count', type=int, required=True)
    parser.add_argument('--allow-missing-basis-count', action='store_true')
    parser.add_argument('--parameters', nargs=4, default=('0', '0.1', '1', '10'))
    args = parser.parse_args()
    if (args.mapped_dimensions < 1 or args.basis_count < 1 or len(set(args.parameters)) != 4
            or any(Fraction(x) < 0 for x in args.parameters)):
        raise ValueError('Invalid explicit audit schema')
    out = args.directory/'calibrated_oof_audit.json'
    if out.exists():
        raise FileExistsError('Preserve audit')
    helpers = {}
    for key, name in (('count', 'audit_source_panel_cv.py'), ('fold', 'audit_source_crossfit.py'),
                      ('role', 'audit_calibrated_source_cv.py')):
        helpers[key] = runpy.run_path(str(REPO_ROOT/'tools'/name))
    path = args.directory/'crossfit.json'
    receipt = json.loads((args.directory/'iteration.json').read_text())
    if file_sha256(path) != receipt['crossfit_sha256']:
        raise ValueError('Saved CV changed')
    for name, expected in receipt['code_pins'].items():
        if file_sha256(REPO_ROOT/name) != expected:
            raise ValueError('Executed producer source changed')
    data = json.loads(path.read_text()); panel = data['source_panel']
    variants = ('raw', 'jpeg90_444_after_resize256', 'jpeg60_420_after_resize256')
    if (tuple(panel['variants']) != variants or panel['sources'] != 1260
            or panel['conditions'] != 3 or panel['folds'] != 5):
        raise ValueError('Wrong registered panel')
    results = {}; method_tags_absent = 0; basis_counts_absent = 0
    for mode, candidates in data['results'].items():
        if set(candidates) != set(args.parameters):
            raise ValueError('Wrong parameter set')
        counts = {}
        for parameter, saved in candidates.items():
            if len(saved['scores']) != 11340 or len(saved['fit_diagnostics']) != 5:
                raise ValueError('Wrong CV cardinality')
            rates = helpers['count']['verify'](saved['scores'], saved, variants)
            if len(rates['metrics']) != 45 or len(rates['paired_drops']) != 75:
                raise ValueError('Wrong saved metric coverage')
            sources = helpers['fold']['verify_folds'](saved['scores'], data['fold_sources'])
            if len(sources) != 1260 or any(len(rows) != 9 for rows in sources.values()):
                raise ValueError('Wrong source view coverage')
            helpers['role']['verify_calibration'](saved['scores'], data['fold_sources'])
            for held, d in enumerate(saved['fit_diagnostics']):
                gradient = d['maximum_absolute_gradient']
                if (d['held_fold'] != held or d['calibration_fold'] != (held+1)%5
                        or d['fit_records'] != 6804 or d['sources'] != 756
                        or d['calibration_records'] != 2268 or d['calibration_source_count'] != 252
                        or d['held_source_count'] != 252 or not math.isfinite(gradient) or gradient > 1e-5
                        or d['consistency_strength'] != float(parameter)
                        or d['mapped_dimensions'] != args.mapped_dimensions):
                    raise ValueError('Recorded role/convergence/schema differs')
                basis = d.get('basis_count')
                if basis is None and args.allow_missing_basis_count:
                    basis_counts_absent += 1
                elif basis != args.basis_count:
                    raise ValueError('Recorded basis count differs or is absent')
                method = d.get('readout_method')
                if method is None:
                    method_tags_absent += 1
                elif method != 'L2 entropy-smoothed source logistic plus score variance':
                    raise ValueError('Unexpected recorded method tag')
            counts[parameter] = rates
        feasible = [k for k in counts if Fraction(counts[k]['minimum_all_scope_ba']) >= Fraction(4, 5)]
        chosen = min(feasible, key=lambda k: (Fraction(counts[k]['maximum_all_scope_drop']),
            -Fraction(counts[k]['minimum_all_scope_ba']), Fraction(k))) if feasible else min(counts,
            key=lambda k: (-Fraction(counts[k]['minimum_all_scope_ba']), Fraction(counts[k]['maximum_all_scope_drop']), Fraction(k)))
        if chosen != data['chosen'][mode][0]:
            raise ValueError('Fit-only selected parameter differs')
        results[mode] = {'passed': True, 'selected': chosen, 'groups_each': 45, 'pairs_each': 75}
    first = args.parameters[0]; saved = data['results']['truth'][first]
    bad = deepcopy(saved['scores'])
    index = next(i for i, row in enumerate(bad) if row['score'] != 0)
    bad[index]['score'] *= -1
    try:
        helpers['count']['verify'](bad, saved, variants)
    except ValueError:
        pass
    else:
        raise ValueError('Wrong real score accepted')
    bad = deepcopy(saved['scores']); bad[0]['calibration_fold'] = bad[0]['fold']
    try:
        helpers['role']['verify_calibration'](bad, data['fold_sources'])
    except ValueError:
        pass
    else:
        raise ValueError('Wrong calibration fold accepted')
    value = {'passed': True, 'results': results, 'mapped_dimensions': args.mapped_dimensions,
        'basis_count': args.basis_count, 'parameters': args.parameters,
        'method_tags_absent': method_tags_absent,
        'basis_counts_absent': basis_counts_absent,
        'allow_missing_basis_count': args.allow_missing_basis_count,
        'changed_score_rejected': True, 'wrong_calibration_fold_rejected': True,
        'crossfit_sha256': file_sha256(path), 'script_sha256': file_sha256(Path(__file__)),
        'scope': 'Saved arithmetic/roles/schema/gradient records;absent method tags not reconstructed;not objective certification/optimizer replay/scientific independence'}
    out.write_text(json.dumps(value, indent=2)+'\n')
    print({'passed': True, 'mapped_dimensions': args.mapped_dimensions, 'basis_count': args.basis_count})


if __name__ == '__main__':
    main()
