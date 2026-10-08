"""Saved-score/role audit with explicit representation dimensions and basis count.

The old fourteen-term auditor stays immutable. Arithmetic uses its existing
independent integer/fold checkers; no producer evaluator/optimizer is replayed.
"""

import argparse
from copy import deepcopy
from fractions import Fraction
import json
import math
import hashlib
import re
import subprocess
from pathlib import Path
import runpy

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT


def verify_stage_partition(rows, folds, diagnostic, held):
    """Reconstruct prescribed members from saved identities/labels, no fitter."""
    training = {}
    for row in rows:
        key = row['domain'], row['src']
        if folds['/'.join(key)] not in (held, (held+1)%5):
            training[key] = row['domain'], row['scene'], int(row['label'] == 'FAKE')
    strata = {}
    for key, stratum in sorted(training.items()):
        strata.setdefault(stratum, []).append(key)
    expected_basis = set(); round_up = False
    for _, members in sorted(strata.items()):
        if len(members) < 2:
            raise ValueError('Recorded split stratum too small')
        def rank(key):
            payload = json.dumps([20261008, *key], ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            return hashlib.sha256(payload).hexdigest(), key
        count = len(members)//2
        if len(members)%2:
            count += int(round_up); round_up = not round_up
        expected_basis.update(sorted(members, key=rank)[:count])
    basis = [tuple(key) for key in diagnostic['basis_source_keys']]
    head = [tuple(key) for key in diagnostic['readout_source_keys']]
    if (diagnostic['partition_seed'] != 20261008 or len(set(basis)) != len(basis)
            or len(set(head)) != len(head) or set(basis) != expected_basis
            or set(head) != set(training)-expected_basis or len(basis) != len(head)
            or diagnostic['input_source_count'] != len(training)
            or diagnostic['basis_source_count'] != len(basis)
            or diagnostic['basis_fit_records'] != 9*len(basis)
            or diagnostic['input_fit_records'] != 9*len(training)):
        raise ValueError('Recorded source-stage partition differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--mapped-dimensions', type=int, required=True)
    parser.add_argument('--basis-count', type=int, required=True)
    parser.add_argument('--allow-missing-basis-count', action='store_true')
    parser.add_argument('--strength-field', choices=('consistency_strength', 'ordering_strength'), default='consistency_strength')
    parser.add_argument('--readout-method', default='L2 entropy-smoothed source logistic plus score variance')
    parser.add_argument('--recorded-auditor-ref', help='Exact Git commit of this auditor if the producer pinned its earlier bytes')
    parser.add_argument('--parameters', nargs=4, default=('0', '0.1', '1', '10'))
    parser.add_argument('--source-split', action='store_true', help='Explicit equal basis/readout source-stage schema')
    args = parser.parse_args()
    if args.recorded_auditor_ref is not None and not re.fullmatch('[0-9a-f]{40}', args.recorded_auditor_ref):
        raise ValueError('Historical auditor ref must be a full Git SHA')
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
            own_name = str(Path(__file__).resolve().relative_to(REPO_ROOT)).replace('\\', '/')
            if args.recorded_auditor_ref is None or name.replace('\\', '/') != own_name:
                raise ValueError('Executed producer source changed')
            blob = subprocess.check_output(['git', 'show', args.recorded_auditor_ref+':'+own_name], cwd=REPO_ROOT)
            if hashlib.sha256(blob).hexdigest() != expected:
                raise ValueError('Historical auditor bytes do not match recorded producer pin')
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
                        or d['fit_records'] != (3402 if args.source_split else 6804)
                        or d['sources'] != (378 if args.source_split else 756)
                        or d['calibration_records'] != 2268 or d['calibration_source_count'] != 252
                        or d['held_source_count'] != 252 or not math.isfinite(gradient) or gradient > 1e-5
                        or d[args.strength_field] != float(parameter)
                        or d['mapped_dimensions'] != args.mapped_dimensions):
                    raise ValueError('Recorded role/convergence/schema differs')
                if args.source_split:
                    verify_stage_partition(saved['scores'], data['fold_sources'], d, held)
                basis = d.get('basis_count')
                if basis is None and args.allow_missing_basis_count:
                    basis_counts_absent += 1
                elif basis != args.basis_count:
                    raise ValueError('Recorded basis count differs or is absent')
                method = d.get('readout_method')
                if method is None:
                    method_tags_absent += 1
                elif method != args.readout_method:
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
        'strength_field': args.strength_field, 'readout_method': args.readout_method,
        'recorded_auditor_ref': args.recorded_auditor_ref,
        'source_split_schema': args.source_split,
        'changed_score_rejected': True, 'wrong_calibration_fold_rejected': True,
        'crossfit_sha256': file_sha256(path), 'script_sha256': file_sha256(Path(__file__)),
        'scope': 'Saved arithmetic/roles/schema/gradient records;absent method tags not reconstructed;not objective certification/optimizer replay/scientific independence'}
    out.write_text(json.dumps(value, indent=2)+'\n')
    print({'passed': True, 'mapped_dimensions': args.mapped_dimensions, 'basis_count': args.basis_count})


if __name__ == '__main__':
    main()
