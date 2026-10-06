"""Exact two-point metric counterexample to the printed min/max set identities."""

from fractions import Fraction
import itertools
from pathlib import Path

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR, REPO_ROOT
from experiments.origin_detection.phase_statistics.run_iteration import write_json


def predicates(a, b):
    left, right = a > 1, b > 1
    return {'union': left or right, 'intersection': left and right,
            'minimum': min(a, b) > 1, 'maximum': max(a, b) > 1}


def check_expected(actual, expected):
    if actual != expected:
        raise ValueError('Known exact metric witness differs')


def main():
    output = WORK_DIR/'robust_statistics/perturbation_review/logic.json'
    if output.exists():
        raise FileExistsError('Preserve first source-logic audit')
    witness = predicates(Fraction(2), Fraction(1, 2))
    expected = {'union': True, 'intersection': False, 'minimum': False, 'maximum': True}
    check_expected(witness, expected)
    try:
        check_expected(witness, {**expected, 'union': False})
    except ValueError:
        corrupted_rejected = True
    else:
        raise ValueError('Changed expected union not caught')
    cases = []
    for a, b in itertools.product((Fraction(1, 2), Fraction(1), Fraction(2)), repeat=2):
        p = predicates(a, b)
        if p['minimum'] != p['intersection'] or p['maximum'] != p['union']:
            raise ValueError('Corrected set identity failed')
        cases.append({'a': str(a), 'b': str(b), **p,
                      'printed_union_failed': p['union'] != p['minimum'],
                      'printed_intersection_failed': p['intersection'] != p['maximum']})
    if sum(c['printed_union_failed'] for c in cases) != 4 or sum(c['printed_intersection_failed'] for c in cases) != 4:
        raise ValueError('Unexpected preregistered failure count')
    pins = {str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in
            (Path(__file__), Path(__file__).parent/'README.md')}
    write_json(output, {'witness': witness, 'cases': cases, 'corrupted_expected_rejected': corrupted_rejected,
        'code_pins': pins, 'pdf_sha256': file_sha256(output.parent/'minder.pdf'),
        'metric_validity': 'Any strictly positive distance on two points,with zero diagonal and symmetry,is a metric',
        'scope': 'Counterexample to printed set identities;not falsification of empirical algorithm results'})
    print('9 exact cases;4 failures of each printed identity;corrected identities and changed-value refusal passed')


if __name__ == '__main__':
    main()
