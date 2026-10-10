"""Exact finite audit of reconstruction cancellation's manifold premise."""

from fractions import Fraction
from pathlib import Path

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR
from experiments.origin_detection.phase_statistics.run_iteration import write_json


def fractional(a, b):
    def projection(point):
        return point[0], Fraction(0)

    def reconstruct(point):
        projected = projection(point)
        return projected[0] + a, projected[1] + b

    source = Fraction(3, 7), Fraction(0)
    first = reconstruct(source)
    second = reconstruct(first)
    error1 = tuple(abs(x-y) for x, y in zip(source, first))
    error2 = tuple(abs(x-y) for x, y in zip(first, second))
    return tuple(x-y for x, y in zip(error1, error2))


def integer_route(a, b):
    # Common-denominator coordinate composition, without Fraction operations.
    points = [(3, 0)]
    for _ in range(2):
        old = points[-1]
        points.append((old[0] + a, b))
    residuals = [tuple(abs(points[i][j] - points[i+1][j]) for j in range(2)) for i in range(2)]
    return tuple(str(Fraction(residuals[0][j] - residuals[1][j], 7)) for j in range(2))


def main():
    directory = WORK_DIR / 'robust_statistics/did_review'
    destination = directory / 'counterexample.json'
    protocol = directory / 'counterexample_protocol.json'
    if destination.exists():
        raise FileExistsError('Preserve exact cancellation evidence')
    if not protocol.exists():
        raise FileNotFoundError('Pre-computation protocol missing')
    records = []
    for a in range(-2, 3):
        for b in range(-2, 3):
            values = fractional(Fraction(a, 7), Fraction(b, 7))
            rendered = tuple(str(v) for v in values)
            if rendered != integer_route(a, b) or values != (Fraction(0), Fraction(abs(b), 7)):
                raise ValueError('Exact direct/coordinate cancellation control differs')
            records.append({'horizontal': str(Fraction(a, 7)), 'vertical': str(Fraction(b, 7)),
                            'second_error': rendered})
    known = fractional(Fraction(0), Fraction(1, 7))
    wrong_rejected = known != (Fraction(0), Fraction(0))
    if not wrong_rejected:
        raise ValueError('Wrong cancellation failed to be refused')
    write_json(destination, {'passed': True, 'cases': len(records), 'exact_matches': len(records),
        'horizontal_only_cancellations': 5, 'vertical_noncancellations': 20, 'wrong_zero_rejected': wrong_rejected,
        'records': records, 'protocol_sha256': file_sha256(protocol),
        'code_sha256': file_sha256(Path(__file__)),
        'scope': 'Finite exact composition;shared manifold/constant perturbation assumptions;'
                 'no fitted anchors;not physical pixels or full empirical refutation'})
    print('25 exact matches;5 tangential cancellations;20 normal failures;wrong zero refused', flush=True)


if __name__ == '__main__':
    main()
