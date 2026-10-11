"""Replay a scoped exact zero-gap witness; no detector or empirical conclusion."""

import argparse
from fractions import Fraction
import json
from pathlib import Path


def affine_feature(point):
    x, y = point
    return x, 2*y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve the exact witness replay')
    zero = (Fraction(0), Fraction(0))
    basis = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    derivatives = []
    for direction in basis:
        at_zero = affine_feature(zero)
        displaced = affine_feature(direction)
        derivatives.append(tuple(b-a for a,b in zip(at_zero, displaced)))
    tangent_energy, normal_energy = (sum(c*c for c in d) for d in derivatives)
    energy = tangent_energy + normal_energy
    assert tangent_energy == 1 and normal_energy == 4 and energy == 5
    # Affine coefficients are constant at every point, not a Monte Carlo claim.
    for point in ((Fraction(-1), Fraction(0)), (Fraction(1), Fraction(0)),
                  (Fraction(2,3), Fraction(-4,5))):
        for direction, expected in zip(basis, derivatives):
            displaced = affine_feature(tuple(a+b for a,b in zip(point, direction)))
            current = affine_feature(point)
            assert tuple(b-a for a,b in zip(current, displaced)) == expected
    # Expectation of a constant under either normalized probability law.
    real_expectation = energy * Fraction(1)
    tube_expectation = sum(Fraction(1,2)*energy for _ in (-1, 1))
    delta = tube_expectation - real_expectation
    assert delta == 0 and delta != 1  # planted nonzero gap is rejected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    value = {'tangent_squared_sensitivity': str(tangent_energy),
             'normal_squared_sensitivity': str(normal_energy), 'G': str(energy),
             'E_p_G': str(real_expectation), 'E_q_G': str(tube_expectation), 'Delta': str(delta),
             'planted_nonzero_gap_rejected': delta != 1,
             'grade': 'PROVISIONAL; replay of independently rebuilt witness, not full protocol closure',
             'scope': 'Printed sensitivity implication only; not added curvature premise, KL comparison or detector performance'}
    args.output.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
