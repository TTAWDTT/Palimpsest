"""One-off exact controls for a proposed phase-energy cue, not an AI detector.

Compare direct Fraction indexing and NumPy cyclic differences. The fixtures
have periodic boundaries; they do not simulate a real display or camera.
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path

import numpy as np


def exact_phase_energy(values):
    x = tuple(Fraction(v) for v in values)
    if len(x) < 4 or len(x) % 2:
        raise ValueError('Expected an even periodic signal with length >=4')
    energies = tuple(sum((x[(i+1) % len(x)]-x[i])**2 for i in range(p, len(x), 2))
        / (len(x)//2) for p in (0, 1))
    total = sum(energies)
    return energies, None if total == 0 else (energies[1]-energies[0])/total


def vector_phase_energy(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 4 or len(x) % 2 or not np.isfinite(x).all():
        raise ValueError('Expected a finite even periodic signal')
    delta = np.roll(x, -1)-x
    energies = np.mean(delta.reshape(-1, 2)**2, axis=0)
    total = float(energies.sum())
    return energies, None if total == 0 else float((energies[1]-energies[0])/total)


def verify(values, expected_energies, expected_contrast):
    energies, contrast = exact_phase_energy(values)
    if energies != tuple(Fraction(v) for v in expected_energies) or contrast != expected_contrast:
        raise ValueError('Known phase-energy answer mismatch')
    actual, score = vector_phase_energy(values)
    if not np.array_equal(actual, np.array([float(v) for v in energies])) or score != (
            None if contrast is None else float(contrast)):
        raise ValueError('Exact/vector phase-energy routes differ')
    return {'signal': [str(Fraction(v)) for v in values], 'energies': [str(v) for v in energies],
        'contrast': None if contrast is None else str(contrast), 'vector_agreement': True}


def controls():
    latent = tuple(map(Fraction, (-1, 0, 1, 0)))
    repeated = tuple(v for v in latent for _ in range(2))
    filtered = tuple((repeated[i]+repeated[i-1])/2 for i in range(len(repeated)))
    # Expected values are independently written before the numerical route runs.
    fixtures = (
        ('nearest_repeat', repeated, (0, 1), Fraction(1)),
        ('brightness_shift', tuple(v+3 for v in repeated), (0, 1), Fraction(1)),
        ('phase_shift', repeated[-1:]+repeated[:-1], (1, 0), Fraction(-1)),
        ('two_tap_filter', filtered, (Fraction(1, 4), Fraction(1, 4)), Fraction(0)),
        ('constant', (Fraction(2),)*8, (0, 0), None),
    )
    answers = {name: verify(values, energies, contrast) for name, values, energies, contrast in fixtures}
    ordinary_resize = tuple(latent[i//2] for i in range(8))
    if ordinary_resize != repeated:
        raise ValueError('Equivalent ordinary upsampling control changed')
    retained_energy = sum(v*v for v in filtered)/len(filtered)
    if retained_energy != Fraction(3, 8):
        raise ValueError('Filter erasure must retain nonzero signal energy')
    caught = 0
    for values, energies, wrong in ((filtered, (Fraction(1, 4),)*2, Fraction(1)),
                                   (repeated, (0, 1), Fraction(0))):
        try:
            verify(values, energies, wrong)
        except ValueError:
            caught += 1
        else:
            raise ValueError('Deliberately wrong contrast was not caught')
    for bad in ([0, 1, 2], [0, 0], [0, 1, np.nan, 0]):
        try:
            vector_phase_energy(bad)
        except ValueError:
            caught += 1
        else:
            raise ValueError('Invalid signal was not refused')
    return {'fixtures': answers, 'positive_fixtures': len(fixtures), 'negative_controls_caught': caught,
        'ordinary_resize_matches_generator_proxy_exactly': True,
        'filtered_signal_mean_square': str(retained_energy),
        'verdict': 'The proposed contrast is operator-sensitive, not AI-specific or filter-invariant.',
        'scope': 'Finite exact periodic fixtures; not a general CNN theorem, classifier BA or physical recapture validation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = controls()
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'fixtures'}))


if __name__ == '__main__':
    main()
