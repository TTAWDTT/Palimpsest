"""Finite exact controls for phase energy and operator moments, not an AI detector.

The original energy suite is a one-off cue audit. The moments suite reuses
the population evaluator and compares matrix propagation with exhaustive
rational sign assignments. Neither suite simulates a real display or camera.
"""

import argparse
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from palimpsest.evaluation.polyphase import (
    linear_population_moments,
    periodic_upsampling_matrix,
    phase_summary,
)


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


def _enumerate_population(case):
    """Direct rational convolution of every sign assignment, no matrix reuse."""
    n, stride = case['latent_length'], case['stride']
    h = tuple(Fraction(v) for v in case['kernel'])
    mu, color = Fraction(case['latent_mean']), Fraction(case['color_previous'])
    size = n * stride
    outputs = []
    for signs in product((-1, 1), repeat=n):
        z = tuple(mu + signs[i] + color * signs[(i-1) % n] for i in range(n))
        u = tuple(z[i//stride] if i % stride == 0 else Fraction(0) for i in range(size))
        outputs.append(tuple(sum(h[k] * u[(i-k) % size] for k in range(len(h)))
                             for i in range(size)))
    count = len(outputs)
    mean = tuple(sum(x[i] for x in outputs)/count for i in range(size))
    covariance = tuple(tuple(sum((x[i]-mean[i])*(x[j]-mean[j]) for x in outputs)/count
                             for j in range(size)) for i in range(size))
    fourth = tuple(sum((x[i]-mean[i])**4 for x in outputs)/count for i in range(size))
    group = lambda x: tuple(sum(x[p::stride])/n for p in range(stride))
    adjacent = tuple(covariance[i][(i+1) % size] for i in range(size))
    differences = tuple(sum((x[(i+1) % size]-x[i])**2 for x in outputs)/count
                        for i in range(size))
    stationary = (all(mean[i] == mean[(i+1) % size] for i in range(size))
                  and all(covariance[i][j] == covariance[(i+1) % size][(j+1) % size]
                          for i in range(size) for j in range(size)))
    summary = {'means': group(mean), 'variances': group(tuple(covariance[i][i] for i in range(size))),
               'adjacent_covariances': group(adjacent), 'difference_energies': group(differences),
               'centered_fourth_moments': group(fourth), 'second_order_stationary': stationary}
    return mean, covariance, summary, count


def _check_case(case, rtol, atol):
    mean, covariance, exact, assignments = _enumerate_population(case)
    for key, expected in case['expected'].items():
        target = expected if isinstance(expected, bool) else tuple(Fraction(v) for v in expected)
        if exact[key] != target:
            raise ValueError(f"{case['name']}: known answer mismatch for {key}")
    n = case['latent_length']
    a = periodic_upsampling_matrix([float(Fraction(v)) for v in case['kernel']], case['stride'], n)
    color = float(Fraction(case['color_previous']))
    b = np.eye(n) + color*np.roll(np.eye(n), 1, axis=0)
    predicted = linear_population_moments(a, np.full(n, float(Fraction(case['latent_mean']))), b @ b.T)
    np.testing.assert_allclose(predicted.mean, np.array(mean, dtype=float), rtol=rtol, atol=atol)
    np.testing.assert_allclose(predicted.covariance, np.array(covariance, dtype=float), rtol=rtol, atol=atol)
    summary = phase_summary(predicted, case['stride'])
    for key in ('means', 'variances', 'adjacent_covariances', 'difference_energies'):
        np.testing.assert_allclose(summary[key], np.array(exact[key], dtype=float), rtol=rtol, atol=atol)
    actual_stationary = max(summary['mean_shift_residual'], summary['covariance_shift_residual']) <= atol
    if actual_stationary != exact['second_order_stationary']:
        raise ValueError('Stationarity residual disagrees with exact population')
    return {'name':case['name'], 'sign_assignments':assignments,
            'exact':{key:(value if isinstance(value,bool) else [str(v) for v in value])
                     for key,value in exact.items()},
            'matrix_summary':summary, 'matrix_and_enumeration_agree':True}


def moment_controls(spec):
    """Full fixture path; wrong known answers must fail for every case."""
    if spec.get('schema_version') != 1:
        raise ValueError('Unsupported controls schema')
    cases = spec['fixtures']
    if len(cases) != spec['registered_cases'] or not cases:
        raise ValueError('Registered fixture denominator changed')
    if len({c['name'] for c in cases}) != len(cases):
        raise ValueError('Duplicate fixture name')
    if any(not isinstance(c['latent_length'],int) or not 2 <= c['latent_length'] <= 8
           or not isinstance(c['stride'],int) or isinstance(c['stride'],bool)
           or not 1 <= c['stride'] <= 4 for c in cases):
        raise ValueError('Exact enumeration requires small explicit integer dimensions')
    rtol, atol = spec['rtol'], spec['atol']
    if rtol != 1e-12 or atol != 1e-12:
        raise ValueError('Registered numerical tolerance changed')
    started = perf_counter()
    results = [_check_case(c,rtol,atol) for c in cases]
    caught = []
    for case in cases:
        corrupted = deepcopy(case)
        corrupted['expected']['variances'][0] = str(Fraction(case['expected']['variances'][0])+1)
        try:
            _check_case(corrupted,rtol,atol)
        except ValueError:
            caught.append(case['name'])
        else:
            raise ValueError('Wrong known variance was not caught')
    return {'fixtures':results, 'positive_fixtures':len(results),
            'negative_controls_caught':len(caught), 'corruptions':caught,
            'elapsed_seconds':perf_counter()-started, 'rtol':rtol,'atol':atol,
            'scope':'Exact finite periodic Rademacher populations plus matrix moment propagation; not real generator evidence, AI specificity, BA or recapture validation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--suite', choices=('energy','moments'), default='energy')
    parser.add_argument('--spec',type=Path,default=Path(__file__).resolve().parents[4]
                        /'configs/evaluation/polyphase_operator_controls.json')
    args = parser.parse_args()
    if args.suite == 'energy':
        result = controls()
    else:
        spec_bytes = args.spec.read_bytes()
        result = moment_controls(json.loads(spec_bytes))
        result['spec_sha256'] = sha256(spec_bytes).hexdigest()
        result['code_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
        import palimpsest.evaluation.polyphase as operator_module
        result['operator_code_sha256'] = sha256(Path(operator_module.__file__).read_bytes()).hexdigest()
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'fixtures'}))


if __name__ == '__main__':
    main()
