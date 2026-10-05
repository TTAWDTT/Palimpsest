"""AUC can stay exact while a fixed decision threshold loses useful accuracy.

Standalone analytic counterexample with independent density quadrature. This
does not estimate any real dataset or certify the project's eventual detector.
"""

import argparse
from math import exp, pi, sqrt
from pathlib import Path
import json

from scipy.integrate import quad
from scipy.special import ndtr


def audit():
    # Natural~N(-1+a,1), AI~N(1+a,1), score=x, threshold=0.
    # Pair-score ranking depends only on their difference, hence is a-independent.
    result = {'definition': 'Natural N(-1+a,1), AI N(1+a,1), fixed score=x, threshold=0',
              'scope': 'Own analytic counterexample; no dataset inference or generalization claim', 'cases': {}}
    for shift in (0., 3.):
        auc = float(ndtr(sqrt(2)))
        natural = float(ndtr(1-shift)); ai = float(ndtr(1+shift))
        quadrature = []
        for tol in (1e-10, 1e-12):
            def density(x, mean):
                return exp(-.5*(x-mean)**2)/sqrt(2*pi)
            tn, e0 = quad(density, -float('inf'), 0., args=(-1+shift,), epsabs=tol, epsrel=tol)
            tp, e1 = quad(density, 0., float('inf'), args=(1+shift,), epsabs=tol, epsrel=tol)
            # Independent definition of the score difference, N(2,2).
            rank, ea = quad(lambda x: exp(-.25*(x-2)**2)/sqrt(4*pi), 0., float('inf'), epsabs=tol, epsrel=tol)
            error = max(abs(tn-natural), abs(tp-ai), abs(rank-auc))
            if error > 1e-11: raise ValueError('Gaussian special-function/quadrature disagreement')
            quadrature.append({'tolerance': tol, 'ba': (tn+tp)/2, 'auc': rank,
                               'max_absolute_disagreement': error, 'reported_quadrature_error': max(e0,e1,ea)})
        result['cases'][f'{shift:g}'] = {'auc': auc, 'balanced_accuracy': (natural+ai)/2,
                                       'natural_accuracy': natural, 'ai_accuracy': ai, 'quadrature': quadrature}
    a, b = result['cases']['0'], result['cases']['3']
    result['balanced_accuracy_drop'] = a['balanced_accuracy']-b['balanced_accuracy']
    result['auc_drop'] = a['auc']-b['auc']
    # Deliberately false inference: stable AUC implies BA change<=2pp.
    result['false_stability_claim_rejected'] = result['auc_drop'] == 0 and result['balanced_accuracy_drop'] > .02
    if not result['false_stability_claim_rejected']: raise ValueError('Negative inference control failed')
    result['precision_limit'] = 'Float64 at two quadrature tolerances, not arbitrary precision certification'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve counterexample receipt')
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'auc_drop': result['auc_drop'], 'ba_drop': result['balanced_accuracy_drop'],
                      'false_claim_rejected': result['false_stability_claim_rejected']}))


if __name__ == '__main__': main()
