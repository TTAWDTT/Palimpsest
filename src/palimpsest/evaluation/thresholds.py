"""Exact finite-sample threshold intervals for two class-accuracy requirements."""

import math

import numpy as np


def threshold_interval(scores, labels, minimum=.55):
    """For AI iff score>t, return the feasible half-open interval [lower,upper).

    This only describes the supplied finite observations; it is not a confidence
    interval or a population-level impossibility result.
    """
    s, y = np.asarray(scores, float), np.asarray(labels)
    if (s.ndim != 1 or y.shape != s.shape or set(y.tolist()) != {0, 1}
            or not np.isfinite(s).all() or not np.isfinite(minimum) or not 0 < minimum <= 1):
        raise ValueError('Invalid finite threshold interval inputs')
    natural, ai = np.sort(s[y == 0]), np.sort(s[y == 1])
    n = math.ceil(minimum*len(natural)); a = math.ceil(minimum*len(ai))
    lower, upper = float(natural[n-1]), float(ai[len(ai)-a])
    return {'lower_inclusive': lower, 'upper_exclusive': upper, 'feasible': lower < upper,
            'natural_records': len(natural), 'ai_records': len(ai),
            'required_natural_correct': n, 'required_ai_correct': a}
