"""Independent exact ordering and derivative checks for the constant-loss toy."""

from fractions import Fraction

import pytest

from experiments.origin_detection.frozen_loss_audit.audit_loss import fixed_loss, verify_constant_offsets


def test_exact_minimizer_and_all_pair_ordering():
    points = tuple(Fraction(n, 3) for n in range(-6, 7))
    offset = verify_constant_offsets(fixed_loss, lambda t: fixed_loss(t) + Fraction(1, 2), points)
    assert offset == Fraction(1, 2)
    # Expanded quadratic3*t^2+2*t+3;derivative6*t+2.
    minimum = Fraction(-1, 3)
    assert 6 * minimum + 2 == 0
    assert fixed_loss(minimum) == Fraction(8, 3)
    for a in points:
        assert fixed_loss(a) - fixed_loss(minimum) == 3 * (a - minimum) ** 2
        for b in points:
            assert (fixed_loss(a) + offset) - (fixed_loss(b) + offset) == 3 * (a*a - b*b) + 2 * (a - b)


def test_trainable_feature_offset_refused():
    points = tuple(Fraction(n, 2) for n in range(-2, 3))
    with pytest.raises(ValueError, match='not constant'):
        verify_constant_offsets(fixed_loss, lambda t: fixed_loss(t) + t*t/2, points)
