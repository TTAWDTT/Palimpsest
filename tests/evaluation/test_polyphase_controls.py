"""Finite mechanism controls must reject overbroad attribution and invariance."""

from fractions import Fraction

from experiments.origin_detection.non_neural.generator_mechanisms.audit_polyphase import controls


def test_exact_phase_controls_and_specificity_counterexample():
    result = controls()
    assert result['positive_fixtures'] == 5
    assert result['negative_controls_caught'] == 5
    assert result['ordinary_resize_matches_generator_proxy_exactly']
    assert result['fixtures']['nearest_repeat']['contrast'] == '1'
    assert result['fixtures']['two_tap_filter']['contrast'] == '0'
    assert Fraction(result['filtered_signal_mean_square']) > 0
    assert result['fixtures']['constant']['contrast'] is None
