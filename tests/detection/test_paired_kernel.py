"""Artificial controls precede real cache opening for local paired scores."""

from fractions import Fraction

import numpy as np
import pytest

from palimpsest.detection.algorithms.paired_kernel import (
    PairedKernelRule, gaussian_rows, solve_panel, source_centers,
)
from experiments.origin_detection.paired_kernel.run_iteration import wrong_sources
from experiments.origin_detection.source_view_risk.run_iteration import VARIANTS


def test_known_fraction_solution_and_variance_shrinkage():
    phi = np.array([[1., 0], [.5, .5], [0, 1], [.5, .5]])
    labels, sources = np.array([0, 0, 1, 1], np.uint8), np.array([0, 0, 1, 1])
    diagnostics = []
    for strength in (0, 1):
        alpha, bias, diagnostic = solve_panel(phi, labels, np.full(4, .25), sources, np.eye(2), strength=strength)
        expected = float(Fraction(1, 4)/(Fraction(1, 4)+strength*Fraction(1, 8)+Fraction(1, 100)+Fraction(1, 1000000)))
        np.testing.assert_allclose(alpha, [-expected, expected], atol=1e-12, rtol=0)
        assert abs(bias) < 1e-12 and diagnostic['relative_normal_residual'] < 1e-12
        diagnostics.append(diagnostic)
    assert diagnostics[1]['weighted_source_variance'] < diagnostics[0]['weighted_source_variance']
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(alpha, [-expected+.001, expected], atol=1e-12, rtol=0)


def test_known_kernel_and_exact_portable_single_batch(tmp_path):
    anchors = np.array([[0.], [2.]])
    x = np.array([[0.], [1.], [2.]])
    expected = np.array([[1., np.exp(-2)], [np.exp(-.5), np.exp(-.5)], [np.exp(-2), 1.]])
    np.testing.assert_array_equal(gaussian_rows(x, anchors, 1.), expected)
    rule = PairedKernelRule(('x',), ((0.,), (2.,)), 1., (-1., 1.), .25, .1, 'toy')
    path = tmp_path/'rule.json'
    rule.save(path)
    restored = PairedKernelRule.load(path)
    np.testing.assert_array_equal(restored.score(x), rule.score(x))
    np.testing.assert_array_equal(np.array([rule.score(row[None, :])[0] for row in x]), rule.score(x))


def test_source_center_scale_and_refusal():
    centers, width = source_centers([[0.], [2.], [4.], [6.]], [0, 0, 1, 1])
    np.testing.assert_array_equal(centers, [[1.], [5.]])
    assert width == 16
    with pytest.raises(ValueError):
        source_centers([[1.], [1.], [1.], [1.]], [0, 0, 1, 1])
    for width in (0, -1, np.nan):
        with pytest.raises(ValueError):
            gaussian_rows([[0.]], [[1.]], width)
    with pytest.raises(ValueError):
        solve_panel(np.eye(4), [0, 0, 1, 1], np.full(4, .25), [0, 1, 0, 1], np.eye(4), strength=1)


def test_wrong_source_control_preserves_class_scene_panels():
    rows = [{'domain': 'toy', 'scene': 'scene', 'src': str(g), 'role': 'fit',
             'condition': c, 'variant': v, 'label': 'REAL' if g < 6 else 'FAKE'}
            for g in range(12) for c in ('original', 'processed1', 'processed2') for v in VARIANTS[:2]]
    rows.sort(key=lambda r: tuple(r[k] for k in ('domain', 'src', 'condition', 'variant')))
    lookup = {s: i for i, s in enumerate(sorted({r['src'] for r in rows}))}
    sources = np.array([lookup[r['src']] for r in rows])
    wrong = wrong_sources(rows, sources)
    for group in np.unique(wrong):
        selected = [r for r, g in zip(rows, wrong) if g == group]
        assert len(selected) == 6 and len({r['label'] for r in selected}) == 1
        assert len({(r['condition'], r['variant']) for r in selected}) == 6
