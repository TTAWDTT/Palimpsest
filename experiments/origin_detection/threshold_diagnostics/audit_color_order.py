"""Own known-answer counterexamples to extending color-order invariance."""

import json

import numpy as np

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR
from pathlib import Path


def comparison(a, b):
    a, b = np.asarray(a), np.asarray(b)
    less = bool(np.all(a <= b) and np.any(a != b))
    greater = bool(np.all(b <= a) and np.any(a != b))
    return {'strictly_dominated': less, 'strictly_dominates': greater,
            'incomparable': not bool(np.all(a <= b) or np.all(b <= a))}


def main():
    path = WORK_DIR/'robust_statistics/color_dependency_review/order_counterexamples.json'
    if path.exists():
        raise FileExistsError('Preserve color-order audit')
    a, b = np.array([2., 0., 0.]), np.array([0., 2., 0.])
    matrix = np.array([[1., 1., .1], [1., 2., .2], [1., 3., .4]])
    matrix /= matrix.sum(axis=1, keepdims=True)
    before, after = comparison(a, b), comparison(matrix@a, matrix@b)
    if not before['incomparable'] or not after['strictly_dominated'] or abs(np.linalg.det(matrix)) <= 1e-6:
        raise ValueError('Color mixing counterexample failed')
    patch_a, patch_b = np.array([0., 4.]), np.array([2., 3.])
    resize_before = bool(patch_a.mean() < patch_b.mean())
    resize_after = bool((patch_a**2).mean() < (patch_b**2).mean())
    if not resize_before or resize_after:
        raise ValueError('Tone/averaging counterexample failed')
    c, d = np.array([.4, .6, .8]), np.array([.45, .65, .85])
    if not comparison(c, d)['strictly_dominated'] or comparison(np.floor(c), np.floor(d))['strictly_dominated']:
        raise ValueError('Quantization counterexample failed')
    monotone = comparison(np.array([1., 2., 3.])**[1., 2., 3.], np.array([2., 3., 4.])**[1., 2., 3.])
    if not monotone['strictly_dominated']:
        raise ValueError('Strict component-wise monotone control failed')
    result = {'script_sha256': file_sha256(Path(__file__)), 'passed': True,
              'mixing': {'matrix': matrix.tolist(), 'determinant': float(np.linalg.det(matrix)),
                         'before': before, 'after': after},
              'tone_and_averaging': {'before_mean': [float(patch_a.mean()), float(patch_b.mean())],
                                      'after_mean': [float((patch_a**2).mean()), float((patch_b**2).mean())]},
              'quantization_loses_strict_order': True, 'component_monotone_control': monotone,
              'scope': 'Own finite arithmetic controls; no image classification/physical experiment or universal proof'}
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
