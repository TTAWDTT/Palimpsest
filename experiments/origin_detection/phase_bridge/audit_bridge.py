"""Surviving block-DCT signs do not force unchanged global Fourier phase."""

import cmath
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.fft import idctn

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR


def explicit_block(coefficients):
    alpha = [1 / math.sqrt(8)] + [math.sqrt(2 / 8)] * 7
    return np.array([[sum(alpha[u] * alpha[v] * coefficients[u, v]
        * math.cos(math.pi * (2 * y + 1) * u / 16)
        * math.cos(math.pi * (2 * x + 1) * v / 16)
        for u in range(8) for v in range(8)) for x in range(8)] for y in range(8)])


def explicit_frequency(block):
    return sum(float(block[y, x]) * cmath.exp(-2j * math.pi * x / 8)
               for y in range(8) for x in range(8))


def main():
    output = WORK_DIR / 'robust_statistics/broad_methods_review/phase_bridge.json'
    if output.exists():
        raise FileExistsError('Preserve phase-bridge numerical witness')
    coefficients = np.zeros((8, 8))
    coefficients[0, 0], coefficients[0, 1], coefficients[0, 2] = 1024, 6.1, 9.9
    table = np.ones((8, 8))
    table[0, 1], table[0, 2] = 11, 10
    dequantized = np.rint(coefficients / table) * table
    active = coefficients != 0
    if not np.array_equal(np.sign(coefficients[active]), np.sign(dequantized[active])):
        raise ValueError('Witness did not preserve every active DCT sign')
    originals = [idctn(c, norm='ortho') for c in (coefficients, dequantized)]
    explicit = [explicit_block(c) for c in (coefficients, dequantized)]
    block_error = max(float(np.abs(a - b).max()) for a, b in zip(originals, explicit))
    frequencies = [np.fft.fft2(b)[0, 1] for b in originals]
    alternate = [explicit_frequency(b) for b in explicit]
    frequency_error = max(abs(a - b) for a, b in zip(frequencies, alternate))
    if block_error > 1e-12 or frequency_error > 1e-10:
        raise ValueError('Explicit DCT/DFT cross-check failed')
    delta = float(np.angle(frequencies[1] * np.conj(frequencies[0])))
    if abs(delta) < .1 or min(abs(v) for v in frequencies) < 1:
        raise ValueError('Finite nonzero Fourier phase witness failed')
    value = {'dct_nonzero_signs_preserved': True, 'before_active': coefficients[active].tolist(),
        'after_active': dequantized[active].tolist(), 'active_quantization': table[active].tolist(),
        'frequency_index': [0, 1], 'before_phase': float(np.angle(frequencies[0])),
        'after_phase': float(np.angle(frequencies[1])), 'wrapped_phase_change_rad': delta,
        'wrapped_phase_change_degrees': math.degrees(delta),
        'explicit_block_max_error': block_error, 'explicit_frequency_max_error': frequency_error,
        'pixel_bounds': [[float(b.min()), float(b.max())] for b in originals],
        'planted_zero_phase_change_rejected': abs(delta) > 1e-10,
        'numpy': np.__version__, 'scipy': scipy.__version__,
        'script_sha256': file_sha256(Path(__file__)),
        'scope': 'Two numeric paths sharing witness;floating block operator,not JPEGpixel/device validation'}
    output.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
