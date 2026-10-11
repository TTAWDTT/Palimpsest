"""Synthetic original-prefix parity and dispersion repeat; no real images."""

import json
from pathlib import Path

import numpy as np

from palimpsest.detection.representations.frozen_patch_dispersion import FrozenPatchDispersion
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import WORK_DIR, MODELS_ROOT
from experiments.origin_detection.phase_statistics.run_iteration import write_json

OUTPUT = WORK_DIR / 'robust_statistics/patch_dispersion'
SOURCE = WORK_DIR / 'robust_statistics/compact_encoder_review/runtime_receipt.json'


def main():
    destination = OUTPUT / 'runtime_controls.json'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError('Preserve patch dispersion runtime controls')
    encoder = FrozenPatchDispersion(MODELS_ROOT / 'dinov2_small/dinov2_vits14_pretrain.pth', SOURCE)
    cases = []
    for h, w in ((256, 256), (301, 417), (417, 301)):
        y, x = np.indices((h, w))
        image = np.stack([(x + y) % 256, (3 * x + y) % 256, (x + 7 * y) % 256], axis=2).astype(np.uint8)
        first = encoder.extract(image).values
        repeat = encoder.extract(image).values
        old = encoder.base.extract(image).values
        if not np.array_equal(first, repeat) or not np.array_equal(first[:768], old):
            raise ValueError('Frozen dispersion repeat/original prefix changed')
        if not np.isfinite(first).all() or first.shape != (1152,):
            raise ValueError('Frozen dispersion descriptor invalid')
        cases.append({'height': h, 'width': w, 'repeat_exact': True, 'old_prefix_exact': True})
    write_json(destination, {'passed': True, 'cases': cases, 'native_images_used': 0,
        'encoder': encoder.provenance, 'script_sha256': file_sha256(Path(__file__)),
        'source_receipt_sha256': file_sha256(SOURCE),
        'scope': 'Synthetic runtime parity,not a propagation robustness result'})
    print(json.dumps({'passed': True, 'synthetic_cases': 3}), flush=True)


if __name__ == '__main__':
    main()
