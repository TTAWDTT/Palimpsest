"""Known-input prefix parity gate, before any new target-image probe extraction."""

import json
from pathlib import Path

import numpy as np

from palimpsest.detection.representations.response_clip import ResponseFrozenClip, gaussian_probe
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR, MODELS_ROOT
from experiments.origin_detection.phase_statistics.run_iteration import write_json

OUTPUT = WORK_DIR / 'robust_statistics/response_probe'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / 'runtime_controls.json'
    if destination.exists():
        raise FileExistsError('Preserve response runtime gate')
    encoder = ResponseFrozenClip(MODELS_ROOT / 'd3/ViT-L-14.pt')
    torch = encoder.torch
    rng = np.random.default_rng(20261006)
    witnesses = [np.zeros((3,224,224), np.float32),
        np.linspace(-1,1,3*224*224,dtype=np.float32).reshape(3,224,224),
        rng.normal(size=(3,224,224)).astype(np.float32)]
    measurements = []
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        for witness in witnesses:
            tensor = torch.from_numpy(witness).unsqueeze(0).to(encoder.device).half()
            _, expected = encoder.forward_tokens(tensor)
            actual = encoder.prefix_token(tensor)
            wrong = encoder.prefix_token(tensor, block_number=11)
            probed = torch.from_numpy(gaussian_probe(witness)).unsqueeze(0).to(encoder.device).half()
            p, repeat = encoder.prefix_token(probed), encoder.prefix_token(probed)
            row = {'prefix_exact': torch.equal(expected, actual),
                'maximum_difference': float((expected-actual).abs().max().item()),
                'wrong_block_detected': not torch.equal(expected, wrong),
                'probe_repeat_exact': torch.equal(p, repeat), 'finite': bool(torch.isfinite(p).all())}
            measurements.append(row)
            if not all(row[k] for k in ('prefix_exact','wrong_block_detected','probe_repeat_exact','finite')):
                write_json(destination, {'passed':False, 'measurements':measurements})
                raise ValueError('Fixed prefix execution gate failed')
    paths = [Path(__file__), REPO_ROOT/'src/palimpsest/detection/representations/response_clip.py',
        REPO_ROOT/'src/palimpsest/detection/representations/intermediate_clip.py']
    write_json(destination, {'passed':True, 'measurements':measurements, 'encoder':encoder.provenance,
        'code_pins':{str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in paths},
        'scope':'Known synthetic execution checks,not detection or propagation evidence'})
    print(json.dumps({'passed':True,'witnesses':3,'wrong_block_detected':True}), flush=True)


if __name__ == '__main__':
    main()
