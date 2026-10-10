"""Exact original final output gate for one fixed midpoint execution trace."""

import json
from pathlib import Path

from palimpsest.detection.representations.intermediate_clip import IntermediateFrozenClip
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import REPO_ROOT, WORK_DIR, MODELS_ROOT
from experiments.origin_detection.phase_statistics.run_iteration import write_json

OUTPUT = WORK_DIR / 'robust_statistics/intermediate_encoder'


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / 'runtime_controls.json'
    if destination.exists():
        raise FileExistsError('Preserve midpoint runtime gate')
    encoder = IntermediateFrozenClip(MODELS_ROOT / 'd3/ViT-L-14.pt')
    torch = encoder.torch
    generator = torch.Generator(device=encoder.device).manual_seed(20261006)
    witnesses = [torch.zeros((1,3,224,224), device=encoder.device).half(),
                 torch.linspace(-1,1,3*224*224,device=encoder.device).reshape(1,3,224,224).half(),
                 torch.randn((1,3,224,224),generator=generator,device=encoder.device).half()]
    measurements = []
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        for witness in witnesses:
            expected = encoder.model(witness)
            final, middle = encoder.forward_tokens(witness)
            repeated, middle_repeat = encoder.forward_tokens(witness)
            passed = (torch.equal(expected,final) and torch.equal(final,repeated)
                      and torch.equal(middle,middle_repeat) and bool(torch.isfinite(middle).all()))
            measurements.append({'final_exact': torch.equal(expected,final), 'repeat_exact': torch.equal(middle,middle_repeat),
                                 'maximum_final_difference': float((expected-final).abs().max().item())})
            if not passed:
                write_json(destination, {'passed': False, 'measurements': measurements,
                           'scope': 'Refused exact final parity;no tolerance widening'})
                raise ValueError('Midpoint execution altered final output')
    paths = [Path(__file__), REPO_ROOT / 'src/palimpsest/detection/representations/intermediate_clip.py']
    write_json(destination, {'passed': True, 'measurements': measurements, 'encoder': encoder.provenance,
        'code_pins': {str(p.relative_to(REPO_ROOT)):file_sha256(p) for p in paths},
        'scope': 'Synthetic execution parity only;no image accuracy or independence evidence'})
    print(json.dumps({'passed': True, 'witnesses': 3, 'maximum_final_difference': 0}), flush=True)


if __name__ == '__main__':
    main()
