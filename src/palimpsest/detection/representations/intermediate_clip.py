"""One fixed midpoint LN2 token alongside the pinned final CLIP embedding.

Intermediate forensic features have direct prior work (RINE). This extractor
does not implement RINE's trainable projections, importance estimator or head.
"""

from time import perf_counter

import numpy as np

from .frozen_clip import FrozenClip, ClipFeatures, FEATURE_NAMES as GLOBAL_NAMES
from .frozen_clip import prepare224, shifted_unit_feature

MID_NAMES = tuple(f'frozen_clip/midpoint12_ln2_unit_shifted/{i}' for i in range(1024))
FEATURE_NAMES = GLOBAL_NAMES + MID_NAMES


def shifted_midpoint(values):
    values = np.asarray(values, np.float64)
    if values.shape != (1024,) or not np.isfinite(values).all() or np.linalg.norm(values) <= 1e-12:
        raise ValueError('Invalid midpoint token')
    return (values / np.linalg.norm(values) + 1) / 2


class IntermediateFrozenClip(FrozenClip):
    def __init__(self, checkpoint, *, device='cuda:0'):
        super().__init__(checkpoint, device=device)
        self.blocks = list(self.model.transformer.resblocks.children())
        if len(self.blocks) != 24:
            raise ValueError('Expected 24 pinned CLIP blocks')
        self.provenance.update({'intermediate': 'Block12 LN2 CLS,after attention residual,beforeMLP',
            'descriptor_dimension': 1792, 'storage': 'Independent L2 unit shift for final768 andmidpoint1024',
            'implementation': 'Original traced submodules,replayed visual flow;requires separate final parity gate',
            'prior_work': 'RINE/ECCV2024;own source-risk conventional readout adaptation'})

    def forward_tokens(self, tensor):
        model, torch = self.model, self.torch
        x = model.conv1(tensor).reshape(1, 1024, -1).permute(0, 2, 1)
        cls = model.class_embedding.to(dtype=x.dtype) + torch.zeros((1, 1, 1024), dtype=x.dtype, device=x.device)
        x = torch.cat((cls, x), dim=1)
        x = model.ln_pre(x + model.positional_embedding.to(dtype=x.dtype)).permute(1, 0, 2)
        midpoint = None
        for index, block in enumerate(self.blocks, 1):
            if index == 12:
                x = x + block.attn(block.ln_1(x))
                normalized = block.ln_2(x)
                midpoint = normalized[0, 0].clone()
                x = x + block.mlp(normalized)
            else:
                x = block(x)
        final = model.ln_post(x[0]) @ model.proj
        return final, midpoint

    def extract(self, image):
        start = perf_counter()
        pixels = prepare224(image)
        ready = perf_counter()
        torch = self.torch
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            tensor = torch.from_numpy(pixels.copy()).unsqueeze(0).to(self.device).half()
            final, midpoint = self.forward_tokens(tensor)
            final = final[0].float().cpu().numpy()
            midpoint = midpoint.float().cpu().numpy()
            torch.cuda.synchronize(self.device)
        values = np.concatenate((shifted_unit_feature(final), shifted_midpoint(midpoint)))
        return ClipFeatures(values, 1000*(ready-start), 1000*(perf_counter()-ready))
