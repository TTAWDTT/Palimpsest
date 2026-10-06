"""A fixed canonical Gaussian probe, frozen prefix, and response descriptor.

The probe measures local representation sensitivity, not propagation invariance.
RINE/RIGID/MINDER and intermediate-response detection are direct prior work.
"""

from time import perf_counter

import cv2
import numpy as np

from .frozen_clip import ClipFeatures, FEATURE_NAMES as GLOBAL_NAMES, prepare224, shifted_unit_feature
from .intermediate_clip import IntermediateFrozenClip, shifted_midpoint

PROBE_NAMES = tuple(f'frozen_clip/probe_sigma1_mid12/{i}' for i in range(1024))
DELTA_NAMES = tuple(f'frozen_clip/response_sigma1_mid12/{i}' for i in range(1024))
SENSITIVITY_NAMES = ('frozen_clip/response_sigma1_cosine',)
RESPONSE_NAMES = DELTA_NAMES + SENSITIVITY_NAMES
FEATURE_NAMES = GLOBAL_NAMES + RESPONSE_NAMES


def gaussian_probe(pixels):
    """CHW FP32 normalized crop; fixed7x7 sigma1, independent channels, reflect101."""
    pixels = np.asarray(pixels)
    if pixels.shape != (3, 224, 224) or pixels.dtype != np.float32 or not np.isfinite(pixels).all():
        raise ValueError('Probe requires finite normalized CHW FP32 crop')
    result = cv2.GaussianBlur(pixels.transpose(1, 2, 0), (7, 7), 1,
                              sigmaY=1, borderType=cv2.BORDER_REFLECT_101)
    return np.ascontiguousarray(result.transpose(2, 0, 1))


def response_feature(original, probed):
    """Unit vector response; .5+.25*(u-v), then cosine disagreement in[0,1]."""
    units = []
    for value in (original, probed):
        value = np.asarray(value, np.float64)
        unit = 2*value - 1
        if (value.shape != (1024,) or not np.isfinite(value).all()
                or np.any(value < 0) or np.any(value > 1)
                or abs(np.linalg.norm(unit)-1) > 1e-6):
            raise ValueError('Response requires shifted unit midpoint vectors')
        units.append(unit)
    u, v = units
    cosine = np.clip((1-float(u@v))/2, 0, 1)
    return np.concatenate((.5+.25*(u-v), [cosine]))


class ResponseFrozenClip(IntermediateFrozenClip):
    def __init__(self, checkpoint, *, device='cuda:0'):
        super().__init__(checkpoint, device=device)
        self.provenance.update({'probe': 'prepare224 then FP32 Gaussian7x7 sigma1 reflect101, then fp16',
            'representation': 'Block12 LN2 CLS response plus final768; no learned probe or encoder',
            'execution': 'Cache extraction:probe12 blocks only;deployment:raw24+probe12 blocks',
            'descriptor_dimension': 1793, 'propagation_invariance_certified': False})

    def prefix_token(self, tensor, *, block_number=12):
        """Original traced prefix, stopping before the fixed block's MLP.

        Stem replay matches the signed intermediate extractor. Keep its published
        source immutable; the runtime gate compares this prefix bit for bit.
        Nondefault block_number is for the planted runtime wrong-block control.
        """
        if block_number not in (11, 12):
            raise ValueError('Only fixed12 and runtime wrong-block11 are supported')
        model, torch = self.model, self.torch
        x = model.conv1(tensor).reshape(1, 1024, -1).permute(0, 2, 1)
        cls = model.class_embedding.to(dtype=x.dtype) + torch.zeros((1, 1, 1024), dtype=x.dtype, device=x.device)
        x = torch.cat((cls, x), dim=1)
        x = model.ln_pre(x + model.positional_embedding.to(dtype=x.dtype)).permute(1, 0, 2)
        for index, block in enumerate(self.blocks, 1):
            if index == block_number:
                x = x + block.attn(block.ln_1(x))
                return block.ln_2(x)[0, 0].clone()
            x = block(x)
        raise RuntimeError('Frozen prefix missing')

    def extract(self, image):
        """New probe-only cache; original features are joined from signed stage24."""
        start = perf_counter()
        pixels = gaussian_probe(prepare224(image))
        ready = perf_counter()
        torch = self.torch
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            tensor = torch.from_numpy(pixels).unsqueeze(0).to(self.device).half()
            token = self.prefix_token(tensor).float().cpu().numpy()
            torch.cuda.synchronize(self.device)
        return ClipFeatures(shifted_midpoint(token), 1000*(ready-start), 1000*(perf_counter()-ready))

    def extract_response(self, image):
        """Single-image deployment features, using the same fixed cache operations."""
        start = perf_counter()
        pixels = prepare224(image)
        probed = gaussian_probe(pixels)
        ready = perf_counter()
        torch = self.torch
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            tensor = torch.from_numpy(pixels.copy()).unsqueeze(0).to(self.device).half()
            final, middle = self.forward_tokens(tensor)
            probe = self.prefix_token(torch.from_numpy(probed).unsqueeze(0).to(self.device).half())
            values = np.concatenate((shifted_unit_feature(final[0].float().cpu().numpy()),
                response_feature(shifted_midpoint(middle.float().cpu().numpy()),
                                 shifted_midpoint(probe.float().cpu().numpy()))))
            torch.cuda.synchronize(self.device)
        return ClipFeatures(values, 1000*(ready-start), 1000*(perf_counter()-ready))
