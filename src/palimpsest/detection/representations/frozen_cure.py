"""Frozen released CuRe representation before and after its response projection.

Raw vectors are stored without L2 normalization or bounding transforms. Only
the later conventional readout is fitted locally; the LoRA was trained by the
author and is frozen here. Input is a file to preserve the official PNG branch.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from palimpsest.detection.baselines.cure import CureFileDetector, official_margin

FULL_NAMES = tuple(f'cure/adapted_raw/{i}' for i in range(1024))
PROJECTED_NAMES = tuple(f'cure/response_raw/{i}' for i in range(128))
FEATURE_NAMES = FULL_NAMES + PROJECTED_NAMES


def pack_features(full, projected):
    full, projected = np.asarray(full, np.float64), np.asarray(projected, np.float64)
    if full.shape != (1024,) or projected.shape != (128,) or not all(np.isfinite(v).all() for v in (full, projected)):
        raise ValueError('Invalid raw CuRe feature vectors')
    return np.concatenate((full, projected))


@dataclass(frozen=True)
class CureFeatures:
    values: np.ndarray
    probability_fake: float
    decode_preprocess_ms: float
    encoder_ms: float


class FrozenCure:
    def __init__(self, vendor, adapter, base, *, source_pins, device='cuda:0'):
        if not str(device).startswith('cuda'):
            raise ValueError('This fixed CuRe feature protocol requires CUDA')
        self.base = CureFileDetector(vendor, adapter, base, source_pins=source_pins, device=device)
        self.provenance = {**self.base.provenance,
            'features': '1024 post-LN patch mean and128 author (features-mu)@W;unbounded raw',
            'backbone_parameters_trained_here': 0, 'representation_parameters_trained_here': 0,
            'storage': 'exact FP32/FP16 to FP64 cast;no L2 or nonlinear transform'}

    def extract_file(self, path):
        base = self.base
        start = perf_counter()
        tensor = base.preprocess(path).unsqueeze(0)
        if tensor.shape != (1, 3, 336, 336) or not base.torch.isfinite(tensor).all():
            raise ValueError('Invalid CuRe file tensor')
        ready = perf_counter()
        with base.torch.inference_mode(), base.torch.amp.autocast('cuda', enabled=True):
            full = base.model.backbone(tensor.to(base.device))
            projected = (full - base.model.mu) @ base.model.W
            probability = float(base.model.head(projected).softmax(dim=1)[0, 1].float().cpu())
        values = pack_features(full[0].float().cpu().numpy(), projected[0].float().cpu().numpy())
        base.torch.cuda.synchronize(base.device)
        end = perf_counter()
        official_margin(probability)
        return CureFeatures(values, probability, 1000 * (ready-start), 1000 * (end-ready))
