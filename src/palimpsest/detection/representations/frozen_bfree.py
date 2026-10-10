"""Released B-Free pre-head crop mean; author neural weights remain frozen."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from PIL import Image

from palimpsest.detection.baselines.bfree import BFreeDetector, infer_pil
from palimpsest.io.hashing import file_sha256

FEATURE_NAMES = tuple(f'bfree/adapted_crop_mean/{i}' for i in range(768))
WEIGHT_SHA = '5948ca78f4d94e820c250d24cdf155035b4a85960443800bfe6bb7f06bffe947'


def pack_mean(values):
    values = np.asarray(values, np.float64)
    if values.shape != (768,) or not np.isfinite(values).all():
        raise ValueError('Invalid B-Free pooled mean')
    return values.copy()


@dataclass(frozen=True)
class BFreeFeatures:
    values: np.ndarray
    official_logit: float
    linear_mean_error: float
    end_to_end_ms: float


class FrozenBFree:
    def __init__(self, vendor, weights, *, source_pins, device='cuda:0'):
        if not str(device).startswith('cuda'):
            raise ValueError('This B-Free representation protocol requires CUDA')
        for name, sha in source_pins.items():
            if file_sha256(vendor/name) != sha:
                raise ValueError('Pinned B-Free source differs')
        weight = weights/'BFREE_dino2reg4/model_epoch_best.pth'
        if file_sha256(weight) != WEIGHT_SHA:
            raise ValueError('Pinned B-Free weight differs')
        self.base = BFreeDetector(vendor, weights, device=device, tile_patches=64)
        self.base.torch.backends.cuda.matmul.allow_tf32 = False
        head = self.base.model.model.head
        if (not isinstance(head, self.base.torch.nn.Linear) or head.in_features != 768 or head.out_features != 1
                or head.weight.dtype != self.base.torch.float32):
            raise ValueError('Unexpected B-Free final head schema/dtype')
        self.captured = []
        self.handle = head.register_forward_pre_hook(self._capture)
        self.provenance = {'weight_sha256': WEIGHT_SHA, 'source_pins': source_pins,
            'weight_config_sha256': file_sha256(weight.parent/'config.yaml'),
            'features': 'Device FP32 mean of5x768 pre-linear-head pooled vectors,exact FP64 storage',
            'preprocess': 'Unchanged author504 patch-space five crops;audited tile64/crop-first>8MP',
            'neural_parameters_trained_here': 0, 'dtype': 'FP32,no AMP,no L2',
            'vendor_license': 'Informational/nonprofit author terms retained'}

    def _capture(self, module, inputs):
        del module
        if len(inputs) != 1 or inputs[0].shape != (5, 768) or inputs[0].dtype != self.base.torch.float32:
            raise ValueError('Unexpected B-Free five-crop head input')
        self.captured.append(inputs[0].detach().clone())

    def extract_file(self, path):
        start = perf_counter()
        self.captured.clear()
        with Image.open(path) as opened:
            result = infer_pil(opened, self.base.torch, self.base.model, self.base.transform, self.base.device)
        if len(self.captured) != 1:
            raise ValueError('B-Free head hook must fire once per query')
        values = self.captured[0].mean(dim=0)
        with self.base.torch.inference_mode():
            head = self.base.model.model.head
            reconstructed = float(self.base.torch.nn.functional.linear(
                values[None, :], head.weight, head.bias)[0, 0].cpu())
        # Use the verified Linear parameters directly for this algebra control;
        # calling the head module again would re-enter the five-crop observer.
        array = pack_mean(values.cpu().numpy())
        error = abs(reconstructed-result['score'])
        if not np.isfinite(result['score']) or error > 1e-5:
            raise ValueError('B-Free crop mean/head algebra differs')
        self.captured.clear()
        return BFreeFeatures(array, result['score'], error, (perf_counter()-start)*1000)

    def close(self):
        self.handle.remove()
        self.captured.clear()

    def direct_file_logit(self, path):
        """Original scorer with our observer disabled, for interface controls."""
        self.handle.remove()
        try:
            with Image.open(path) as opened:
                result = infer_pil(opened, self.base.torch, self.base.model, self.base.transform, self.base.device)
        finally:
            self.handle = self.base.model.model.head.register_forward_pre_hook(self._capture)
            self.captured.clear()
        return result['score']
