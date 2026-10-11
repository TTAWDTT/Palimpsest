"""File-based official CuRe inference, including its PNG-specific preprocessing.

This is deliberately a file adapter: passing only RGB would erase the author's
PNG branch. The author source and weights remain outside the repository.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from time import perf_counter

from palimpsest.io.hashing import file_sha256

ADAPTER_SHA = '17cabb334563c15abc8827db87dc272a4aef10f4279025faee1ef8d32c444ae1'
BASE_SHA = '0cdab5b338cbaa1e7a5dcd1b2fb4c9f4d5df1abd289564658edbab64a650e7e8'


def official_margin(probability):
    """Preserve author p>=.5 ties while the shared evaluator uses margin>0."""
    value = float(probability)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Invalid CuRe probability')
    return math.nextafter(0.0, 1.0) if value == .5 else value - .5


@dataclass(frozen=True)
class CureFilePrediction:
    probability_fake: float
    decode_preprocess_ms: float
    forward_ms: float
    end_to_end_ms: float

    @property
    def margin(self):
        return official_margin(self.probability_fake)


class CureFileDetector:
    """Fixed author model and batch1 fp16 protocol; no calibration or fitting."""

    def __init__(self, vendor_code: Path, adapter: Path, base: Path, *, source_pins, device='cuda:0'):
        if file_sha256(adapter) != ADAPTER_SHA or file_sha256(base) != BASE_SHA:
            raise ValueError('CuRe checkpoint digest differs')
        for relative, sha in source_pins.items():
            if file_sha256(vendor_code / relative) != sha:
                raise ValueError('CuRe source snapshot differs')
        import torch
        sys.path.insert(0, str(vendor_code.resolve()))
        import cure
        if Path(cure.__file__).resolve() != (vendor_code / 'cure/__init__.py').resolve():
            raise ValueError('A different cure package was already imported')
        torch.set_num_threads(4)
        self.torch, self.device = torch, device
        self.model = cure.load_model(adapter, device, base)
        if any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('CuRe parameters are not frozen')
        self.preprocess = cure.preprocess_image
        self.provenance = {'adapter_sha256': ADAPTER_SHA, 'base_sha256': BASE_SHA,
            'source_pins': source_pins, 'device': device, 'torch': torch.__version__,
            'precision': 'author CUDA fp16 outer autocast; model FP32 storage',
            'cpu_threads': 4, 'batch_size': 1, 'strict_loading': True,
            'parameters_fitted_here': 0, 'all_parameters_frozen': True,
            'preprocessing': 'author file decode;PNG JPEG96;short390 center390 resize336 JPEG70 normalize.5',
            'tie': 'p>=.5 FAKE;strict-margin evaluator receives smallest positive float at tie',
            'license': 'Apache-2.0', 'pretraining_overlap': 'unknown'}

    def predict_file(self, path: Path):
        start = perf_counter()
        tensor = self.preprocess(path).unsqueeze(0)
        if tensor.shape != (1, 3, 336, 336) or not self.torch.isfinite(tensor).all():
            raise ValueError('CuRe preprocessor returned invalid tensor')
        ready = perf_counter()
        with self.torch.inference_mode(), self.torch.amp.autocast('cuda', enabled=str(self.device).startswith('cuda')):
            probability = float(self.model(tensor.to(self.device)).softmax(dim=1)[0, 1].float().cpu())
        if str(self.device).startswith('cuda'):
            self.torch.cuda.synchronize(self.device)
        end = perf_counter()
        official_margin(probability)
        return CureFilePrediction(probability, 1000 * (ready-start), 1000 * (end-ready), 1000 * (end-start))
