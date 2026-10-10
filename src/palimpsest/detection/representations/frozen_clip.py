"""Local CLIP vision representation with explicit pins and no downloads.

This is a pretrained neural encoder. It must not be called a non-neural
statistical detector or a newly trained origin model.
"""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, __version__ as pillow_version

from palimpsest.contracts import validate_rgb
from palimpsest.detection.baselines.d3 import CLIP_SHA256
from palimpsest.io.hashing import file_sha256

FEATURE_NAMES = tuple(f'frozen_clip/unit_shifted/{i}' for i in range(768))
MEAN = np.array([.48145466, .4578275, .40821073], np.float32)[:, None, None]
STD = np.array([.26862954, .26130258, .27577711], np.float32)[:, None, None]


def prepare224(image):
    """PIL bicubic short-side resize and rounded center crop; stored orientation."""
    validate_rgb(image)
    h, w = image.shape[:2]
    size = (224, int(224*h/w)) if w <= h else (int(224*w/h), 224)
    im = Image.fromarray(image).resize(size, Image.Resampling.BICUBIC)
    left, top = round((size[0]-224)/2), round((size[1]-224)/2)
    rgb = np.asarray(im.crop((left, top, left+224, top+224)))
    return (rgb.transpose(2, 0, 1).astype(np.float32)/255-MEAN)/STD


def shifted_unit_feature(values):
    """Affine storage in [0,1]; (2*stored-1) is the L2 unit embedding."""
    return (unit_feature(values)+1)/2


def unit_feature(values):
    values = np.asarray(values, np.float64)
    if values.shape != (768,) or not np.isfinite(values).all():
        raise ValueError('Invalid CLIP feature')
    norm = np.linalg.norm(values)
    if norm <= 1e-12: raise ValueError('Degenerate CLIP feature')
    return values/norm


@dataclass(frozen=True)
class ClipFeatures:
    values: np.ndarray
    preprocess_ms: float
    statistics_ms: float


class FrozenClip:
    def __init__(self, checkpoint: Path, *, device='cuda:0'):
        if file_sha256(checkpoint) != CLIP_SHA256:
            raise ValueError('Frozen CLIP weight SHA differs')
        import torch
        if not str(device).startswith('cuda'): raise ValueError('This fixed diagnostic uses CUDA')
        self.torch = torch; self.device = torch.device(device)
        # Execute the published TorchScript vision method directly. The earlier
        # eager reconstruction failed its declared parity gate and was rejected.
        archive = torch.jit.load(str(checkpoint), map_location='cpu').eval()
        state = archive.visual.state_dict()
        if (state['conv1.weight'].shape != (1024, 3, 14, 14) or state['positional_embedding'].shape != (257, 1024)
                or state['proj'].shape != (1024, 768)):
            raise ValueError('Expected CLIP ViT-L/14 vision dimensions')
        self.model = archive.visual.eval().to(self.device)
        for parameter in self.model.parameters(): parameter.requires_grad_(False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            witness = torch.linspace(-1, 1, 3*224*224, device=self.device).reshape(1, 3, 224, 224).half()
            actual = self.model(witness)
            repeat = self.model(witness)
            if actual.shape != (1, 768) or not torch.equal(actual, repeat) or not torch.isfinite(actual).all():
                raise ValueError('Frozen archive vision witness failed')
        frozen = all(not value.requires_grad for value in self.model.parameters())
        if not frozen: raise ValueError('Encoder is not frozen')
        del archive, state
        self.provenance = {'weights_sha256': CLIP_SHA256, 'device': str(self.device), 'torch': torch.__version__,
                           'gpu': torch.cuda.get_device_name(self.device), 'precision': 'official JIT vision fp16',
                           'backbone_parameters_trained': 0, 'all_parameters_frozen': frozen,
                           'synthetic_repeat_exact': True, 'implementation': 'original archive.visual without reconstruction', 'jit_optimized_execution': False,
                           'storage': '(L2_unit+1)/2', 'numpy': np.__version__, 'pillow': pillow_version,
                           'torch_module': str(Path(torch.__file__).resolve()), 'numpy_module': str(Path(np.__file__).resolve())}

    def extract(self, image):
        torch = self.torch; start = perf_counter()
        pixels = prepare224(image)
        ready = perf_counter()
        with torch.inference_mode(), torch.jit.optimized_execution(False):
            tensor = torch.from_numpy(pixels.copy()).unsqueeze(0).to(self.device).half()
            values = self.model(tensor)[0].float().cpu().numpy()
            torch.cuda.synchronize(self.device)
        return ClipFeatures(shifted_unit_feature(values), 1000*(ready-start), 1000*(perf_counter()-ready))
