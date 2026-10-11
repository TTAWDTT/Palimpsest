"""Pinned DEAR-r native-resolution research baseline, with external author code.

Weights are CC BY-NC 4.0 with declared OpenRAIL-M use restrictions. Author
code remains external; this adapter does not implement RAD or train a model.
"""

from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from PIL import Image

from palimpsest.contracts import Prediction, validate_rgb
from palimpsest.io.hashing import file_sha256
from .spatial_tiling import tiled_feature_mean

WEIGHT_SHA256 = '430fde11debe1850ab24af43945b4d68f2bb3ee52a837d84cf525eb603ccde97'


class DearRDetector:
    name = 'DEAR-r official native-resolution'

    def __init__(self, vendor_code: Path, checkpoint: Path, *, source_pins, device='cuda:0'):
        if file_sha256(checkpoint) != WEIGHT_SHA256:
            raise ValueError('DEAR-r official weight digest differs')
        for relative, sha in source_pins.items():
            if file_sha256(vendor_code / relative) != sha:
                raise ValueError('DEAR source snapshot differs')
        import torch
        from torchvision import transforms
        sys.path.insert(0, str(vendor_code.resolve()))
        from dear.detector.rajan_mask_gated_detector import RajanMaskGatedDetector
        self.torch = torch
        self.device = device
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        detector = RajanMaskGatedDetector(device=device, pretrained=False)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if set(state) != {'model'}:
            raise ValueError('Unexpected DEAR checkpoint schema')
        # Author load uses strict=False to accept unwrapped base weights too.
        # This adapter accepts only the complete published DEAR-r snapshot.
        detector.model.load_state_dict(state['model'], strict=True)
        detector.eval()
        for parameter in detector.model.parameters():
            parameter.requires_grad_(False)
        gate = detector.model.gate.gate.detach().cpu().numpy()
        if gate.shape != (2048,) or not np.isin(gate, (0, 1)).all() or np.count_nonzero(gate) != 820:
            raise ValueError('Unexpected DEAR-r bilateral gate')
        self.detector = detector
        self.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
        witness = np.full((96, 96, 3), 128, np.uint8)
        first = self.predict(witness).score
        second = self.predict(witness).score
        if first != second:
            raise ValueError('DEAR repeated witness changed')
        self.provenance = {'weights_sha256': WEIGHT_SHA256, 'source_pins': source_pins,
            'gate_retained': int(np.count_nonzero(gate)), 'gate_total': len(gate),
            'preprocessing': 'Official ToTensor/ImageNet normalization;no resize/crop;stored orientation',
            'precision': 'FP32', 'torch': torch.__version__, 'device': device,
            'parameters_trained': 0, 'all_parameters_frozen': True,
            'strict_loading': True, 'synthetic_repeat_exact': True,
            'license': 'CC BY-NC4.0 plus declared OpenRAIL-M use restrictions',
            'pretraining_overlap': 'unknown;not independent training provenance audit'}

    def predict(self, image):
        validate_rgb(image)
        start = perf_counter()
        tensor = self.transform(Image.fromarray(image)).unsqueeze(0)
        ready = perf_counter()
        with self.torch.inference_mode():
            score = float(self._forward(tensor.to(self.device)).item())
        if str(self.device).startswith('cuda'):
            self.torch.cuda.synchronize(self.device)
        end = perf_counter()
        return Prediction(self.name, score, 0, 'logit', timing_ms={
            'preprocess_ms': 1000 * (ready-start), 'forward_ms': 1000 * (end-ready),
            'predict_ms': 1000 * (end-start)}, metadata={
            'status': 'Official fixed baseline;no target-data calibration',
            'input': 'Native RGB;no resizing or cropping'})

    def _forward(self, tensor):
        return self.detector.predict(tensor)


class TiledDearRDetector(DearRDetector):
    """Explicit execution variant; no automatic resolution fallback."""

    name = 'DEAR-r aligned spatial execution'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provenance['execution'] = 'Stride8,64pixel halo,64x64output cells;FP64 sum/FP32 head'
        self.provenance['native_equivalence'] = 'Requires separately signed numerical parity gate'

    def _forward(self, tensor):
        model = self.detector.model
        mean = tiled_feature_mean(tensor, model.forward_features_no_gate)
        return model.backbone.forward_head(model.gate(mean))
