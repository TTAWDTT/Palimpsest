"""Frozen DINO patch standard deviations; fixed diagonal second moments only."""

from time import perf_counter

import numpy as np

from .frozen_dinov2_small import FEATURE_NAMES as BASE_NAMES
from .frozen_dinov2_small import FrozenDinoV2Small, SmallFeatures, prepare224, shifted_units

DISPERSION_NAMES = tuple(f'frozen_dinov2_small/patch_sd_unit/{i}' for i in range(384))
FEATURE_NAMES = BASE_NAMES + DISPERSION_NAMES


def patch_dispersion(patches):
    """Centered population SD of L2-normalized patches, then shifted unit SD.

    This is sqrt(diag(C)), not diag(sqrt(C)). Patches are image features,
    not assumed independent draws from a Gaussian distribution.
    """
    values = np.asarray(patches, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2 or values.shape[1] != 384 or not np.isfinite(values).all():
        raise ValueError('Invalid frozen dispersion patch matrix')
    norms = np.linalg.norm(values, axis=1)
    if np.min(norms) <= 1e-12:
        raise ValueError('Invalid frozen dispersion patch norm')
    units = values / norms[:, None]
    centered = units - units.mean(axis=0)
    sd = np.sqrt(np.mean(centered ** 2, axis=0))
    norm = np.linalg.norm(sd)
    normalized = np.zeros(384) if norm <= 1e-12 else sd / norm
    return (normalized + 1) / 2


class FrozenPatchDispersion:
    def __init__(self, checkpoint, source_receipt, *, device='cuda:0'):
        self.base = FrozenDinoV2Small(checkpoint, source_receipt, device=device)
        self.provenance = {**self.base.provenance,
            'descriptor': 'Original768 plus independently unit centered patch SD384;FP64 moment',
            'zero_spread': 'All384 stored channels=.5', 'dimension': 1152,
            'extra_encoder_for_dispersion': False, 'full_covariance_pooling': False}

    def extract(self, image):
        start = perf_counter()
        pixels = prepare224(image)
        ready = perf_counter()
        torch, model, device = self.base.torch, self.base.model, self.base.device
        with torch.inference_mode():
            result = model.forward_features(torch.from_numpy(pixels.copy()).unsqueeze(0).to(device))
            cls = result['x_norm_clstoken'][0].cpu().numpy()
            patches = result['x_norm_patchtokens'][0]
            mean = patches.mean(dim=0).cpu().numpy()
            patch_values = patches.cpu().numpy()
            torch.cuda.synchronize(device)
        values = np.concatenate((shifted_units(cls, mean), patch_dispersion(patch_values)))
        return SmallFeatures(values, 1000 * (ready - start), 1000 * (perf_counter() - ready))
