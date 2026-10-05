"""Official D3 forward under our explicitly documented batch-one protocol.

RNG state is part of the predictor: unrelated torch callers must not alter its
shuffle sequence. Model construction is explicit and never downloads weights.
The official batch128 validation protocol is different from this API.
"""

from pathlib import Path
import sys
from threading import Lock
from time import perf_counter
from PIL import Image
from palimpsest.contracts import RGBImage, Prediction, validate_rgb
from palimpsest.io.hashing import file_sha256

CLIP_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
_RNG_LOCK = Lock()


def load_vendor(vendor: Path):
    import timm.layers.helpers as timm_helpers

    sys.path.insert(0, str(vendor.resolve()))
    # The original release imports unused MOCO with an old timm module name.
    sys.modules["timm.models.layers.helpers"] = timm_helpers
    from models.clip import clip
    from models.clip_models import CLIPModelShuffleAttentionPenultimateLayer

    return clip, CLIPModelShuffleAttentionPenultimateLayer


class D3Detector:
    name = "D3 batch1"

    def __init__(
        self,
        vendor: Path,
        clip_checkpoint: Path,
        head_checkpoint: Path,
        *,
        device="cuda:0",
        seed=418,
    ):
        import torch
        from torchvision import transforms

        if file_sha256(clip_checkpoint) != CLIP_SHA256:
            raise ValueError("CLIP checkpoint SHA-256 mismatch")
        self.torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(
                "This adapter currently supports CUDA only (our historical batch-one protocol)"
            )
        self.cuda_index = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        self.seed = seed
        clip, model_class = load_vendor(vendor)
        # Restore vendor download hook after construction; no home cache writes.
        download = clip._download
        try:
            clip._download = lambda _url, _root: str(clip_checkpoint.resolve())
            self.model = model_class(
                "ViT-L/14", shuffle_times=1, original_times=1, patch_size=[14]
            )
        finally:
            clip._download = download
        self.model.attention_head.load_state_dict(
            torch.load(head_checkpoint, map_location="cpu", weights_only=True),
            strict=True,
        )
        self.model.eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with (
            _RNG_LOCK,
            torch.random.fork_rng(devices=[self.cuda_index]),
            torch.cuda.device(self.device),
            torch.inference_mode(),
        ):
            for _ in range(3):
                self.model(torch.zeros(1, 3, 224, 224, device=self.device))
            torch.cuda.synchronize(self.device)
            torch.random.default_generator.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            self.cpu_rng = torch.get_rng_state()
            self.cuda_rng = torch.cuda.get_rng_state(self.device)

    def predict(self, image: RGBImage) -> Prediction:
        validate_rgb(image)
        torch = self.torch
        start = perf_counter()
        tensor = self.transform(Image.fromarray(image)).unsqueeze(0)
        prepared = perf_counter()
        with (
            _RNG_LOCK,
            torch.random.fork_rng(devices=[self.cuda_index]),
            torch.cuda.device(self.device),
            torch.inference_mode(),
        ):
            torch.set_rng_state(self.cpu_rng)
            torch.cuda.set_rng_state(self.cuda_rng, self.device)
            score = float(self.model(tensor.to(self.device)).flatten()[0].item())
            torch.cuda.synchronize(self.device)
            self.cpu_rng = torch.get_rng_state()
            self.cuda_rng = torch.cuda.get_rng_state(self.device)
        return Prediction(
            self.name,
            score,
            score_kind="logit",
            timing_ms={
                "preprocess": (prepared - start) * 1000,
                "forward": (perf_counter() - prepared) * 1000,
            },
            metadata={
                "protocol": f"224x224 CLIP; batch1; sequential RNG shuffle; seed={self.seed}"
            },
        )
