"""B-Free official preprocessing/forward, including the audited large-image path.

The vendor code and weights remain external; its nonprofit license still
applies. No torch import, download or model loading occurs on module import.
"""

from pathlib import Path
from contextlib import redirect_stdout
import sys
import time
from PIL import Image
from palimpsest.contracts import RGBImage, Prediction, validate_rgb


def load_official_model(
    vendor_code: Path, weights_root: Path, device: str, tile_patches: int
):
    sys.path.insert(0, str(vendor_code.resolve()))
    import torch
    import yaml
    from torchvision.transforms import Compose

    from networks import get_network, load_weights
    from utils.normalization import get_list_norm

    torch.backends.cudnn.allow_tf32 = False

    config_path = weights_root / "BFREE_dino2reg4" / "config.yaml"
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model_path = config_path.parent / config["weights_file"]
    model = load_weights(get_network(config["arch"]), str(model_path))
    model = model.to(device).eval()
    if tile_patches:
        convolution = model.patch_embed.proj
        if not isinstance(convolution, torch.nn.Conv2d):
            raise ValueError("Unexpected B-Free patch projection type")
        if (
            convolution.kernel_size != convolution.stride
            or convolution.padding != (0, 0)
            or convolution.dilation != (1, 1)
        ):
            raise ValueError("Tiling requires independent nonoverlapping patches")

        class TiledPatchProjection(torch.nn.Module):
            def __init__(self, projection, patches_per_tile):
                super().__init__()
                self.projection = projection
                self.patches_per_tile = patches_per_tile

            def forward(self, image):
                patch_height, patch_width = self.projection.stride
                height = image.shape[-2] // patch_height
                width = image.shape[-1] // patch_width
                if height <= self.patches_per_tile and width <= self.patches_per_tile:
                    return self.projection(image)
                embeddings = image.new_empty(
                    (image.shape[0], self.projection.out_channels, height, width)
                )
                for top in range(0, height, self.patches_per_tile):
                    bottom = min(top + self.patches_per_tile, height)
                    for left in range(0, width, self.patches_per_tile):
                        right = min(left + self.patches_per_tile, width)
                        tile = image[
                            :,
                            :,
                            top * patch_height : bottom * patch_height,
                            left * patch_width : right * patch_width,
                        ]
                        embeddings[:, :, top:bottom, left:right] = self.projection(tile)
                return embeddings

        model.patch_embed.proj = TiledPatchProjection(convolution, tile_patches)
    # Official normalization logs must not corrupt a caller's JSON stdout.
    with redirect_stdout(sys.stderr):
        transform = Compose(get_list_norm(config["norm_type"]))
    return torch, model, transform, config


def crop_first_inputs(opened, transform, model, torch):
    """Produce the same five patch-aligned crops as Wrapper5crops without a full tensor."""
    projection = model.patch_embed.proj
    convolution = getattr(projection, "projection", projection)
    patch_height, patch_width = model.patch_embed.grid_size
    stride_height, stride_width = convolution.stride
    if (
        convolution.kernel_size != convolution.stride
        or convolution.padding != (0, 0)
        or convolution.dilation != (1, 1)
    ):
        raise ValueError("Crop-first inference requires independent patch projection")
    embedded_height = opened.height // stride_height
    embedded_width = opened.width // stride_width
    if embedded_height < patch_height or embedded_width < patch_width:
        raise ValueError("Crop-first inference requires at least one full patch crop")
    center_top = (embedded_height - patch_height) // 2
    center_left = (embedded_width - patch_width) // 2
    last_top = embedded_height - patch_height
    last_left = embedded_width - patch_width
    positions = (
        (center_top, center_left),
        (0, 0),
        (last_top, 0),
        (last_top, last_left),
        (0, last_left),
    )
    crops = []
    for top, left in positions:
        box = (
            left * stride_width,
            top * stride_height,
            (left + patch_width) * stride_width,
            (top + patch_height) * stride_height,
        )
        crops.append(transform(opened.crop(box).convert("RGB")))
    return torch.stack(crops)


def infer_crop_first(crops, model):
    embeddings = model.patch_embed.proj(crops)
    if model.patch_embed.flatten:
        embeddings = embeddings.flatten(2).transpose(1, 2)
    embeddings = model.patch_embed.norm(embeddings)
    return model.model(embeddings).mean(dim=0, keepdim=True)


def infer_pil(opened, torch, model, transform, device: str):

    start = time.perf_counter()
    width, height = opened.size
    crop_first = width * height > 8_000_000
    if crop_first:
        image = crop_first_inputs(opened, transform, model, torch)
    else:
        image = transform(opened.convert("RGB")).unsqueeze(0)
    prepared = time.perf_counter()

    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    compute_start = time.perf_counter()
    with torch.inference_mode():
        output = (
            infer_crop_first(image.to(device), model)
            if crop_first
            else model(image.to(device))
        )
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)

    if output.shape[1] == 1:
        score = output[0, 0].item()
    elif output.shape[1] == 2:
        score = (output[0, 1] - output[0, 0]).item()
    else:
        raise ValueError(f"Unsupported B-Free output shape: {tuple(output.shape)}")
    completed = time.perf_counter()
    return {
        "score": score,
        "preprocess_mode": "crop_first_5patch" if crop_first else "full_projection",
        "width": width,
        "height": height,
        "decode_preprocess_ms": (prepared - start) * 1000,
        "gpu_transfer_forward_ms": (completed - compute_start) * 1000,
        "end_to_end_ms": (completed - start) * 1000,
    }


class BFreeDetector:
    name = "B-Free"

    def __init__(
        self, vendor_code: Path, weights_root: Path, *, device="cuda:0", tile_patches=0
    ):
        self.device = device
        self.torch, self.model, self.transform, self.config = load_official_model(
            vendor_code, weights_root, device, tile_patches
        )

    def predict(self, image: RGBImage) -> Prediction:
        validate_rgb(image)
        start = time.perf_counter()
        opened = Image.fromarray(image)
        converted = time.perf_counter()
        result = infer_pil(opened, self.torch, self.model, self.transform, self.device)
        return Prediction(
            method=self.name,
            score=result["score"],
            score_kind="logit",
            timing_ms={
                "preprocess": result["decode_preprocess_ms"]
                + (converted - start) * 1000,
                "forward": result["gpu_transfer_forward_ms"],
            },
            metadata={
                "preprocess_mode": result["preprocess_mode"],
                "protocol": "official five-crop; crop-first above 8M pixels",
            },
        )
