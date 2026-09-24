"""Run the unmodified B-Free network with per-image end-to-end timings."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "filename" not in rows[0]:
        raise ValueError("Input CSV must contain at least one filename")
    return rows


def load_successful_prefix(path: Path, manifest: list[dict[str, str]]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        previous = list(csv.DictReader(handle))
    successful = []
    failed_seen = False
    for index, row in enumerate(previous):
        if (index >= len(manifest)
                or row["filename"] != manifest[index]["filename"]
                or row["src"] != manifest[index].get("src", "")
                or row["label"] != manifest[index].get("label", "")):
            raise ValueError(f"Resume file differs from manifest at row {index}")
        if row["error"] or not row["score"]:
            failed_seen = True
        elif failed_seen:
            raise ValueError("Resume file has a success after its first failure")
        else:
            successful.append(row)
    return successful


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
        if (convolution.kernel_size != convolution.stride or
                convolution.padding != (0, 0) or
                convolution.dilation != (1, 1)):
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
                            :, :, top * patch_height:bottom * patch_height,
                            left * patch_width:right * patch_width,
                        ]
                        embeddings[:, :, top:bottom, left:right] = self.projection(tile)
                return embeddings

        model.patch_embed.proj = TiledPatchProjection(convolution, tile_patches)
    transform = Compose(get_list_norm(config["norm_type"]))
    return torch, model, transform, config


def crop_first_inputs(opened, transform, model, torch):
    """Produce the same five patch-aligned crops as Wrapper5crops without a full tensor."""
    projection = model.patch_embed.proj
    convolution = getattr(projection, "projection", projection)
    patch_height, patch_width = model.patch_embed.grid_size
    stride_height, stride_width = convolution.stride
    if (convolution.kernel_size != convolution.stride
            or convolution.padding != (0, 0)
            or convolution.dilation != (1, 1)):
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
        (center_top, center_left), (0, 0), (last_top, 0),
        (last_top, last_left), (0, last_left),
    )
    crops = []
    for top, left in positions:
        box = (
            left * stride_width, top * stride_height,
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


def infer_image(image_path: Path, torch, model, transform, device: str):
    from PIL import Image

    start = time.perf_counter()
    with Image.open(image_path) as opened:
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
        output = infer_crop_first(image.to(device), model) if crop_first else model(image.to(device))
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-code", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tile-patches", type=int, default=0)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--max-new-images", type=int, default=0)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    previous = load_successful_prefix(args.resume_from, rows) if args.resume_from else []
    new_rows = rows[len(previous):]
    if args.max_new_images:
        new_rows = new_rows[:args.max_new_images]
    torch, model, transform, config = load_official_model(
        args.vendor_code, args.weights_root, args.device, args.tile_patches
    )
    dataset_root = args.dataset_root.resolve()

    def image_path_from_row(row: dict[str, str]) -> Path:
        image_path = (dataset_root / row["filename"]).resolve()
        if not image_path.is_relative_to(dataset_root):
            raise ValueError(f"Image path leaves dataset root: {row['filename']}")
        return image_path

    for row in new_rows[: args.warmup]:
        image_path = image_path_from_row(row)
        if image_path.is_file():
            infer_image(image_path, torch, model, transform, args.device)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename", "src", "label", "published_bfree_score", "score", "width", "height",
        "preprocess_mode",
        "decode_preprocess_ms", "gpu_transfer_forward_ms", "end_to_end_ms", "error",
    ]
    completed = list(previous)
    errors = 0
    stopping_error = None
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in previous:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
        for row in new_rows:
            result = {
                "filename": row["filename"],
                "src": row.get("src", ""),
                "label": row.get("label", ""),
                "published_bfree_score": row.get("B-Free", ""),
                "error": "",
            }
            try:
                image_path = image_path_from_row(row)
                result.update(infer_image(image_path, torch, model, transform, args.device))
                completed.append(result)
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                errors += 1
                stopping_error = result["error"]
            writer.writerow(result)
            handle.flush()
            if stopping_error:
                break

    summary = {
        "model": "BFREE_dino2reg4",
        "config": config,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": args.device,
        "input_images": len(rows),
        "completed_images": len(completed),
        "errors": errors,
        "remaining_images": len(rows) - len(completed),
        "warmup_images": min(args.warmup, len(new_rows)),
        "tile_patches": args.tile_patches,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "crop_first_threshold_pixels": 8_000_000,
        "stopping_error": stopping_error,
    }
    for timing_key in (
        "decode_preprocess_ms", "gpu_transfer_forward_ms", "end_to_end_ms"
    ):
        values = [float(item[timing_key]) for item in completed]
        if values:
            summary[timing_key] = {
                "p50": statistics.median(values),
                "p95": percentile(values, 0.95),
            }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if stopping_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
