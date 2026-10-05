"""Official SAM 2 automatic mask generation, adapted to RegionLocator.

Candidates are class-agnostic segments, not certified screens/paper or AI
objects. Requires an explicitly installed official package and local checkpoint.
"""

from pathlib import Path
import numpy as np
from palimpsest.contracts import Box, RGBImage, Region, validate_rgb


class SAM2Locator:
    name = "sam2.1_automatic"

    def __init__(
        self,
        checkpoint: Path,
        *,
        device="cuda:0",
        points_per_side=16,
        max_regions=8,
        min_area_fraction=0.02,
    ):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if max_regions < 1 or points_per_side < 1 or not 0 < min_area_fraction < 1:
            raise ValueError("Invalid SAM2 proposal parameters")
        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        self.torch = torch
        self.device = torch.device(device)
        model = build_sam2(
            "configs/sam2.1/sam2.1_hiera_t.yaml",
            str(checkpoint),
            device=device,
            apply_postprocessing=False,
        )
        # No custom CUDA postprocessing kernel is required for this configuration.
        self.generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=points_per_side,
            points_per_batch=64,
            output_mode="binary_mask",
            min_mask_region_area=0,
        )
        self.max_regions = max_regions
        self.min_area_fraction = min_area_fraction

    def locate(self, image: RGBImage) -> tuple[Region, ...]:
        validate_rgb(image)
        with self.torch.inference_mode():
            records = self.generator.generate(image)
            if self.device.type == "cuda":
                self.torch.cuda.synchronize(self.device)
        records = [
            r
            for r in records
            if r["area"] >= self.min_area_fraction * image.shape[0] * image.shape[1]
        ]
        # Stable, explicit selection; these quality scores concern segmentation.
        records.sort(
            key=lambda r: (-float(r["predicted_iou"]), -r["area"], tuple(r["bbox"]))
        )
        regions = []
        for record in records[: self.max_regions]:
            mask = np.asarray(record["segmentation"], dtype=bool)
            ys, xs = np.nonzero(mask)
            box = Box(
                int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            )
            region = Region(
                str(len(regions)),
                box,
                "class_agnostic_segment",
                confidence=float(np.clip(record["predicted_iou"], 0, 1)),
                mask=mask,
            )
            region.validate_shape(image.shape[:2])
            regions.append(region)
        return tuple(regions)
