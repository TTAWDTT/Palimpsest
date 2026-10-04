"""Check patch-aligned crop-first B-Free logits against the original forward."""

import csv
import json
from pathlib import Path

from PIL import Image

from experiments.baselines.run_bfree_baseline import (
    crop_first_inputs,
    infer_crop_first,
    load_official_model,
)


ROOT = Path("E:/ai_image_origin_research/data/derived/rr_test")
MANIFEST = Path("E:/ai_image_origin_research/data/manifests/rr_test_files.csv")
WEIGHTS = Path("E:/ai_image_origin_research/models/bfree")
VENDOR = Path("work/vendor/bfree/code")
OUTPUT = Path("work/bfree_crop_first_validation.json")


def main():
    with MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    cases = []
    for image_format in ("JPEG", "PNG"):
        candidates = [
            row
            for row in rows
            if row["format"] == image_format
            and 8_000_000 < int(row["width"]) * int(row["height"]) < 20_000_000
        ]
        cases.append(
            min(candidates, key=lambda row: int(row["width"]) * int(row["height"]))
        )
    torch, model, transform, config = load_official_model(VENDOR, WEIGHTS, "cuda:0", 64)
    results = []
    for row in cases:
        with Image.open(ROOT / row["filename"]) as opened:
            full_input = transform(opened.convert("RGB")).unsqueeze(0).to("cuda:0")
            crops = crop_first_inputs(opened, transform, model, torch).to("cuda:0")
        with torch.inference_mode():
            full_output = model(full_input)
            cropped_output = infer_crop_first(crops, model)
        difference = (full_output - cropped_output).abs().max().item()
        results.append(
            {
                "filename": row["filename"],
                "format": row["format"],
                "width": row["width"],
                "height": row["height"],
                "full_score": full_output[0].tolist(),
                "crop_first_score": cropped_output[0].tolist(),
                "max_absolute_logit_difference": difference,
            }
        )
    result = {"model": config["model_name"], "cases": results}
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if any(item["max_absolute_logit_difference"] > 1e-4 for item in results):
        raise ValueError("Crop-first validation exceeds the logit tolerance")


if __name__ == "__main__":
    main()
