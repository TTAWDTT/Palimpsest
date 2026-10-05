"""Check existing weights against historical scores and exercise real region chains.

Only two D3 sequence-prefix inputs, two B-Free shared inputs and one historical
large-image input are recomputed. No accuracy estimation or research fitting.
"""

import argparse
import csv
import importlib.metadata
import itertools
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image
from palimpsest.paths import ResearchPaths
from palimpsest.io.hashing import file_sha256
from palimpsest.detection.baselines.bfree import BFreeDetector
from palimpsest.detection.baselines.d3 import D3Detector
from palimpsest.detection.files import predict_file
from palimpsest.localization.baselines.quadrilateral import QuadrilateralLocator
from palimpsest.localization.baselines.sam2 import SAM2Locator
from palimpsest.pipelines.image import ImageOriginPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = ResearchPaths.load()
    root = paths.data / "derived/rr_test"
    with (paths.work / "rr_bfree_complete.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)
        bfree_cases = list(itertools.islice(reader, 2))
        bfree_cases.append(
            next(r for r in reader if r["preprocess_mode"] == "crop_first_5patch")
        )
    with (paths.work / "d3_rr_full.csv").open(newline="", encoding="utf-8") as handle:
        d3_cases = list(itertools.islice(csv.DictReader(handle), 2))
    start = time.perf_counter()
    detector = BFreeDetector(
        paths.work / "vendor/bfree/code", paths.models / "bfree", tile_patches=64
    )
    bfree_load_ms = (time.perf_counter() - start) * 1000
    bfree = []
    for row in bfree_cases:
        result = predict_file(detector, root / row["filename"])
        difference = abs(result.prediction.score - float(row["score"]))
        bfree.append(
            {
                "filename": row["filename"],
                "sha256": file_sha256(root / row["filename"]),
                "historical_score": float(row["score"]),
                "new_score": result.prediction.score,
                "absolute_difference": difference,
                "preprocess": result.prediction.metadata["preprocess_mode"],
                "end_to_end_ms": result.end_to_end_ms,
            }
        )
        if difference > 1e-4:
            raise ValueError(f"B-Free parity failed: {bfree[-1]}")
    with Image.open(root / bfree_cases[0]["filename"]) as loaded:
        source = np.asarray(loaded.convert("RGB"))
    scene = np.zeros((650, 760, 3), np.uint8)
    scene[60:576, 120:636] = 255
    scene[62:574, 122:634] = source
    # The boundary is an integration fixture, not a real recapture.
    fixture = args.output.with_suffix(".scene.png")
    if fixture.exists():
        raise FileExistsError(fixture)
    fixture.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(scene).save(fixture)
    region_results = {}
    for locator in (
        QuadrilateralLocator(max_regions=2),
        SAM2Locator(paths.models / "sam2/sam2.1_hiera_tiny.pt", max_regions=2),
    ):
        result = ImageOriginPipeline(detector, locator).predict_file(fixture)
        if not result.regions:
            raise ValueError(f"No regions in integration fixture: {locator.name}")
        region_results[locator.name] = result.to_dict()
    del detector, locator
    import torch

    torch.cuda.empty_cache()
    start = time.perf_counter()
    detector = D3Detector(
        paths.work / "vendor/d3",
        paths.models / "d3/ViT-L-14.pt",
        paths.work / "vendor/d3/ckpt/classifier.pth",
    )
    d3_load_ms = (time.perf_counter() - start) * 1000
    d3 = []
    for row in d3_cases:
        before = torch.get_rng_state().clone()
        result = predict_file(detector, root / row["filename"])
        if not torch.equal(torch.get_rng_state(), before):
            raise ValueError("D3 inference changed caller CPU RNG")
        difference = abs(result.prediction.score - float(row["score"]))
        d3.append(
            {
                "filename": row["filename"],
                "historical_score": float(row["score"]),
                "new_score": result.prediction.score,
                "absolute_difference": difference,
                "end_to_end_ms": result.end_to_end_ms,
            }
        )
        if difference > 1e-4:
            raise ValueError(f"D3 parity failed: {d3[-1]}")
        # A separate caller's draws must not perturb the next shuffle input.
        torch.rand(13)
        torch.rand(13, device="cuda")
    payload = {
        "scope": "integration/parity only; no accuracy claim; no real recapture fixture",
        "tolerance": 1e-4,
        "model_load_ms": {"bfree": bfree_load_ms, "d3": d3_load_ms},
        "bfree": bfree,
        "d3": d3,
        "region_chains": region_results,
        "scene_sha256": file_sha256(fixture),
        "sam2_code_commit": subprocess.check_output(
            ["git", "-C", str(paths.work / "vendor/sam2"), "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "sam2_checkpoint_sha256": file_sha256(
            paths.models / "sam2/sam2.1_hiera_tiny.pt"
        ),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "Pillow",
                "torch",
                "torchvision",
                "timm",
                "scipy",
                "opencv-python-headless",
                "SAM-2",
            )
        },
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "bfree_max_difference": max(r["absolute_difference"] for r in bfree),
                "d3_max_difference": max(r["absolute_difference"] for r in d3),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
