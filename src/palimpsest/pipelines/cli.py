"""Run explicit region or whole-image original-content inference."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from palimpsest.paths import ResearchPaths
from palimpsest.pipelines.image import ImageOriginPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--detector", choices=("bfree", "d3", "benford"), required=True)
    parser.add_argument(
        "--localizer", choices=("quadrilateral", "sam2", "whole-image"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--vendor", type=Path, help="Override official B-Free code or D3 checkout"
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="B-Free weights directory, D3 CLIP checkpoint or fitted Benford NPZ",
    )
    parser.add_argument("--d3-head", type=Path)
    parser.add_argument("--sam2-checkpoint", type=Path)
    parser.add_argument("--max-regions", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        help="New JSON path (refuses overwrite); defaults to stdout",
    )
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if args.max_regions < 1:
        parser.error("max-regions must be positive")
    paths = ResearchPaths.load()
    start = perf_counter()
    if args.detector == "bfree":
        from palimpsest.detection.baselines.bfree import BFreeDetector

        detector = BFreeDetector(
            args.vendor or paths.work / "vendor/bfree/code",
            args.weights or paths.models / "bfree",
            device=args.device,
        )
    elif args.detector == "d3":
        from palimpsest.detection.baselines.d3 import D3Detector

        vendor = args.vendor or paths.work / "vendor/d3"
        detector = D3Detector(
            vendor,
            args.weights or paths.models / "d3/ViT-L-14.pt",
            args.d3_head or vendor / "ckpt/classifier.pth",
            device=args.device,
        )
    else:
        from palimpsest.detection.baselines.benford import BenfordRFDetector

        if args.weights is None:
            parser.error(
                "Benford requires an explicitly fitted --weights NPZ; no implicit training"
            )
        detector = BenfordRFDetector.load(args.weights)
    if args.localizer == "quadrilateral":
        from palimpsest.localization.baselines.quadrilateral import QuadrilateralLocator

        locator = QuadrilateralLocator(max_regions=args.max_regions)
    elif args.localizer == "sam2":
        from palimpsest.localization.baselines.sam2 import SAM2Locator

        locator = SAM2Locator(
            args.sam2_checkpoint or paths.models / "sam2/sam2.1_hiera_tiny.pt",
            device=args.device,
            max_regions=args.max_regions,
        )
    else:
        locator = None
    loaded = perf_counter()
    result = ImageOriginPipeline(detector, locator).predict_file(args.image).to_dict()
    result["input"] = str(args.image.resolve())
    result["timing_ms"]["model_load"] = (loaded - start) * 1000
    result["timing_ms"]["cold_call"] = (perf_counter() - start) * 1000
    result["timing_scope"] = (
        "end_to_end includes decode/localize/crop/predict; excludes model load and JSON/output I/O; cold_call includes model load"
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
