"""Run the unmodified B-Free network with per-image end-to-end timings."""

from __future__ import annotations

from palimpsest.evaluation.timing import percentile
from palimpsest.detection.baselines.bfree import BFreeDetector
from palimpsest.detection.files import predict_file

import argparse
import csv
import json
import statistics
from pathlib import Path


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "filename" not in rows[0]:
        raise ValueError("Input CSV must contain at least one filename")
    return rows


def load_successful_prefix(
    path: Path, manifest: list[dict[str, str]]
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        previous = list(csv.DictReader(handle))
    successful = []
    failed_seen = False
    for index, row in enumerate(previous):
        if (
            index >= len(manifest)
            or row["filename"] != manifest[index]["filename"]
            or row["src"] != manifest[index].get("src", "")
            or row["label"] != manifest[index].get("label", "")
        ):
            raise ValueError(f"Resume file differs from manifest at row {index}")
        if row["error"] or not row["score"]:
            failed_seen = True
        elif failed_seen:
            raise ValueError("Resume file has a success after its first failure")
        else:
            successful.append(row)
    return successful


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
    previous = (
        load_successful_prefix(args.resume_from, rows) if args.resume_from else []
    )
    new_rows = rows[len(previous) :]
    if args.max_new_images:
        new_rows = new_rows[: args.max_new_images]
    detector = BFreeDetector(
        args.vendor_code,
        args.weights_root,
        device=args.device,
        tile_patches=args.tile_patches,
    )
    torch, config = detector.torch, detector.config
    dataset_root = args.dataset_root.resolve()

    def image_path_from_row(row: dict[str, str]) -> Path:
        image_path = (dataset_root / row["filename"]).resolve()
        if not image_path.is_relative_to(dataset_root):
            raise ValueError(f"Image path leaves dataset root: {row['filename']}")
        return image_path

    for row in new_rows[: args.warmup]:
        image_path = image_path_from_row(row)
        if image_path.is_file():
            predict_file(detector, image_path)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "src",
        "label",
        "published_bfree_score",
        "score",
        "width",
        "height",
        "preprocess_mode",
        "decode_preprocess_ms",
        "gpu_transfer_forward_ms",
        "end_to_end_ms",
        "error",
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
                file_result = predict_file(detector, image_path)
                prediction = file_result.prediction
                result.update(
                    {
                        "score": prediction.score,
                        "width": file_result.width,
                        "height": file_result.height,
                        "preprocess_mode": prediction.metadata["preprocess_mode"],
                        "decode_preprocess_ms": file_result.decode_ms
                        + prediction.timing_ms["preprocess"],
                        "gpu_transfer_forward_ms": prediction.timing_ms["forward"],
                        "end_to_end_ms": file_result.end_to_end_ms,
                    }
                )
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
        "decode_preprocess_ms",
        "gpu_transfer_forward_ms",
        "end_to_end_ms",
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
