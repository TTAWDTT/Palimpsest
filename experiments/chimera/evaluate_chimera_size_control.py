"""Compare native and 256x256 Chimera recaptures under fixed B-Free weights."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.chimera.evaluate_chimera_bfree import (
    condition_metrics,
    file_sha256,
    paired_metrics,
    rows,
)


CONDITIONS = ("recap_mac", "recap_monitor")


def validated_inference(
    manifest_path: Path, csv_path: Path, summary_path: Path, expected_count: int
) -> dict[str, list[dict[str, str]]]:
    manifest = rows(manifest_path)
    inference = rows(csv_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if len(manifest) != expected_count or len(inference) != expected_count:
        raise RuntimeError(f"manifest/inference rows not {expected_count}")
    if (
        summary["completed_images"] != expected_count
        or summary["remaining_images"] != 0
        or summary["errors"] != 0
        or summary["stopping_error"]
    ):
        raise RuntimeError("inference summary incomplete")
    grouped = {}
    seen = set()
    for expected, row in zip(manifest, inference):
        if any(row[field] != expected[field] for field in ("filename", "src", "label")):
            raise RuntimeError(f"identity/order mismatch for {expected['filename']}")
        if (
            row["filename"] in seen
            or row["error"]
            or not math.isfinite(float(row["score"]))
        ):
            raise RuntimeError(
                f"duplicate, error or nonfinite score: {row['filename']}"
            )
        seen.add(row["filename"])
        grouped.setdefault(expected["condition"], []).append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=Path(
            "E:/ai_image_origin_research/data/manifests/chimera_bfree_manifest.csv"
        ),
    )
    parser.add_argument(
        "--base-csv", type=Path, default=Path("work/chimera_bfree_full.csv")
    )
    parser.add_argument(
        "--base-summary", type=Path, default=Path("work/chimera_bfree_full.json")
    )
    parser.add_argument(
        "--control-manifest",
        type=Path,
        default=Path(
            "E:/ai_image_origin_research/data/manifests/chimera_recap256_bfree_manifest.csv"
        ),
    )
    parser.add_argument(
        "--control-csv", type=Path, default=Path("work/chimera_bfree_recap256.csv")
    )
    parser.add_argument(
        "--control-summary", type=Path, default=Path("work/chimera_bfree_recap256.json")
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("work/chimera_bfree_size_control_evaluation.json"),
    )
    args = parser.parse_args()

    audit = json.loads(
        Path("work/chimera_recap256_audit.json").read_text(encoding="utf-8")
    )
    if file_sha256(args.control_manifest) != audit["output_manifest_sha256"]:
        raise RuntimeError("control manifest changed")
    base = validated_inference(
        args.base_manifest, args.base_csv, args.base_summary, 3600
    )
    control = validated_inference(
        args.control_manifest, args.control_csv, args.control_summary, 2400
    )
    if set(control) != set(CONDITIONS) or any(
        len(control[condition]) != 1200 for condition in CONDITIONS
    ):
        raise RuntimeError("control condition coverage incomplete")
    indexed = {
        condition: {row["src"]: row for row in group}
        for condition, group in {
            **base,
            **{f"{k}_256": v for k, v in control.items()},
        }.items()
    }
    rng = np.random.default_rng(20260925)
    result = {
        "scope": "Chimera published ordinary screen recaptures, downsampled to 256x256 source dimensions",
        "method": "Pillow RGB conversion, LANCZOS resize, PNG; no model retraining or threshold tuning",
        "base_manifest_sha256": file_sha256(args.base_manifest),
        "control_manifest_sha256": file_sha256(args.control_manifest),
        "base_inference_csv_sha256": file_sha256(args.base_csv),
        "control_inference_csv_sha256": file_sha256(args.control_csv),
        "original": condition_metrics(base["stylegan2_orig"]),
        "conditions": {},
        "by_scene_class": {
            scene: {
                condition: condition_metrics(
                    [
                        row
                        for row in control[condition]
                        if row["src"].startswith(f"{scene}/")
                    ]
                )
                for condition in CONDITIONS
            }
            for scene in ("cat", "church", "horse")
        },
    }
    for condition in CONDITIONS:
        result["conditions"][condition] = {
            "native": condition_metrics(base[condition]),
            "downsampled_256": condition_metrics(control[condition]),
            "native_to_256": paired_metrics(
                indexed[condition], indexed[f"{condition}_256"], rng
            ),
            "original_to_256": paired_metrics(
                indexed["stylegan2_orig"], indexed[f"{condition}_256"], rng
            ),
        }
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                condition: {
                    version: metrics["zero_threshold_balanced_accuracy"]
                    for version, metrics in result["conditions"][condition].items()
                    if version in ("native", "downsampled_256")
                }
                for condition in CONDITIONS
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
