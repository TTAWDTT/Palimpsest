"""Compare native and 256x256 Chimera recaptures under fixed B-Free weights."""

from palimpsest.paths import DATA_ROOT, WORK_DIR

from palimpsest.data.inference import validated_inference

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.origin_detection.score_transfer.chimera.evaluate_bfree import (
    condition_metrics,
    file_sha256,
    paired_metrics,
)


CONDITIONS = ("recap_mac", "recap_monitor")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-manifest",
        type=Path,
        default=DATA_ROOT / "manifests/chimera_bfree_manifest.csv",
    )
    parser.add_argument(
        "--base-csv", type=Path, default=WORK_DIR / "chimera_bfree_full.csv"
    )
    parser.add_argument(
        "--base-summary", type=Path, default=WORK_DIR / "chimera_bfree_full.json"
    )
    parser.add_argument(
        "--control-manifest",
        type=Path,
        default=DATA_ROOT / "manifests/chimera_recap256_bfree_manifest.csv",
    )
    parser.add_argument(
        "--control-csv", type=Path, default=WORK_DIR / "chimera_bfree_recap256.csv"
    )
    parser.add_argument(
        "--control-summary", type=Path, default=WORK_DIR / "chimera_bfree_recap256.json"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=WORK_DIR / "chimera_bfree_size_control_evaluation.json",
    )
    args = parser.parse_args()

    audit = json.loads(
        (WORK_DIR / "chimera_recap256_audit.json").read_text(encoding="utf-8")
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
