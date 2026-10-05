"""Run only new B-Free control images, with explicit foreground subprocesses."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

from palimpsest.io.hashing import file_sha256
from palimpsest.paths import MODELS_ROOT, REPO_ROOT, WORK_DIR
from experiments.origin_detection.propagation_controls.rr.protocol import (
    CONFIG,
    CONTROLS,
    CONTROL_ROOT,
    RESULT_ROOT,
    load_inputs,
)


def code_fingerprints() -> dict:
    paths = [
        "src/palimpsest/contracts.py",
        "src/palimpsest/detection/files.py",
        "src/palimpsest/detection/baselines/bfree.py",
        "experiments/origin_detection/baselines/run_bfree.py",
        "experiments/origin_detection/propagation_controls/rr/run_controls.py",
    ]
    return {path: file_sha256(REPO_ROOT / path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Audit a failed run and resume only its valid prefix",
    )
    args = parser.parse_args()
    load_inputs()
    weights = MODELS_ROOT / "bfree"
    weight_path = weights / "BFREE_dino2reg4/model_epoch_best.pth"
    weight_sha = file_sha256(weight_path)
    if weight_sha != "5948ca78f4d94e820c250d24cdf155035b4a85960443800bfe6bb7f06bffe947":
        raise ValueError("B-Free weights differ from the historical fixed detector")
    vendor = WORK_DIR / "vendor/bfree/code"
    vendor_hashes = {
        path.relative_to(vendor).as_posix(): file_sha256(path)
        for path in sorted(vendor.rglob("*.py"))
    }
    if not vendor_hashes:
        raise ValueError("Missing B-Free official code")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    for control in CONTROLS:
        manifest = RESULT_ROOT / f"{control}.manifest.csv"
        output = RESULT_ROOT / f"{control}.csv"
        summary_path = RESULT_ROOT / f"{control}.summary.json"
        provenance_path = RESULT_ROOT / f"{control}.run.json"
        signature = {
            "manifest_sha256": file_sha256(manifest),
            "index_sha256": file_sha256(RESULT_ROOT / f"{control}.index.json"),
            "config_sha256": file_sha256(CONFIG),
            "weight_sha256": weight_sha,
            "weight_config_sha256": file_sha256(weight_path.parent / "config.yaml"),
            "code_sha256": code_fingerprints(),
            "vendor_code_sha256": vendor_hashes,
            "hardware": "CUDA device recorded below; no concurrent control inference",
            "protocol": "B-Free official five crops, batch1, tile64, TF32 off, warmup20",
        }
        resume_audit = None
        if provenance_path.exists():
            previous = json.loads(provenance_path.read_text(encoding="utf-8"))
            # The orchestrator may gain recovery code; the scoring implementation,
            # vendor, weights and every image/manifest signature must stay fixed.
            runner_path = (
                "experiments/origin_detection/propagation_controls/rr/run_controls.py"
            )
            inference_code = {
                key: value
                for key, value in signature["code_sha256"].items()
                if key != runner_path
            }
            previous_code = {
                key: value
                for key, value in previous["code_sha256"].items()
                if key != runner_path
            }
            if previous_code != inference_code or any(
                previous.get(key) != value
                for key, value in signature.items()
                if key != "code_sha256"
            ):
                raise ValueError(f"Existing run signature differs: {control}")
            if (
                previous.get("status") == "complete"
                and file_sha256(output) == previous["output_csv_sha256"]
            ):
                if file_sha256(summary_path) != previous["summary_sha256"]:
                    raise ValueError("Completed summary fingerprint changed")
                print(f"Reuse completed {control}; no inference", flush=True)
                continue
            if not args.resume or previous.get("status") != "failed":
                raise ValueError(
                    f"Incomplete run requires explicit audited --resume: {control}"
                )
            if (
                file_sha256(output) != previous["output_csv_sha256"]
                or file_sha256(summary_path) != previous["summary_sha256"]
            ):
                raise ValueError("Failed CSV/summary fingerprint changed")
            from experiments.origin_detection.baselines.run_bfree import (
                load_manifest,
                load_successful_prefix,
            )

            prefix = load_successful_prefix(output, load_manifest(manifest))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if len(prefix) != summary["completed_images"]:
                raise ValueError(
                    "Failed summary disagrees with successful prefix length"
                )
            for row in prefix:
                if not math.isfinite(float(row["score"])) or any(
                    not math.isfinite(float(row[key])) or float(row[key]) < 0
                    for key in (
                        "decode_preprocess_ms",
                        "gpu_transfer_forward_ms",
                        "end_to_end_ms",
                    )
                ):
                    raise ValueError("Nonfinite score/timing in successful prefix")
            index = json.loads(
                (RESULT_ROOT / f"{control}.index.json").read_text(encoding="utf-8")
            )
            for image in index["images"]:
                if file_sha256(CONTROL_ROOT / image["filename"]) != image["sha256"]:
                    raise ValueError("Control image changed before recovery")
            resume_audit = {
                "previous_run": previous,
                "failed_summary": summary,
                "preserved_successful_rows": len(prefix),
                "inference_code_unchanged": True,
                "image_hashes_verified": len(index["images"]),
            }
            print(
                f"Audited recovery of {control}: preserve {len(prefix)} successful rows",
                flush=True,
            )
        if output.exists() and resume_audit is None:
            raise ValueError(f"Untrusted output already exists: {output}")
        import torch

        signature.update(
            {
                "git_revision": revision,
                "working_tree_dirty": bool(dirty),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "status": "running",
            }
        )
        if resume_audit is not None:
            signature["resume_audit"] = resume_audit
        provenance_path.write_text(
            json.dumps(signature, indent=2) + "\n", encoding="utf-8"
        )
        command = [
            sys.executable,
            "-m",
            "experiments.origin_detection.baselines.run_bfree",
            "--vendor-code",
            str(vendor),
            "--weights-root",
            str(weights),
            "--dataset-root",
            str(CONTROL_ROOT),
            "--manifest",
            str(manifest),
            "--output-csv",
            str(output),
            "--summary-json",
            str(summary_path),
            "--tile-patches",
            "64",
            "--warmup",
            "20",
        ]
        if resume_audit is not None:
            command.extend(["--resume-from", str(output)])
        env = os.environ | {
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        remaining = 1000 - (
            resume_audit["preserved_successful_rows"] if resume_audit else 0
        )
        print(
            f"Start {control}: {remaining} remaining images, foreground child process",
            flush=True,
        )
        process = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
        signature["exit_code"] = process.returncode
        signature["command"] = command
        signature["status"] = "failed"
        if output.exists():
            signature["output_csv_sha256"] = file_sha256(output)
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            signature["summary_sha256"] = file_sha256(summary_path)
            if (
                process.returncode == 0
                and summary["completed_images"] == 1000
                and summary["errors"] == 0
                and summary["remaining_images"] == 0
            ):
                signature["status"] = "complete"
        provenance_path.write_text(
            json.dumps(signature, indent=2) + "\n", encoding="utf-8"
        )
        if signature["status"] != "complete":
            raise RuntimeError(f"Control run failed or incomplete: {control}")
    print("Both control runs complete; child processes have exited", flush=True)


if __name__ == "__main__":
    main()
