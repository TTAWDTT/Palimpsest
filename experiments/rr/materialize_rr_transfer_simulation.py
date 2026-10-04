"""Write development-set transfer simulations for fixed-detector score tests."""

from __future__ import annotations

from experiments.paths import REPO_ROOT

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageOps

from experiments.rr.analyze_rr_simulation_inputs import ROOT, load_pairs
from experiments.rr.rr_simulator_v0 import (
    choose_donor,
    choose_encoding_donor,
    fit_donors,
    render_simulated_jpeg,
    split_sources,
)


BASE = REPO_ROOT
DEVELOPMENT = BASE / "work" / "rr_simulator_crop_independent_1000_evaluation.json"
OUTPUT_ROOT = Path(r"E:\ai_image_origin_research\data\derived\rr_simulation_dev")
MANIFEST = Path(
    r"E:\ai_image_origin_research\data\manifests\rr_transfer_simulation_dev_bfree.csv"
)
REGISTRY = Path(
    r"E:\ai_image_origin_research\data\manifests\rr_simulation_source_registry.csv"
)
INDEX = BASE / "work" / "rr_transfer_simulation_dev_index.json"
MAX_PIXELS = 8_000_000
NEIGHBORS = 30


def write_once(path: Path, payload: bytes) -> None:
    """Create an immutable artifact atomically, or verify its existing bytes."""
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing artifact differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=DEVELOPMENT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--index", type=Path, default=INDEX)
    args = parser.parse_args()
    evaluation = json.loads(args.evaluation.read_text(encoding="utf-8"))
    if (
        evaluation["geometry"],
        evaluation["jpeg_profile"],
        evaluation["max_pixels"],
        evaluation["neighbors"],
    ) != ("crop-aware", "independent", MAX_PIXELS, NEIGHBORS):
        raise ValueError("development settings mismatch")
    selected = list(
        dict.fromkeys(row["source"] for row in evaluation["per_source_features"])
    )
    selected_hash = hashlib.sha256("\n".join(selected).encode()).hexdigest()
    if (
        len(selected) != 1000
        or selected_hash != evaluation["validation_source_ids_sha256"]
    ):
        raise ValueError("development source list mismatch")
    with REGISTRY.open(encoding="utf-8", newline="") as handle:
        registered_development = {
            row["source"]
            for row in csv.DictReader(handle)
            if row["status"] == "development"
        }
    if set(selected) != registered_development:
        raise ValueError("selected sources do not match frozen development registry")
    sources = load_pairs()
    calibration, _ = split_sources(sources)
    if calibration.keys() & set(selected):
        raise ValueError("calibration/development overlap")
    donors = fit_donors(
        calibration,
        evaluation["donor_count_per_condition"]["transfer"],
        "crop-aware",
        evaluation.get("balanced_donors", False),
    )
    donor_hash = hashlib.sha256(
        "\n".join(donor["source"] for donor in donors["transfer"]).encode()
    ).hexdigest()
    if donor_hash != evaluation["donor_source_ids_sha256"]["transfer"]:
        raise ValueError("re-fitted donor sources differ from evaluated simulator")
    parameter_hash = hashlib.sha256(
        json.dumps(donors["transfer"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected_parameter_hash = evaluation.get("donor_parameters_sha256", {}).get(
        "transfer"
    )
    if (
        expected_parameter_hash is not None
        and parameter_hash != expected_parameter_hash
    ):
        raise ValueError("re-fitted donor parameters differ from evaluated simulator")
    rows = []
    index = []
    for source in selected:
        original_row = sources[source]["original"]
        geometry_donor = choose_donor(
            source,
            original_row,
            "transfer",
            donors,
            NEIGHBORS,
            evaluation.get("geometry_donor_sampling", "size-neighbor"),
        )
        encoding_donor = choose_encoding_donor(
            source, "transfer", donors, evaluation.get("balanced_donors", False)
        )
        with Image.open(ROOT / original_row["filename"]) as opened:
            original = ImageOps.exif_transpose(opened).convert("RGB")
        payload, width, height, fingerprint = render_simulated_jpeg(
            original, geometry_donor, MAX_PIXELS, encoding_donor
        )
        label, _ = source.split("/", 1)
        filename = f"transfer/{label}/{hashlib.sha256(source.encode()).hexdigest()}.jpg"
        target = args.output_root / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        write_once(target, payload)
        rows.append(
            {
                "filename": filename,
                "src": source,
                "label": "FAKE" if label == "ai" else "REAL",
                "B-Free": "",
            }
        )
        index.append(
            {
                "source": source,
                "filename": filename,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "width": width,
                "height": height,
                "geometry_donor": geometry_donor["source"],
                "geometry_mode": geometry_donor["geometry_mode"],
                "encoding_donor": encoding_donor["source"],
                "jpeg_luma_fingerprint": fingerprint,
            }
        )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=("filename", "src", "label", "B-Free"))
    writer.writeheader()
    writer.writerows(rows)
    write_once(args.manifest, buffer.getvalue().encode("utf-8"))
    index_payload = (
        json.dumps(
            {
                "evaluation_sha256": hashlib.sha256(
                    args.evaluation.read_bytes()
                ).hexdigest(),
                "source_ids_sha256": selected_hash,
                "rows": index,
            },
            indent=2,
        )
        + "\n"
    )
    write_once(args.index, index_payload.replace("\n", "\r\n").encode("utf-8"))
    print(
        json.dumps(
            {
                "images": len(rows),
                "manifest": str(args.manifest),
                "manifest_sha256": hashlib.sha256(
                    args.manifest.read_bytes()
                ).hexdigest(),
                "output_root": str(args.output_root),
            }
        )
    )


if __name__ == "__main__":
    main()
