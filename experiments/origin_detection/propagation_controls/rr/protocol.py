"""Frozen inputs and deterministic digital controls; no score-driven fitting."""

from __future__ import annotations

import csv
import io
import json
import tomllib
from pathlib import Path

import numpy as np
from PIL import Image

from palimpsest.contracts import Origin
from palimpsest.data.rr import load_pairs
from palimpsest.evaluation.cached import ExpectedImage
from palimpsest.io.hashing import file_sha256
from palimpsest.paths import DATA_ROOT, REPO_ROOT, WORK_DIR
from palimpsest.simulation.publication import PublicationParameters, apply_publication
from experiments.origin_detection.platform_statistics.rr.protocol import (
    jpeg_settings,
    render_simulated_jpeg,
)

CONFIG = REPO_ROOT / "configs/evaluation/rr_propagation.toml"
CONTROL_ROOT = DATA_ROOT / "derived/rr_propagation_controls"
RESULT_ROOT = WORK_DIR / "rr_propagation_controls"
CURRENT_ROOT = DATA_ROOT / "derived/rr_simulation_dev_balanced"
CURRENT_INDEX = WORK_DIR / "rr_transfer_simulation_balanced_index.json"
CURRENT_SCORES = WORK_DIR / "rr_transfer_simulation_balanced_bfree.csv"
REGISTRY = DATA_ROOT / "manifests/rr_simulation_source_registry.csv"
CONTROLS = ("fixed_resize_jpeg", "matched_resize_jpeg")
PINNED = {
    CURRENT_INDEX: "61987cb43aa0f22fb751672187ecd142a4a5ecd120e18a89dd871b3ac486c4e4",
    CURRENT_SCORES: "edffa3c1ddb357e8be8eed8a3bb786972abeb4976cd6e013d24e70ecdd153b7d",
    REGISTRY: "1361f527e7b142530ff1363b24271cf40932cf120df701b3ec6cef6531b7fee5",
}


def load_inputs() -> tuple[dict, dict, dict, dict]:
    for path, fingerprint in PINNED.items():
        if file_sha256(path) != fingerprint:
            raise ValueError(f"Frozen input fingerprint mismatch: {path}")
    with CONFIG.open("rb") as stream:
        config = tomllib.load(stream)
    with REGISTRY.open(encoding="utf-8-sig", newline="") as stream:
        roles = {row["source"]: row["status"] for row in csv.DictReader(stream)}
    rows = json.loads(CURRENT_INDEX.read_text(encoding="utf-8"))["rows"]
    current = {row["source"]: row for row in rows}
    selected = {source for source, role in roles.items() if role == "development"}
    if len(rows) != 1000 or len(current) != 1000 or current.keys() != selected:
        raise ValueError("Expected exactly the frozen 1000 development sources")
    if sum(source.startswith("ai/") for source in selected) != 500:
        raise ValueError("Development classes must be balanced")
    for row in rows:
        for name in ("geometry_donor", "encoding_donor"):
            if roles.get(row[name]) != "calibration":
                raise ValueError("Donor must belong to calibration")
    sources = load_pairs()
    return config, sources, current, roles


def control_filename(control: str, current_row: dict) -> str:
    return f"{control}/" + current_row["filename"].split("/", 1)[1]


def render_control(
    original: Image.Image, control: str, row: dict, sources: dict, config: dict
) -> bytes:
    if control == "fixed_resize_jpeg":
        scale = min(1.0, config["max_long_edge"] / max(original.size))
        size = tuple(max(1, round(v * scale)) for v in original.size)
        parameters = PublicationParameters(
            output_size=size,
            resampling=config["resampling"],
            encoding="jpeg",
            jpeg_quality=config["jpeg_quality"],
            jpeg_subsampling=config["jpeg_subsampling"],
        )
        return apply_publication(np.asarray(original), parameters).encoded_bytes
    if control != "matched_resize_jpeg":
        raise ValueError(f"Unknown control: {control}")
    # Reuse the frozen target-independent geometry and calibration codec draw;
    # change only crop to resize. Published target transfer pixels are not used.
    qtables, sampling, fingerprint = jpeg_settings(
        sources[row["encoding_donor"]]["transfer"]
    )
    donor = {
        "width_ratio": row["width"] / original.width,
        "height_ratio": row["height"] / original.height,
        "geometry_mode": "resize",
        "qtables": qtables,
        "subsampling": sampling,
        "jpeg_luma_fingerprint": fingerprint,
    }
    return render_simulated_jpeg(original, donor, 8_000_000)[0]


def expected_images(
    sources: dict, current: dict, *, include_controls: bool
) -> list[ExpectedImage]:
    result = []
    for source, row in current.items():
        label = Origin.AI if source.startswith("ai/") else Origin.NATURAL
        for condition, real_condition in (
            ("original", "original"),
            ("released_transfer", "transfer"),
        ):
            result.append(
                ExpectedImage(
                    source,
                    condition,
                    label,
                    sources[source][real_condition]["filename"],
                )
            )
        result.append(
            ExpectedImage(source, "current_simulation", label, row["filename"])
        )
        if include_controls:
            for control in CONTROLS:
                result.append(
                    ExpectedImage(
                        source, control, label, control_filename(control, row)
                    )
                )
    return result


def write_new_or_verify(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite a different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(path)


def csv_bytes(rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=("filename", "src", "label", "B-Free"))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")
