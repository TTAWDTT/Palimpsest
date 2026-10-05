"""Materialize two prespecified digital controls and audit image diagnostics."""

from __future__ import annotations

import hashlib
import io
import json
import time
from collections import Counter

import numpy as np
from PIL import Image, JpegImagePlugin

from palimpsest.data.images import resize_unit_float
from palimpsest.data.rr import ROOT
from palimpsest.evaluation.image_pairs import image_metrics
from palimpsest.io.hashing import file_sha256
from experiments.origin_detection.propagation_controls.rr.protocol import (
    CONFIG,
    CURRENT_INDEX,
    CURRENT_ROOT,
    CONTROL_ROOT,
    RESULT_ROOT,
    CONTROLS,
    control_filename,
    csv_bytes,
    load_inputs,
    render_control,
    write_new_or_verify,
)


def inspect_bytes(payload: bytes) -> dict:
    with Image.open(io.BytesIO(payload)) as opened:
        return {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "width": opened.width,
            "height": opened.height,
            "subsampling": JpegImagePlugin.get_sampling(opened),
            "jpeg_luma_fingerprint": hashlib.sha256(
                np.asarray(opened.quantization[0], dtype=np.uint16).tobytes()
            ).hexdigest()[:16],
        }


def features(original: Image.Image, payload: bytes, side: int) -> dict:
    with Image.open(io.BytesIO(payload)) as image:
        return {
            **image_metrics(
                resize_unit_float(original, side),
                resize_unit_float(image.convert("RGB"), side),
            ),
            "width_ratio": image.width / original.width,
            "height_ratio": image.height / original.height,
        }


def main() -> None:
    config, sources, current, _ = load_inputs()
    manifests = {name: [] for name in CONTROLS}
    indexes = {name: [] for name in CONTROLS}
    measurements, orientations, parity = [], Counter(), Counter()
    for number, (source, row) in enumerate(current.items(), 1):
        original_row = sources[source]["original"]
        if file_sha256(ROOT / original_row["filename"]) != original_row["sha256"]:
            raise ValueError(f"Original image fingerprint mismatch: {source}")
        with Image.open(ROOT / original_row["filename"]) as opened:
            orientation = opened.getexif().get(274, 1)
            orientations[orientation] += 1
            if orientation != 1:
                raise ValueError(
                    "Nontrivial EXIF orientation requires a new aligned protocol"
                )
            original = opened.convert("RGB")
        released_row = sources[source]["transfer"]
        released = (ROOT / released_row["filename"]).read_bytes()
        if hashlib.sha256(released).hexdigest() != released_row["sha256"]:
            raise ValueError(f"Released image fingerprint mismatch: {source}")
        existing = (CURRENT_ROOT / row["filename"]).read_bytes()
        if hashlib.sha256(existing).hexdigest() != row["sha256"]:
            raise ValueError(f"Current simulation image fingerprint mismatch: {source}")
        payloads = {"released_transfer": released, "current_simulation": existing}
        for control in CONTROLS:
            start = time.perf_counter()
            payload = render_control(original, control, row, sources, config)
            elapsed = (time.perf_counter() - start) * 1000
            filename = control_filename(control, row)
            write_new_or_verify(CONTROL_ROOT / filename, payload)
            info = {"source": source, "filename": filename, **inspect_bytes(payload)}
            indexes[control].append(info)
            manifests[control].append(
                {
                    "filename": filename,
                    "src": source,
                    "label": "FAKE" if source.startswith("ai/") else "REAL",
                    "B-Free": "",
                }
            )
            payloads[control] = payload
            if control == "matched_resize_jpeg":
                equal = payload == existing
                parity[
                    f"{row['geometry_mode']}/{'identical' if equal else 'different'}"
                ] += 1
                if row["geometry_mode"] == "resize" and not equal:
                    raise ValueError(
                        f"Resize ablation should be byte-identical: {source}"
                    )
            measurements.append(
                {"source": source, "condition": control, "render_ms": elapsed}
            )
        for condition, payload in payloads.items():
            measurements.append(
                {
                    "source": source,
                    "condition": condition,
                    "features": features(original, payload, config["diagnostic_side"]),
                    "jpeg_luma_fingerprint": inspect_bytes(payload)[
                        "jpeg_luma_fingerprint"
                    ],
                }
            )
        if number % 100 == 0:
            print(f"Prepared {number}/{len(current)} sources", flush=True)
    for control in CONTROLS:
        write_new_or_verify(
            RESULT_ROOT / f"{control}.manifest.csv", csv_bytes(manifests[control])
        )
        write_new_or_verify(
            RESULT_ROOT / f"{control}.index.json",
            (
                json.dumps(
                    {
                        "config_sha256": file_sha256(CONFIG),
                        "current_index_sha256": file_sha256(CURRENT_INDEX),
                        "images": indexes[control],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8"),
        )
    # Timings are observations and change across reruns; no immutable claim.
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "diagnostics.json").write_text(
        json.dumps(
            {
                "config": config,
                "config_sha256": file_sha256(CONFIG),
                "original_exif_orientations": dict(orientations),
                "matched_control_byte_parity": dict(parity),
                "measurements": measurements,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"sources": len(current), "byte_parity": dict(parity)}))


if __name__ == "__main__":
    main()
