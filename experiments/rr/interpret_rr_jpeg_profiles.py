"""Match published RR JPEG quantization profiles to Pillow encodings."""

from __future__ import annotations

from experiments.paths import REPO_ROOT

import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


BASE = REPO_ROOT
AUDIT = BASE / "work" / "rr_transformation_provenance_audit.json"
OUTPUT = BASE / "work" / "rr_jpeg_profile_interpretation.json"


def fingerprint(quantization: dict[int, list[int]]) -> str:
    payload = b"".join(
        np.asarray(quantization[index], dtype=np.uint16).tobytes()
        for index in sorted(quantization)
    )
    return hashlib.sha256(payload).hexdigest()[:16]


def pillow_reference_profiles() -> dict[str, int]:
    image = Image.new("RGB", (8, 8), "white")
    profiles = {}
    for quality in range(1, 101):
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=quality, subsampling=2)
        stream.seek(0)
        with Image.open(stream) as encoded:
            profiles[fingerprint(encoded.quantization)] = quality
    return profiles


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    standards = pillow_reference_profiles()
    matched = {}
    for condition, profile in audit["conditions"].items():
        quality_counts: Counter[str] = Counter()
        for identifier, count in profile["qtables"].items():
            quality_counts[str(standards.get(identifier, "unmatched"))] += count
        if sum(quality_counts.values()) != sum(profile["qtables"].values()):
            raise ValueError(f"quality count mismatch: {condition}")
        matched[condition] = {
            "quality_counts": dict(quality_counts),
            "unmatched_fingerprints": {
                identifier: count
                for identifier, count in profile["qtables"].items()
                if identifier not in standards
            },
        }
    OUTPUT.write_text(json.dumps(matched, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(matched, indent=2))


if __name__ == "__main__":
    main()
