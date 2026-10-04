"""Screen DESCAN Valid/Test clean patches for possible near duplicates."""

from __future__ import annotations

from experiments.paths import REPO_ROOT

import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn


BASE = REPO_ROOT
ARCHIVE_ROOT = Path(r"E:\ai_image_origin_research\data\raw\descan18k")
OUTPUT = BASE / "work" / "descan_near_duplicate_screen.json"


def fingerprints(split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    names, phashes, color_vectors = [], [], []
    with zipfile.ZipFile(ARCHIVE_ROOT / f"{split}.zip") as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.startswith(f"{split}/clean/") and name.lower().endswith(".tif")
        )
        for member in members:
            with Image.open(io.BytesIO(archive.read(member))) as image:
                gray = np.asarray(
                    image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
                    dtype=np.float32,
                )
                rgb = (
                    np.asarray(
                        image.convert("RGB").resize((32, 32), Image.Resampling.LANCZOS),
                        dtype=np.float32,
                    )
                    / 255
                )
            coefficients = dctn(gray, type=2, norm="ortho")[:8, :8].ravel()[1:]
            phashes.append(coefficients > np.median(coefficients))
            color_vectors.append(rgb.ravel())
            names.append(Path(member).name)
    return names, np.stack(phashes), np.stack(color_vectors)


def main() -> None:
    valid_names, valid_hashes, valid_vectors = fingerprints("Valid")
    test_names, test_hashes, test_vectors = fingerprints("Test")
    hamming = np.count_nonzero(valid_hashes[:, None] != test_hashes[None], axis=2)
    valid_unit = valid_vectors / np.linalg.norm(valid_vectors, axis=1, keepdims=True)
    test_unit = test_vectors / np.linalg.norm(test_vectors, axis=1, keepdims=True)
    cosine = valid_unit @ test_unit.T
    sorted_pairs = sorted(
        (
            (int(hamming[i, j]), float(cosine[i, j]), valid_names[i], test_names[j])
            for i in range(len(valid_names))
            for j in range(len(test_names))
        )
    )
    closest_color_pairs = sorted(sorted_pairs, key=lambda pair: pair[1], reverse=True)
    output = {
        "scope": "candidate screen only; perceptual hash/cosine cannot prove source independence",
        "valid_images": len(valid_names),
        "test_images": len(test_names),
        "phash_hamming_pair_counts_at_most": {
            str(limit): int(np.count_nonzero(hamming <= limit))
            for limit in (4, 8, 12, 16)
        },
        "minimum_phash_hamming": int(hamming.min()),
        "maximum_resized_rgb_cosine": float(cosine.max()),
        "closest_phash_pairs": [
            {
                "hamming": distance,
                "resized_rgb_cosine": similarity,
                "valid_clean": valid,
                "test_clean": test,
            }
            for distance, similarity, valid, test in sorted_pairs[:30]
        ],
        "closest_color_pairs": [
            {
                "hamming": distance,
                "resized_rgb_cosine": similarity,
                "valid_clean": valid,
                "test_clean": test,
            }
            for distance, similarity, valid, test in closest_color_pairs[:30]
        ],
    }
    OUTPUT.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in output.items()
                if key not in {"closest_phash_pairs", "closest_color_pairs"}
            },
            indent=2,
        )
    )
    print("top matches:", output["closest_phash_pairs"][:5])
    print("top color matches:", output["closest_color_pairs"][:5])


if __name__ == "__main__":
    main()
