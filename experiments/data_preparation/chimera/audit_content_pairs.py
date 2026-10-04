"""Check released Chimera filename pairing with image-content pHashes."""

from palimpsest.paths import DATA_ROOT, WORK_DIR

import csv
import hashlib
from pathlib import Path
import json

import numpy as np
from PIL import Image
from scipy.fft import dctn


ROOT = DATA_ROOT / "derived/chimera_paired"
MANIFEST = DATA_ROOT / "manifests/chimera_bfree_manifest.csv"
OUTPUT = WORK_DIR / "chimera_content_pair_audit.json"
EXPECTED_MANIFEST_SHA = (
    "39da9e9eac3bd2d133c53769f9c5a158c1769039c67560c0bbfc4660d09a7d24"
)
CLASSES = ("cat", "church", "horse")
LABELS = ("0_real", "1_fake")
TARGETS = ("recap_mac", "recap_monitor")


def phash(path: Path) -> int:
    with Image.open(path) as image:
        grey = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32)
    coefficients = dctn(grey, norm="ortho")[:8, :8].ravel()
    threshold = np.median(coefficients[1:])
    return sum(int(bit) << index for index, bit in enumerate(coefficients > threshold))


def main() -> None:
    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != EXPECTED_MANIFEST_SHA:
        raise RuntimeError("signed inference manifest has changed")
    with MANIFEST.open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    if len(manifest) != 3600:
        raise RuntimeError("expected signed 3600-image manifest")
    hashes = {}
    for row in manifest:
        hashes[row["filename"]] = phash(ROOT / row["filename"])
    results = {}
    outliers = []
    for condition in TARGETS:
        distances = []
        self_rank_one = 0
        self_rank_top_five = 0
        margins = []
        for klass in CLASSES:
            for label in LABELS:
                filenames = sorted(
                    path.name
                    for path in (ROOT / "stylegan2_orig" / klass / label).glob("*.png")
                )
                if len(filenames) != 200:
                    raise RuntimeError("expected 200 filenames per content/label group")
                source_hashes = [
                    hashes[f"stylegan2_orig/{klass}/{label}/{name}"]
                    for name in filenames
                ]
                recap_hashes = [
                    hashes[f"{condition}/{klass}/{label}/{name}"] for name in filenames
                ]
                for index, name in enumerate(filenames):
                    all_distances = [
                        (source_hashes[index] ^ candidate).bit_count()
                        for candidate in recap_hashes
                    ]
                    self_distance = all_distances[index]
                    other_best = min(
                        value
                        for candidate_index, value in enumerate(all_distances)
                        if candidate_index != index
                    )
                    rank = 1 + sum(value < self_distance for value in all_distances)
                    distances.append(self_distance)
                    margins.append(other_best - self_distance)
                    self_rank_one += rank == 1
                    self_rank_top_five += rank <= 5
                    if rank > 1:
                        nearest = filenames[int(np.argmin(all_distances))]
                        outliers.append(
                            {
                                "condition": condition,
                                "source": f"{klass}/{label}/{name}",
                                "self_distance": self_distance,
                                "rank": rank,
                                "nearest_filename": nearest,
                                "nearest_distance": min(all_distances),
                            }
                        )
        results[condition] = {
            "source_count": len(distances),
            "self_phash_distance_median": float(np.median(distances)),
            "self_phash_distance_p95": float(np.quantile(distances, 0.95)),
            "same_filename_strict_rank1": self_rank_one,
            "same_filename_top5": self_rank_top_five,
            "nearest_wrong_minus_same_margin_median": float(np.median(margins)),
        }
    report = {
        "method": "64-bit grayscale DCT pHash; nearest among 200 same class/label recaptures",
        "results": results,
        "outlier_count": len(outliers),
        "outliers": outliers,
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"results": results, "outlier_count": len(outliers)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
