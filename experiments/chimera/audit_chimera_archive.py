"""Audit official Chimera Zenodo data archive without extracting it wholesale."""

from palimpsest.paths import DATA_ROOT, WORK_DIR

from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import PurePosixPath
import tarfile

from PIL import Image
import numpy as np
from scipy.fft import dctn


ARCHIVE = DATA_ROOT / "raw/chimera_data.tar.gz"
OUTPUT = WORK_DIR / "chimera_archive_audit.json"
EXPECTED_SIZE = 3_784_718_041
EXPECTED_MD5 = "e645697149fd75afe6382131020cc394"
GROUPS = (
    "stylegan2_orig",
    "recap_mac",
    "recap_monitor",
    "attack_mac",
    "attack_monitor",
)
CLASSES = ("cat", "church", "horse")
LABELS = ("0_real", "1_fake")


def perceptual_hash(image: Image.Image) -> int:
    grey = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float32)
    coefficients = dctn(grey, norm="ortho")[:8, :8].ravel()
    threshold = float(np.median(coefficients[1:]))
    bits = coefficients > threshold
    return sum(int(bit) << index for index, bit in enumerate(bits))


def main() -> None:
    size = ARCHIVE.stat().st_size
    if size != EXPECTED_SIZE:
        raise RuntimeError(f"unexpected archive size: {size}")
    digest = hashlib.md5()
    with ARCHIVE.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    actual_md5 = digest.hexdigest()
    if actual_md5 != EXPECTED_MD5:
        raise RuntimeError(f"unexpected archive MD5: {actual_md5}")
    ids = defaultdict(set)
    counts = Counter()
    images = []
    directory_count = 0
    unexpected = []
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"unsafe archive member path: {member.name}")
            if member.isdir():
                directory_count += 1
                continue
            if not member.isfile():
                unexpected.append({"name": member.name, "type": "nonregular"})
                continue
            if (
                len(path.parts) != 4
                or path.parts[0] not in GROUPS
                or path.parts[1] not in CLASSES
                or path.parts[2] not in LABELS
                or path.suffix.lower() != ".png"
            ):
                unexpected.append({"name": member.name, "type": "unexpected_file_path"})
                continue
            key = (path.parts[0], path.parts[1], path.parts[2])
            if path.name in ids[key]:
                raise RuntimeError(f"duplicate filename: {member.name}")
            ids[key].add(path.name)
            counts[key] += 1
            images.append((member.name, member.size))
        if len(images) != 6000:
            raise RuntimeError(f"expected 6000 PNG files, found {len(images)}")
        # Decode two deterministic IDs per content/label from every condition.
        probe_names = []
        for klass in CLASSES:
            for label in LABELS:
                selected = sorted(ids[("stylegan2_orig", klass, label)])[:2]
                for group in GROUPS:
                    probe_names.extend(
                        f"{group}/{klass}/{label}/{filename}" for filename in selected
                    )
    probe_set = set(probe_names)
    probe = []
    # A second sequential pass is faster than random extractfile() seeks in a
    # gzip stream, which otherwise repeatedly decompresses most of the file.
    with tarfile.open(ARCHIVE, "r|gz") as archive:
        for member in archive:
            if member.name not in probe_set:
                continue
            source = archive.extractfile(member)
            assert source is not None
            with Image.open(io.BytesIO(source.read())) as image:
                image.load()
                probe.append(
                    {
                        "name": member.name,
                        "shape": [image.height, image.width],
                        "mode": image.mode,
                        "bytes": member.size,
                        "phash_64": f"{perceptual_hash(image):016x}",
                    }
                )
            if len(probe) == len(probe_set):
                break
    if {item["name"] for item in probe} != probe_set:
        raise RuntimeError("not all selected source/recapture members could be decoded")
    phashes = {item["name"]: int(item["phash_64"], 16) for item in probe}
    matching_distances = []
    swapped_distances = []
    pairwise_order_correct = 0
    for klass in CLASSES:
        for label in LABELS:
            selected = sorted(ids[("stylegan2_orig", klass, label)])[:2]
            for index, filename in enumerate(selected):
                source_hash = phashes[f"stylegan2_orig/{klass}/{label}/{filename}"]
                wrong_name = selected[1 - index]
                for group in GROUPS[1:]:
                    correct = (
                        source_hash ^ phashes[f"{group}/{klass}/{label}/{filename}"]
                    ).bit_count()
                    swapped = (
                        source_hash ^ phashes[f"{group}/{klass}/{label}/{wrong_name}"]
                    ).bit_count()
                    matching_distances.append(correct)
                    swapped_distances.append(swapped)
                    pairwise_order_correct += correct < swapped
    coverage = {}
    for klass in CLASSES:
        for label in LABELS:
            key = f"{klass}/{label}"
            original = ids[("stylegan2_orig", klass, label)]
            coverage[key] = {
                "count_by_group": {
                    group: counts[(group, klass, label)] for group in GROUPS
                },
                "all_five_filename_sets_identical": all(
                    ids[(group, klass, label)] == original for group in GROUPS
                ),
                "missing_by_group": {
                    group: len(original - ids[(group, klass, label)])
                    for group in GROUPS
                },
                "extra_by_group": {
                    group: len(ids[(group, klass, label)] - original)
                    for group in GROUPS
                },
            }
    record = {
        "source": "https://zenodo.org/records/14736478",
        "archive": str(ARCHIVE),
        "archive_size": size,
        "archive_md5": actual_md5,
        "directory_count": directory_count,
        "png_count": len(images),
        "compressed_png_member_sizes": {
            "min": min(image_size for _, image_size in images),
            "max": max(image_size for _, image_size in images),
            "total": sum(image_size for _, image_size in images),
        },
        "coverage": coverage,
        "unexpected_members": unexpected,
        "decoded_probes": probe,
        "sample_content_alignment": {
            "matching_pairs": len(matching_distances),
            "matching_phash_hamming_median": float(np.median(matching_distances)),
            "swapped_same_class_phash_hamming_median": float(
                np.median(swapped_distances)
            ),
            "matching_closer_than_swapped_count": pairwise_order_correct,
            "matching_phash_hamming": matching_distances,
            "swapped_same_class_phash_hamming": swapped_distances,
        },
        "focus_metadata_in_paths": any("focus" in name.lower() for name, _ in images),
    }
    OUTPUT.write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "png_count": len(images),
                "directory_count": directory_count,
                "all_coverage": all(
                    item["all_five_filename_sets_identical"]
                    for item in coverage.values()
                ),
                "decoded_probe_count": len(probe),
                "focus_metadata_in_paths": record["focus_metadata_in_paths"],
                "unexpected_member_count": len(unexpected),
                "probe_shapes": sorted({tuple(item["shape"]) for item in probe}),
                "sample_pairwise_order_correct": pairwise_order_correct,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
