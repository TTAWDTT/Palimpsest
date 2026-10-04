"""Audit how cropped PNGs relate by name to DFD color whole-sheet TIFFs.

Name matches are packaging relationships, not source-to-print digital pairs.
"""

from __future__ import annotations

from palimpsest.paths import WORK_DIR

import json
import re
from collections import Counter, defaultdict
from pathlib import PurePosixPath


INVENTORY = WORK_DIR / "dfd_color_inventory.json"
OUT = WORK_DIR / "dfd_color_links.json"
CROP_SUFFIX = re.compile(r"_c\d+(?:_\d+x\d+)?$", re.IGNORECASE)


def main() -> None:
    files = json.loads(INVENTORY.read_text(encoding="utf-8"))["files"]
    tiffs = [row for row in files if row["extension"] == ".tiff"]
    pngs = [row for row in files if row["extension"] == ".png"]
    tiff_keys = {(row["folder"], PurePosixPath(row["name"]).stem) for row in tiffs}
    crop_parents = defaultdict(list)
    unmatched_png = []
    for row in pngs:
        stem = PurePosixPath(row["name"]).stem
        parent = CROP_SUFFIX.sub("", stem)
        if parent == stem:
            unmatched_png.append(row["name"])
        else:
            crop_parents[(row["folder"], parent)].append(row["name"])
    no_whole_sheet = sorted(key for key in crop_parents if key not in tiff_keys)
    no_crop = sorted(key for key in tiff_keys if key not in crop_parents)
    suffix_counter = Counter()
    for row in pngs:
        name = PurePosixPath(row["name"]).stem
        match = CROP_SUFFIX.search(name)
        if match:
            suffix_counter[
                "resized_128x128" if "128x128" in match.group(0) else "native_crop"
            ] += 1
        else:
            suffix_counter["other_png"] += 1
    report = {
        "whole_sheet_tiff_count": len(tiffs),
        "png_count": len(pngs),
        "crop_parent_count": len(crop_parents),
        "crop_suffix_counts": dict(suffix_counter),
        "crop_parents_without_same_named_tiff_count": len(no_whole_sheet),
        "crop_parents_without_same_named_tiff": no_whole_sheet,
        "tiff_without_same_named_crops_count": len(no_crop),
        "tiff_without_same_named_crops": no_crop,
        "unmatched_png_count": len(unmatched_png),
        "unmatched_png_examples": unmatched_png[:20],
        "tiff_by_folder": dict(sorted(Counter(row["folder"] for row in tiffs).items())),
        "png_by_folder": dict(sorted(Counter(row["folder"] for row in pngs).items())),
        "first_tiff_inventory_index": next(
            i for i, row in enumerate(files) if row["extension"] == ".tiff"
        ),
    }
    OUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
