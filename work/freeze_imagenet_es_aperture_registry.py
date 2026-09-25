"""Freeze the 20 ImageNet-ES aperture-development groups with local file hashes."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


MANIFEST = Path("work/imagenet_es_20class_aperture_manifest.json")
OUTPUT = Path(r"E:\ai_image_origin_research\data\manifests\imagenet_es_aperture_development.csv")
EXPECTED_PARAMS = (4, 13, 22)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--role", choices=("development", "content_holdout"), default="development")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_params = tuple(manifest["param_ids"])
    if expected_params not in (EXPECTED_PARAMS, (4, 5, 13, 14, 22, 23)):
        raise ValueError("unexpected aperture/ISO parameter set")
    if len(manifest["source_classes"]) != 20:
        raise ValueError("expected 20 source classes")
    if args.role == "content_holdout" and manifest.get("class_offset") != 20:
        raise ValueError("content holdout must use the frozen class offset 20")
    groups = defaultdict(dict)
    for item in manifest["rows"]:
        member = item["member"]
        parts = member.split("/")
        source_key = (parts[-2], parts[-1])
        param_id = 0 if "sampled_tin_no_resize2" in parts else int(parts[4].split("_")[1])
        if param_id in groups[source_key]:
            raise ValueError(f"duplicate {source_key} {param_id}")
        path = Path(item["local_path"])
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"missing or changed local image {path}")
        groups[source_key][param_id] = item
    if len(groups) != 20 or len(manifest["rows"]) != 20 * (len(expected_params) + 1):
        raise ValueError("unexpected source or file count")
    rows = []
    for (cls, basename), group in sorted(groups.items()):
        if set(group) != {0, *expected_params}:
            raise ValueError(f"incomplete source group {cls}/{basename}")
        for param_id, item in sorted(group.items()):
            rows.append({
                "role": args.role,
                "source_key": f"{cls}/{basename}",
                "class_id": cls,
                "param_id": param_id,
                "f_number": {0: "", 4: "5", 5: "5", 13: "9", 14: "9", 22: "16", 23: "16"}[param_id],
                "iso_nominal": "" if param_id == 0 else ("2000" if param_id in (5, 14, 23) else "250"),
                "shutter_seconds_nominal": "" if param_id == 0 else "1/60",
                "light_condition": "" if param_id == 0 else "l5",
                "member": item["member"],
                "local_path": item["local_path"],
                "sha256": item["sha256"],
                "bytes": item["bytes"],
                "width": item["width"],
                "height": item["height"],
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"registry": str(args.output), "source_groups": len(groups),
                      "images": len(rows), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()},
                     indent=2))


if __name__ == "__main__":
    main()
