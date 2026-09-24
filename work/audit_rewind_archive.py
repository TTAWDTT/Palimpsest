"""Match ReWIND's image archive to its official CSV and verify extracted MD5s."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def read_expected(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected = {row["filename"]: row["md5"].lower() for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Expected manifest has repeated filenames")
    return expected


def archive_name(info: zipfile.ZipInfo) -> str:
    name = info.filename.replace("\\", "/")
    while name.startswith("./"):
        name = name[2:]
    parts = PurePosixPath(name).parts
    if name.startswith("/") or not parts or ".." in parts:
        raise ValueError(f"Unsafe archive path: {info.filename}")
    return "/".join(parts)


def hash_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-csv", type=Path, required=True)
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()

    expected = read_expected(args.expected_csv)
    with zipfile.ZipFile(args.archive) as archive:
        files = {}
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = archive_name(info)
            if name in files:
                raise ValueError(f"Duplicate archive member: {name}")
            files[name] = info

        missing = sorted(set(expected) - set(files))
        extra = sorted(set(files) - set(expected))
        summary = {
            "expected_images": len(expected),
            "archive_files": len(files),
            "missing_count": len(missing),
            "extra_count": len(extra),
            "missing_examples": missing[:20],
            "extra_examples": extra[:20],
            "md5_mismatches": [],
            "extracted_and_verified": 0,
            "already_verified": 0,
        }

        if not missing:
            extract_root = args.extract_root.resolve()
            for filename, official_md5 in expected.items():
                target = (extract_root / filename).resolve()
                if not target.is_relative_to(extract_root):
                    raise ValueError(f"CSV path leaves extraction root: {filename}")
                if target.is_file() and hash_file(target) == official_md5:
                    summary["already_verified"] += 1
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".partial")
                digest = hashlib.md5()
                with archive.open(files[filename]) as source, temporary.open("wb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
                        digest.update(chunk)
                if digest.hexdigest() != official_md5:
                    summary["md5_mismatches"].append(filename)
                    temporary.unlink()
                    continue
                temporary.replace(target)
                summary["extracted_and_verified"] += 1

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if missing or summary["md5_mismatches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
