"""Full CRC, content-pair, and metadata audit for official DIV2K-SCAN XR test.

Requires the complete ETH DIV2K_valid_HR.zip. Never treats matching file IDs
alone as content validation. Input archives stay on E:; JSON stays in work/.
"""

from __future__ import annotations

from palimpsest.io.hashing import file_sha256 as _sha256

from palimpsest.paths import WORK_DIR

import hashlib
import io
import json
import zipfile

import numpy as np
from PIL import Image

from experiments.data_preparation.div2k.audit_scan_test import (
    RAW,
    _correspondence,
    _source_3_2,
)


OUTPUT = WORK_DIR / "div2k_scan_100_pair_audit.json"
EXPECTED_SOURCE_BYTES = 448_993_893
EXPECTED_CAPTURE_BYTES = 315_574_703
MIN_SIFT_INLIERS_FOR_CONTENT_EVIDENCE = 12


def main() -> None:
    source_path = RAW / "DIV2K_valid_HR.zip"
    capture_path = RAW / "test_xr.zip"
    if source_path.stat().st_size != EXPECTED_SOURCE_BYTES:
        raise ValueError("source archive not at official expected byte count")
    if capture_path.stat().st_size != EXPECTED_CAPTURE_BYTES:
        raise ValueError("capture archive byte count changed")
    expected_ids = [f"{i:04d}" for i in range(801, 901)]
    rows = []
    with (
        zipfile.ZipFile(source_path) as sources,
        zipfile.ZipFile(capture_path) as captures,
    ):
        source_names = sorted(
            info.filename
            for info in sources.infolist()
            if info.filename.endswith(".png")
        )
        capture_names = sorted(
            info.filename
            for info in captures.infolist()
            if info.filename.endswith(".png")
        )
        if source_names != [f"DIV2K_valid_HR/{name}.png" for name in expected_ids]:
            raise ValueError("source member inventory does not match 0801-0900")
        if capture_names != [f"xr/{name}.png" for name in expected_ids]:
            raise ValueError("capture member inventory does not match 0801-0900")
        bad_source, bad_capture = sources.testzip(), captures.testzip()
        if bad_source is not None or bad_capture is not None:
            raise ValueError(
                f"ZIP member CRC failure: source={bad_source}, capture={bad_capture}"
            )
        for image_id in expected_ids:
            source_bytes = sources.read(f"DIV2K_valid_HR/{image_id}.png")
            capture_bytes = captures.read(f"xr/{image_id}.png")
            with Image.open(io.BytesIO(source_bytes)) as source_image:
                with Image.open(io.BytesIO(capture_bytes)) as capture_image:
                    if capture_image.size != (2040, 1360):
                        raise ValueError(f"unexpected capture size: {image_id}")
                    prepared = _source_3_2(source_image, capture_image.size)
                    camera = np.asarray(capture_image.convert("RGB"))
                    source_size = source_image.size
                    source_mode = source_image.mode
                    source_metadata = sorted(source_image.info.keys())
                    capture_metadata = sorted(capture_image.info.keys())
            correspondence = _correspondence(prepared, camera)
            rows.append(
                {
                    "id": image_id,
                    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
                    "capture_sha256": hashlib.sha256(capture_bytes).hexdigest(),
                    "source_size": source_size,
                    "source_mode": source_mode,
                    "source_metadata_keys": source_metadata,
                    "capture_metadata_keys": capture_metadata,
                    "correspondence": correspondence,
                    "content_evidence": correspondence["inliers"]
                    >= MIN_SIFT_INLIERS_FOR_CONTENT_EVIDENCE,
                }
            )
            print(image_id, correspondence["inliers"], flush=True)
    result = {
        "source_archive": str(source_path),
        "source_archive_bytes": EXPECTED_SOURCE_BYTES,
        "source_archive_sha256": _sha256(source_path),
        "capture_archive": str(capture_path),
        "capture_archive_bytes": EXPECTED_CAPTURE_BYTES,
        "capture_archive_sha256": _sha256(capture_path),
        "all_members_crc_passed": True,
        "expected_ids": expected_ids,
        "sift_content_evidence_threshold": MIN_SIFT_INLIERS_FOR_CONTENT_EVIDENCE,
        "content_evidence_count": sum(row["content_evidence"] for row in rows),
        "rows": rows,
    }
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("content evidence", result["content_evidence_count"], "/", len(rows))


if __name__ == "__main__":
    main()
