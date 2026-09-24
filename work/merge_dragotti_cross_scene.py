"""Package verified Dragotti cross-scene probe records for inspection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"E:\ai_image_origin_research\data\derived\dragotti_probe")
OLD = ROOT / "outputs/03_过程模拟/Dragotti跨设备配对审计_2026-09-24.json"
OUT = ROOT / "outputs/03_过程模拟/Dragotti跨场景几何复核_2026-09-24.json"


def verify(row: dict) -> dict:
    path = Path(row["file"])
    data = path.read_bytes()
    assert len(data) == row["bytes"], path
    assert hashlib.sha256(data).hexdigest() == row["sha256"], path
    return row


def main() -> None:
    old = json.loads(OLD.read_text(encoding="utf-8"))
    samples = []
    for sample in old["samples"]:
        if sample["source_key"] != "D40-015":
            continue
        samples.append({
            "source_key": sample["source_key"],
            "scene": "indoor shelf, chart and fruit",
            "original": verify(sample["original"]),
            "recaptured": [verify(row) for row in sample["recaptured"]
                           if row["camera"] in ("EOS600D", "D3200")],
        })
    for key, scene in (("D40-025", "outdoor fountain"),
                       ("D40-034", "museum whale exhibit")):
        original = json.loads((DATA / f"{key}_original_manifest.json").read_text(encoding="utf-8"))["records"]
        recaptured = json.loads((DATA / f"{key}_recaptured_manifest.json").read_text(encoding="utf-8"))["records"]
        audit = json.loads((ROOT / f"work/dragotti_{key.replace('-', '_')}_audit.json").read_text(encoding="utf-8"))
        assert len(original) == 1 and len(recaptured) == len(audit["records"]) == 2
        assert {r["sha256"] for r in recaptured} == {r["sha256"] for r in audit["records"]}
        samples.append({
            "source_key": key,
            "scene": scene,
            "original": verify(original[0]),
            "recaptured": [dict(verify(next(x for x in recaptured if x["sha256"] == row["sha256"])),
                                **{k: v for k, v in row.items() if k not in ("file", "sha256")})
                           for row in audit["records"]],
        })
    result = {
        "date": "2026-09-24",
        "scope": "three different scenes from one original camera, two recapture cameras; selected ZIP members only",
        "archive_verified": False,
        "member_verification": "length, ZIP CRC-32, SHA-256",
        "registration": old["registration"],
        "samples": samples,
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for sample in samples:
        print(sample["source_key"], sample["scene"],
              [(r["camera"], round(r["conditional_projected_screen_pitch_if_fit_1080_rows"], 6))
               for r in sample["recaptured"]])


if __name__ == "__main__":
    main()
