"""Probe content registration of Chimera published recaptures at common size.

This estimates a *publication-image alignment*, not camera pose or screen pitch.
Only calibration/development source groups are eligible; reserved is unopened.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("E:/ai_image_origin_research/data/derived/chimera_paired")
CONTROL = Path("E:/ai_image_origin_research/data/derived/chimera_recap256")
SPLIT = Path("E:/ai_image_origin_research/data/manifests/chimera_simulation_source_split.csv")


def load_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (256, 256):
        raise RuntimeError(f"cannot read 256x256 image: {path}")
    image = cv2.GaussianBlur(image, (5, 5), .7)
    return image.astype(np.float32) / 255


def zncc(a: np.ndarray, b: np.ndarray) -> float:
    av = a - a.mean()
    bv = b - b.mean()
    return float((av * bv).sum() / max(np.linalg.norm(av) * np.linalg.norm(bv), 1e-12))


def align(src: np.ndarray, recap: np.ndarray) -> dict:
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
    initial = zncc(src, recap)
    try:
        ecc, warp = cv2.findTransformECC(src, recap, warp, cv2.MOTION_AFFINE, criteria,
                                         inputMask=None, gaussFiltSize=5)
    except cv2.error as error:
        return {"success": False, "error": str(error).splitlines()[0], "initial_zncc": initial}
    aligned = cv2.warpAffine(recap, warp, (256, 256), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REFLECT)
    final = zncc(src, aligned)
    linear = warp[:, :2]
    singular = np.linalg.svd(linear, compute_uv=False)
    reasonable = bool(np.all((singular >= .8) & (singular <= 1.2)) and
                      np.max(np.abs(warp[:, 2])) <= 24 and final >= initial)
    return {"success": reasonable, "ecc": float(ecc), "initial_zncc": initial,
            "aligned_zncc": final, "singular_values": singular.tolist(),
            "translation_xy": warp[:, 2].tolist(), "warp": warp.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-stratum", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("work/chimera_registration_pilot.json"))
    args = parser.parse_args()
    with SPLIT.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    candidates = {}
    for row in rows:
        if row["split"] in ("calibration", "development"):
            candidates.setdefault((row["scene"], row["label"]), []).append(row["src"])
    if len(candidates) != 6:
        raise RuntimeError("expected six scene/label strata")
    selected = [source for stratum, values in sorted(candidates.items())
                for source in sorted(values)[:args.per_stratum]]
    split_for_source = {row["src"]: row["split"] for row in rows}
    records = []
    for source in selected:
        src = load_gray(ROOT / "stylegan2_orig" / source)
        for condition in ("recap_mac", "recap_monitor"):
            recap = load_gray(CONTROL / condition / source)
            records.append({"src": source, "split": split_for_source[source],
                            "condition": condition, **align(src, recap)})
    summary = {}
    for split in ("calibration", "development"):
        summary[split] = {}
        for condition in ("recap_mac", "recap_monitor"):
            subset = [item for item in records if item["split"] == split and item["condition"] == condition]
            valid = [item for item in subset if item["success"]]
            entry = {"pairs": len(subset), "reasonable_affine_count": len(valid)}
            if valid:
                entry.update({
                    "initial_zncc_median": float(np.median([item["initial_zncc"] for item in valid])),
                    "aligned_zncc_median": float(np.median([item["aligned_zncc"] for item in valid])),
                    "singular_value_1_median": float(np.median([item["singular_values"][0] for item in valid])),
                    "singular_value_2_median": float(np.median([item["singular_values"][1] for item in valid])),
                    "translation_norm_median_px": float(np.median([
                        np.linalg.norm(item["translation_xy"]) for item in valid])),
                })
            summary[split][condition] = entry
    result = {
        "scope": "pilot content registration, not physical device pose",
        "per_stratum": args.per_stratum,
        "source_count": len(selected),
        "pair_count": len(records),
        "success_count": sum(record["success"] for record in records),
        "summary": summary,
        "records": records,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("source_count", "pair_count", "success_count")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
