"""Test whether one fixed post-crop affine warp transfers to unseen content."""

import json
from pathlib import Path

import cv2
import numpy as np

from work.probe_chimera_registration import CONTROL, ROOT, load_gray, zncc


AUDIT = Path("work/chimera_registration_nonreserved.json")
OUTPUT = Path("work/chimera_fixed_publication_geometry.json")


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {"median": float(np.median(array)), "p05": float(np.quantile(array, .05)),
            "p95": float(np.quantile(array, .95))}


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit["pair_count"] != 720 or audit["success_count"] != 720:
        raise RuntimeError("full nonreserved registration audit is required")
    result = {
        "scope": "one median affine per published device channel; 240 calibration, 120 development sources",
        "warning": "post-crop content registration only, not physical camera pose or a predictor of raw capture",
        "conditions": {},
    }
    for condition in ("recap_mac", "recap_monitor"):
        calibration = [item for item in audit["records"]
                       if item["condition"] == condition and item["split"] == "calibration"]
        development = [item for item in audit["records"]
                       if item["condition"] == condition and item["split"] == "development"]
        if len(calibration) != 240 or len(development) != 120:
            raise RuntimeError("unexpected calibration/development sizes")
        median_warp = np.median(np.asarray([item["warp"] for item in calibration]), axis=0).astype(np.float32)
        identity, fixed, oracle = [], [], []
        for item in development:
            source = item["src"]
            original = load_gray(ROOT / "stylegan2_orig" / source)
            recapture = load_gray(CONTROL / condition / source)
            aligned = cv2.warpAffine(recapture, median_warp, (256, 256),
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REFLECT)
            identity.append(zncc(original, recapture))
            fixed.append(zncc(original, aligned))
            oracle.append(item["aligned_zncc"])
        result["conditions"][condition] = {
            "calibration_pairs": len(calibration), "development_pairs": len(development),
            "median_calibration_affine": median_warp.tolist(),
            "development_zncc": {"identity": summarize(identity), "fixed_calibration_warp": summarize(fixed),
                                 "per_pair_oracle_warp": summarize(oracle)},
            "fixed_improves_over_identity_count": int((np.asarray(fixed) > np.asarray(identity)).sum()),
            "fixed_within_0_01_of_oracle_count": int((np.asarray(oracle) - np.asarray(fixed) <= .01).sum()),
        }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["conditions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
