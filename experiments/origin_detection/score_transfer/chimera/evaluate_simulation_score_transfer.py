"""Compare fixed B-Free source-score changes on real vs composite proxy recaptures."""

from palimpsest.paths import DATA_ROOT, WORK_DIR

import json

import numpy as np

from experiments.origin_detection.score_transfer.chimera.evaluate_bfree import (
    condition_metrics,
    file_sha256,
    paired_metrics,
)
from palimpsest.data.inference import validated_inference


BASE_MANIFEST = DATA_ROOT / "manifests/chimera_bfree_manifest.csv"
REAL_MANIFEST = DATA_ROOT / "manifests/chimera_recap256_bfree_manifest.csv"
SIM_MANIFEST = DATA_ROOT / "manifests/chimera_composite_sim_dev256_manifest.csv"
AUDIT = WORK_DIR / "chimera_composite_sim_dev256_audit.json"
OUTPUT = WORK_DIR / "chimera_composite_sim_score_transfer.json"


def index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["src"]: row for row in rows}


def transfer(
    original: dict[str, dict], real: dict[str, dict], sim: dict[str, dict]
) -> dict:
    ids = sorted(sim)
    if set(ids) != set(original) or set(ids) != set(real) or len(ids) != 120:
        raise RuntimeError("source IDs not aligned for development comparisons")
    o = np.array([float(original[key]["score"]) for key in ids])
    r = np.array([float(real[key]["score"]) for key in ids])
    s = np.array([float(sim[key]["score"]) for key in ids])
    truth = np.array([original[key]["label"] == "FAKE" for key in ids])
    if any(
        real[key]["label"] != original[key]["label"]
        or sim[key]["label"] != original[key]["label"]
        for key in ids
    ):
        raise RuntimeError("source labels differ")
    r_delta, s_delta = r - o, s - o
    real_degradation = ((o > 0) == truth) & ((r > 0) != truth)
    return {
        "source_count": 120,
        "real_score_shift_mean": float(r_delta.mean()),
        "sim_score_shift_mean": float(s_delta.mean()),
        "real_sim_score_mae": float(np.abs(r - s).mean()),
        "no_change_score_mae": float(np.abs(r - o).mean()),
        "real_sim_shift_pearson": float(np.corrcoef(r_delta, s_delta)[0, 1]),
        "real_sim_prediction_agreement": float(np.mean((r > 0) == (s > 0))),
        "real_correct_to_wrong": int(real_degradation.sum()),
        "real_correct_to_wrong_also_reproduced": int(
            (real_degradation & ((s > 0) != truth)).sum()
        ),
        "finite": bool(np.isfinite([o, r, s]).all()),
    }


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if file_sha256(SIM_MANIFEST) != audit["manifest_sha256"]:
        raise RuntimeError("simulation manifest changed")
    base = validated_inference(
        BASE_MANIFEST,
        WORK_DIR / "chimera_bfree_full.csv",
        WORK_DIR / "chimera_bfree_full.json",
        3600,
    )
    real = validated_inference(
        REAL_MANIFEST,
        WORK_DIR / "chimera_bfree_recap256.csv",
        WORK_DIR / "chimera_bfree_recap256.json",
        2400,
    )
    sim = validated_inference(
        SIM_MANIFEST,
        WORK_DIR / "chimera_composite_sim_bfree_dev.csv",
        WORK_DIR / "chimera_composite_sim_bfree_dev.json",
        240,
    )
    if set(sim) != {"recap_mac", "recap_monitor"}:
        raise RuntimeError("simulation condition coverage changed")
    result = {
        "scope": "120 development sources per device; fixed B-Free score transfer, real 256px recapture vs composite proxy",
        "warning": "composite proxy is fitted in published RGB space, not validated physical capture process",
        "manifest_sha256": file_sha256(SIM_MANIFEST),
        "sim_inference_csv_sha256": file_sha256(
            WORK_DIR / "chimera_composite_sim_bfree_dev.csv"
        ),
        "conditions": {},
    }
    rng = np.random.default_rng(20260925)
    for condition in ("recap_mac", "recap_monitor"):
        original_index = index(base["stylegan2_orig"])
        real_index = index(real[condition])
        sim_index = index(sim[condition])
        ids = set(sim_index)
        original_dev = [original_index[key] for key in sorted(ids)]
        real_dev = [real_index[key] for key in sorted(ids)]
        sim_dev = [sim_index[key] for key in sorted(ids)]
        item = {
            "original": condition_metrics(original_dev),
            "real_recap256": condition_metrics(real_dev),
            "composite_sim256": condition_metrics(sim_dev),
            "original_to_real": paired_metrics(
                {key: original_index[key] for key in ids},
                {key: real_index[key] for key in ids},
                rng,
            ),
            "original_to_sim": paired_metrics(
                {key: original_index[key] for key in ids},
                {key: sim_index[key] for key in ids},
                rng,
            ),
            "real_to_sim": paired_metrics(
                {key: real_index[key] for key in ids},
                {key: sim_index[key] for key in ids},
                rng,
            ),
            "transfer": transfer(
                {key: original_index[key] for key in ids},
                {key: real_index[key] for key in ids},
                {key: sim_index[key] for key in ids},
            ),
        }
        if not item["transfer"]["finite"] or len(sim_dev) != 120:
            raise RuntimeError("nonfinite or incomplete simulation scores")
        result["conditions"][condition] = item
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                condition: {
                    "real_ba": item["real_recap256"][
                        "zero_threshold_balanced_accuracy"
                    ],
                    "sim_ba": item["composite_sim256"][
                        "zero_threshold_balanced_accuracy"
                    ],
                    "score_mae": item["transfer"]["real_sim_score_mae"],
                }
                for condition, item in result["conditions"].items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
