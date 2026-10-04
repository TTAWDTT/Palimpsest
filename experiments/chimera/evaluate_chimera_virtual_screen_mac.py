"""Evaluate one frozen virtual forward chain against paired real Chimera Mac recaptures."""

from palimpsest.paths import DATA_ROOT, WORK_DIR

import json
from pathlib import Path

import numpy as np

from experiments.chimera.evaluate_chimera_bfree import (
    condition_metrics,
    file_sha256,
    paired_metrics,
)
from palimpsest.data.inference import validated_inference
from experiments.chimera.evaluate_chimera_simulation_score_transfer import transfer
from experiments.chimera.probe_chimera_screen_observables import (
    features,
    numeric_summary,
    ROOT,
    CONTROL,
)


BASE_MANIFEST = DATA_ROOT / "manifests/chimera_bfree_manifest.csv"
REAL_MANIFEST = DATA_ROOT / "manifests/chimera_recap256_bfree_manifest.csv"
SIM_MANIFEST = DATA_ROOT / "manifests/chimera_virtual_screen_mac_dev256_manifest.csv"
AUDIT = WORK_DIR / "chimera_virtual_screen_mac_dev256_audit.json"
SIM_CSV = WORK_DIR / "chimera_virtual_screen_mac_dev256_bfree.csv"
SIM_SUMMARY = WORK_DIR / "chimera_virtual_screen_mac_dev256_bfree.json"
CONTROL_MANIFEST = (
    DATA_ROOT / "manifests/chimera_virtual_screen_mac_dev256_co_spatial_manifest.csv"
)
CONTROL_AUDIT = WORK_DIR / "chimera_virtual_screen_mac_dev256_co_spatial_audit.json"
CONTROL_CSV = WORK_DIR / "chimera_virtual_screen_mac_co_spatial_dev256_bfree.csv"
CONTROL_SUMMARY = WORK_DIR / "chimera_virtual_screen_mac_co_spatial_dev256_bfree.json"
FIT_MANIFEST = (
    DATA_ROOT / "manifests/chimera_virtual_screen_mac_dev256_effective_fit_manifest.csv"
)
FIT_AUDIT = WORK_DIR / "chimera_virtual_screen_mac_dev256_effective_fit_audit.json"
FIT_CSV = WORK_DIR / "chimera_virtual_screen_mac_effective_fit_dev256_bfree.csv"
FIT_SUMMARY = WORK_DIR / "chimera_virtual_screen_mac_effective_fit_dev256_bfree.json"
FIT_GRID = WORK_DIR / "chimera_virtual_screen_mac_effective_fit.json"
OUTPUT = WORK_DIR / "chimera_virtual_screen_mac_score_transfer.json"


def index(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    keyed = {row["src"]: row for row in rows}
    if len(keyed) != len(rows):
        raise RuntimeError("duplicate source IDs")
    return keyed


def main() -> None:
    def verify_audit(audit_path: Path, manifest_path: Path) -> dict:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit["output_count"] != 120 or audit["manifest_sha256"] != file_sha256(
            manifest_path
        ):
            raise RuntimeError(
                "simulation materialization audit is incomplete or changed"
            )
        if (
            len(audit["image_sha256"]) != 120
            or len({row["filename"] for row in audit["image_sha256"]}) != 120
        ):
            raise RuntimeError("simulation image fingerprint list is incomplete")
        root = Path(audit["output_root"])
        for row in audit["image_sha256"]:
            if file_sha256(root / row["filename"]) != row["sha256"]:
                raise RuntimeError(f"simulation image changed: {row['filename']}")
        return audit

    audit = verify_audit(AUDIT, SIM_MANIFEST)
    control_audit = verify_audit(CONTROL_AUDIT, CONTROL_MANIFEST)
    fit_audit = verify_audit(FIT_AUDIT, FIT_MANIFEST)
    if fit_audit["effective_fit_sha256"] != file_sha256(FIT_GRID):
        raise RuntimeError("effective fit changed after materialization")
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
    sim = validated_inference(SIM_MANIFEST, SIM_CSV, SIM_SUMMARY, 120)
    control = validated_inference(CONTROL_MANIFEST, CONTROL_CSV, CONTROL_SUMMARY, 120)
    fitted = validated_inference(FIT_MANIFEST, FIT_CSV, FIT_SUMMARY, 120)
    original_index = index(base["stylegan2_orig"])
    real_index = index(real["recap_mac"])
    sim_index = index(sim["virtual_mac"])
    control_index = index(control["virtual_mac_co_spatial"])
    fitted_index = index(fitted["virtual_mac_effective_fit"])
    ids = set(sim_index)
    if (
        len(ids) != 120
        or ids != set(control_index)
        or ids != set(fitted_index)
        or not ids.issubset(original_index)
        or not ids.issubset(real_index)
    ):
        raise RuntimeError("the 120 paired source IDs are not aligned")
    original = {key: original_index[key] for key in ids}
    recapture = {key: real_index[key] for key in ids}
    virtual = {key: sim_index[key] for key in ids}
    virtual_control = {key: control_index[key] for key in ids}
    virtual_fitted = {key: fitted_index[key] for key in ids}
    rng = np.random.default_rng(20260925)
    result = {
        "scope": "120 frozen Chimera development sources, Mac real recapture and one unfitted virtual process",
        "warning": "virtual settings are hypotheses, not MacBook/iPhone calibration; no reserved sources used",
        "simulation_manifest_sha256": audit["manifest_sha256"],
        "simulation_inference_sha256": file_sha256(SIM_CSV),
        "control_manifest_sha256": control_audit["manifest_sha256"],
        "control_inference_sha256": file_sha256(CONTROL_CSV),
        "effective_fit_manifest_sha256": fit_audit["manifest_sha256"],
        "effective_fit_inference_sha256": file_sha256(FIT_CSV),
        "effective_fit_grid_sha256": file_sha256(FIT_GRID),
        "original": condition_metrics(list(original.values())),
        "real_recap256": condition_metrics(list(recapture.values())),
        "virtual_recap256": condition_metrics(list(virtual.values())),
        "co_spatial_control256": condition_metrics(list(virtual_control.values())),
        "effective_fit256": condition_metrics(list(virtual_fitted.values())),
        "original_to_real": paired_metrics(original, recapture, rng),
        "original_to_virtual": paired_metrics(original, virtual, rng),
        "real_to_virtual": paired_metrics(recapture, virtual, rng),
        "transfer": transfer(original, recapture, virtual),
        "control_transfer": transfer(original, recapture, virtual_control),
        "effective_fit_transfer": transfer(original, recapture, virtual_fitted),
    }
    striped_score = np.array([float(virtual[key]["score"]) for key in sorted(ids)])
    control_score = np.array(
        [float(virtual_control[key]["score"]) for key in sorted(ids)]
    )
    result["stripe_ablation"] = {
        "score_difference_mean_striped_minus_control": float(
            (striped_score - control_score).mean()
        ),
        "score_difference_mae": float(np.abs(striped_score - control_score).mean()),
        "score_pearson": float(np.corrcoef(striped_score, control_score)[0, 1]),
        "prediction_flip_count": int(
            np.sum((striped_score > 0) != (control_score > 0))
        ),
    }
    real_features = {}
    virtual_features = {}
    control_features = {}
    fitted_features = {}
    original_features = {}
    for source in sorted(ids):
        original_features[source] = features(ROOT / "stylegan2_orig" / source)
        real_features[source] = features(CONTROL / "recap_mac" / source)
        virtual_features[source] = features(
            Path(audit["output_root"]) / "virtual_mac" / source
        )
        control_features[source] = features(
            Path(control_audit["output_root"]) / "virtual_mac_co_spatial" / source
        )
        fitted_features[source] = features(
            Path(fit_audit["output_root"]) / "virtual_mac_effective_fit" / source
        )
    feature_names = next(iter(original_features.values()))
    feature_summary = {}
    for name in feature_names:
        feature_summary[name] = {
            group: numeric_summary([table[source][name] for source in sorted(ids)])
            for group, table in (
                ("original", original_features),
                ("real", real_features),
                ("virtual", virtual_features),
                ("co_spatial_control", control_features),
                ("effective_fit", fitted_features),
            )
        }
        for group, table in (
            ("real", real_features),
            ("virtual", virtual_features),
            ("co_spatial_control", control_features),
            ("effective_fit", fitted_features),
        ):
            feature_summary[name][group + "_minus_original"] = numeric_summary(
                [
                    table[source][name] - original_features[source][name]
                    for source in sorted(ids)
                ]
            )
    result["published_rgb_observables"] = feature_summary
    result["by_scene"] = {}
    for scene in sorted({source.split("/")[0] for source in ids}):
        scene_ids = sorted(source for source in ids if source.split("/")[0] == scene)
        if len(scene_ids) != 40:
            raise RuntimeError("development scenes are not balanced 40 per scene")
        result["by_scene"][scene] = {}
        for name, table in (
            ("original", original),
            ("real", recapture),
            ("virtual", virtual),
            ("control", virtual_control),
            ("effective_fit", virtual_fitted),
        ):
            metrics = condition_metrics([table[source] for source in scene_ids])
            result["by_scene"][scene][name] = {
                "auc": metrics["auc"],
                "ba": metrics["zero_threshold_balanced_accuracy"],
            }
    result["score_shift_by_label"] = {}
    for label in ("REAL", "FAKE"):
        label_ids = sorted(
            source for source in ids if original[source]["label"] == label
        )
        result["score_shift_by_label"][label] = {
            name: float(
                np.mean(
                    [
                        float(table[source]["score"]) - float(original[source]["score"])
                        for source in label_ids
                    ]
                )
            )
            for name, table in (
                ("real", recapture),
                ("virtual", virtual),
                ("control", virtual_control),
                ("effective_fit", virtual_fitted),
            )
        }
        bootstrap_rng = np.random.default_rng(20260925 if label == "REAL" else 20260926)
        draws = {}
        for name, table in (
            ("real", recapture),
            ("virtual", virtual),
            ("control", virtual_control),
            ("effective_fit", virtual_fitted),
        ):
            shifts = np.array(
                [
                    float(table[source]["score"]) - float(original[source]["score"])
                    for source in label_ids
                ]
            )
            if len(shifts) != 60:
                raise RuntimeError("expected 60 development sources per label")
            draws[name] = shifts
        scene_indices = [
            np.array(
                [
                    index
                    for index, source in enumerate(label_ids)
                    if source.split("/")[0] == scene
                ]
            )
            for scene in sorted(result["by_scene"])
        ]
        if any(len(group) != 20 for group in scene_indices):
            raise RuntimeError("expected 20 development sources per scene and label")
        sampled = np.concatenate(
            [
                group[bootstrap_rng.integers(0, 20, size=(5000, 20))]
                for group in scene_indices
            ],
            axis=1,
        )
        ci = {}
        for name, values in draws.items():
            means = values[sampled].mean(axis=1)
            ci[name] = [
                float(np.quantile(means, 0.025)),
                float(np.quantile(means, 0.975)),
            ]
        for name in ("virtual", "control", "effective_fit"):
            differences = (draws["real"] - draws[name])[sampled].mean(axis=1)
            ci["real_minus_" + name] = [
                float(np.quantile(differences, 0.025)),
                float(np.quantile(differences, 0.975)),
            ]
        result["score_shift_by_label"][label]["stratified_bootstrap_ci95"] = ci
    OUTPUT.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "original",
                    "real_recap256",
                    "virtual_recap256",
                    "co_spatial_control256",
                    "effective_fit256",
                    "transfer",
                    "control_transfer",
                    "effective_fit_transfer",
                    "stripe_ablation",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
