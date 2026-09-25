"""One-pass source-disjoint confirmation of the frozen Raw2Event CFA hypotheses.

The ten new prefixes were selected and written before any new RAW/RGB file was
opened. All count mappings come from earlier ten-source calibration only.
"""

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

from work.audit_raw2event_probe import ROOT, detect_tag, extract_frame
from work.audit_raw2event_split_first_frames import SOURCE_QUAD, REFERENCE_AUDIT, load_originals
from work.evaluate_raw2event_spectral_mix import cached_bands, channel_mask, code_hash
from work.match_raw2event_cifar_source import normalized_gray
from work.probe_raw2event_cfa_phase import mask_for
from work.probe_raw2event_source_to_raw import prepare
from work.refine_raw2event_content_geometry import BOUNDS, adjusted_corners, score
from work.verify_raw2event_split_content import ARCHIVE, OFFICIAL_MD5, load_candidates
from work.fetch_cifar10_python import md5


MANIFEST = Path("E:/ai_image_origin_research/data/manifests/raw2event_phase_confirmation_v1.csv")
EXPECTED_SHA = "c699c6b9ce8ce779f892c5caaab11e19b4c24e0e697ec8116b94115c0175e372"
OLDER_SPLIT = Path("E:/ai_image_origin_research/data/manifests/raw2event_process_split_v1.csv")
DOWNLOADS = Path("work/raw2event_phase_confirmation_downloads")
PHASE_WEIGHTS = Path("work/raw2event_cfa_phase_probe.json")
MIX_WEIGHTS = Path("work/raw2event_spectral_mix_v1.json")
OUT = Path("work/raw2event_phase_confirmation_evaluation.json")
METHODS = ("vertical_rgb", "co_spatial_rgb_control", "simple_rgb_sample_control")
PHASES = ("RGGB", "BGGR", "GRBG", "GBRG")
DEST = np.asarray([[0, 0], [31, 0], [31, 31], [0, 31]], dtype=np.float32)
LABELS = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def content_geometry(rgb: np.ndarray, source: np.ndarray, reference_tag: np.ndarray) -> dict:
    tag, tag_id = detect_tag(rgb)
    if tag_id != 0:
        raise RuntimeError("expected AprilTag 0")
    similarity, _ = cv2.estimateAffinePartial2D(reference_tag, tag, method=cv2.LMEDS)
    if similarity is None:
        raise RuntimeError("Tag similarity estimation failed")
    initial = cv2.transform(np.asarray(SOURCE_QUAD, dtype=np.float32)[None], similarity)[0]
    feature = normalized_gray(source[None])[0]
    before = score(rgb, feature, initial)
    result = minimize(lambda values: -score(rgb, feature, adjusted_corners(initial, values)),
                      [0, 0, 1, 0], method="Powell", bounds=BOUNDS,
                      options={"xtol": 1e-3, "ftol": 1e-5, "maxiter": 80})
    refined = adjusted_corners(initial, result.x)
    after = score(rgb, feature, refined)
    if after < before - 1e-4:
        raise RuntimeError("known-source RGB refinement reduced correlation")
    return {"tag_id": tag_id, "tag_similarity_corners": initial.tolist(),
            "tag_similarity_zncc": before, "source_rgb_refined_corners": refined.tolist(),
            "source_rgb_refined_zncc": after,
            "source_rgb_adjustment_dx_dy_scale_radians": result.x.tolist()}


def identity_rank(rgb: np.ndarray, corners: np.ndarray, row: dict, candidates: dict) -> dict:
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32), DEST)
    query = cv2.warpPerspective(rgb, transform, (32, 32))
    features, keys = candidates[LABELS.index(row["class_name"])]
    scores = features @ normalized_gray(query[None])[0]
    target = keys.index((row["cifar_batch"], int(row["row"])))
    return {"named_source_rank_among_6000": int((scores > scores[target]).sum()) + 1,
            "named_source_zncc": float(scores[target])}


def predict_phase(bands: np.ndarray, phase: str, weights: np.ndarray) -> np.ndarray:
    estimate = np.empty(bands.shape[:2], dtype=np.float32)
    for band in range(3):
        select = mask_for(estimate.shape, phase, band)
        estimate[select] = weights[band, 0] + weights[band, 1] * bands[..., band][select]
    return estimate


def predict_full(bands: np.ndarray, weights: np.ndarray) -> np.ndarray:
    estimate = np.empty(bands.shape[:2], dtype=np.float32)
    for channel in range(3):
        select = channel_mask(estimate.shape, channel)
        estimate[select] = weights[channel, 0] + bands[select] @ weights[channel, 1:]
    return estimate


def metrics(estimate: np.ndarray, prepared: dict) -> dict:
    truth = prepared["actual"][prepared["mask"]]
    predicted = estimate[prepared["mask"]]
    return {"n_pixels": len(truth), "mae_counts": float(np.abs(predicted - truth).mean()),
            "pearson": float(np.corrcoef(predicted, truth)[0, 1]),
            "actual_mean": float(truth.mean()), "predicted_mean": float(predicted.mean())}


def main() -> None:
    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != EXPECTED_SHA:
        raise RuntimeError("frozen confirmation manifest changed")
    if ARCHIVE.stat().st_size != 170_498_071 or md5(ARCHIVE) != OFFICIAL_MD5:
        raise RuntimeError("official CIFAR archive verification failed")
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8", newline="")))
    if len(rows) != 10 or len({row["class_name"] for row in rows}) != 10:
        raise RuntimeError("expected ten frozen distinct classes")
    older = {row["prefix"] for row in csv.DictReader(OLDER_SPLIT.open(encoding="utf-8", newline=""))}
    if len(older) != 35 or older.intersection(row["prefix"] for row in rows):
        raise RuntimeError("confirmation overlaps prior frozen sources")
    for row in rows:
        audit = json.loads((DOWNLOADS / f"{row['prefix']}.json").read_text(encoding="utf-8"))
        if audit["prefix"] != row["prefix"] or not audit["complete_verified"] or len(audit["files"]) != 3:
            raise RuntimeError(f"unverified official triple: {row['prefix']}")
    reference = json.loads(REFERENCE_AUDIT.read_text(encoding="utf-8"))
    reference_tag = np.asarray(reference["samples"]["0"]["tag"]["rgb_corners"], dtype=np.float32)
    sources = load_originals(rows)
    candidates = load_candidates()
    phase_fits = json.loads(PHASE_WEIGHTS.read_text(encoding="utf-8"))["conditions"]
    mix_fits = json.loads(MIX_WEIGHTS.read_text(encoding="utf-8"))["conditions"]
    fingerprint = code_hash()
    output = []
    for number, row in enumerate(rows, start=1):
        prefix = row["prefix"]
        rgb = extract_frame(ROOT / "frames_rgb" / f"{prefix}.mkv", 0, "rgb24", 3, "u1")
        geom = content_geometry(rgb, sources[prefix], reference_tag)
        geom["identity_without_source_refinement"] = identity_rank(
            rgb, np.asarray(geom["tag_similarity_corners"], dtype=np.float32), row, candidates)
        prepared = prepare(prefix, None, sources[prefix],
                           np.asarray(geom["source_rgb_refined_corners"], dtype=np.float32))
        results = {}
        for method in METHODS:
            bands, seconds = cached_bands(row, prepared, method, fingerprint)
            for phase in PHASES:
                weights = np.asarray(phase_fits[method][phase]["weights_black_and_gain_per_color"], dtype=np.float64)
                results[f"{method}__{phase}_diagonal"] = metrics(predict_phase(bands, phase, weights), prepared)
            full = np.asarray(mix_fits[f"{method}__nonnegative_full"]["weights_intercept_and_display_primary_coupling"],
                              dtype=np.float64)
            results[f"{method}__nonnegative_full"] = metrics(predict_full(bands, full), prepared)
            results[f"{method}__render_seconds"] = seconds
        output.append({"prefix": prefix, "class_name": row["class_name"], "capture_day": row["capture_day"],
                       "geometry": geom, "conditions": results})
        print(f"evaluated {number}/10 {row['class_name']} {prefix}", flush=True)
    condition_names = [name for name in output[0]["conditions"] if not name.endswith("render_seconds")]
    aggregate = {name: {"mean_source_mae_counts": float(np.mean([r["conditions"][name]["mae_counts"] for r in output])),
                        "median_source_mae_counts": float(np.median([r["conditions"][name]["mae_counts"] for r in output])),
                        "mean_source_pearson": float(np.mean([r["conditions"][name]["pearson"] for r in output]))}
                 for name in condition_names}
    report = {"manifest_sha256": EXPECTED_SHA, "n_new_sources": len(output),
              "named_identity_top1_under_tag_similarity": sum(r["geometry"]["identity_without_source_refinement"]["named_source_rank_among_6000"] == 1 for r in output),
              "named_identity_top10_under_tag_similarity": sum(r["geometry"]["identity_without_source_refinement"]["named_source_rank_among_6000"] <= 10 for r in output),
              "source_rgb_refined_zncc_min_median": [float(min(r["geometry"]["source_rgb_refined_zncc"] for r in output)),
                                                      float(np.median([r["geometry"]["source_rgb_refined_zncc"] for r in output]))],
              "phase_weights_json_sha256": hashlib.sha256(PHASE_WEIGHTS.read_bytes()).hexdigest(),
              "mix_weights_json_sha256": hashlib.sha256(MIX_WEIGHTS.read_bytes()).hexdigest(),
              "radiance_code_fingerprint": fingerprint,
              "weights_fitted_only_on_prior_ten_calibration_sources": True,
              "phase_and_geometry_algorithm_fixed_without_inspecting_new_raw": True,
              "same_rig_and_class_date_confounds_remain": True,
              "earlier_reserved_untouched": True,
              "aggregate": aggregate, "per_source": output}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"identity_top1": report["named_identity_top1_under_tag_similarity"],
                      "aggregate": {k: v for k, v in aggregate.items() if k.startswith("vertical_rgb")}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
