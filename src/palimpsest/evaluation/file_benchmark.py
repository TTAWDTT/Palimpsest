"""Reusable single-image file benchmark with expected-score and digest checks."""

import random
import math

from palimpsest.detection.files import predict_file
from palimpsest.io.hashing import file_sha256
from .timing import percentile


def benchmark_files(detector, items, *, repeats=3, seed=20261005, score_tolerance=1e-12):
    """Items supply filename/path/sha256/expected_score, never labels to detector.

    Digest reads warm filesystem caches. Startup/warmup is caller-controlled.
    A changed input or score refuses the run, rather than entering its summary.
    """
    if (not items or repeats <= 0 or len({r["filename"] for r in items}) != len(items)
            or not math.isfinite(score_tolerance) or score_tolerance < 0
            or any(not math.isfinite(r["expected_score"]) for r in items)):
        raise ValueError("Invalid benchmark denominator")
    for item in items:
        if file_sha256(item["path"]) != item["sha256"]:
            raise ValueError("Benchmark image digest changed")
    rng, measurements = random.Random(seed), []
    for repeat in range(repeats):
        shuffled = list(items)
        rng.shuffle(shuffled)
        for item in shuffled:
            result = predict_file(detector, item["path"])
            if abs(result.prediction.score - item["expected_score"]) > score_tolerance:
                raise ValueError("Cached/live prediction mismatch")
            measurements.append({"filename": item["filename"], "repeat": repeat, "pixels": result.width * result.height,
                                 "decode_ms": result.decode_ms, "predict_ms": result.prediction.timing_ms["predict_ms"],
                                 "end_to_end_ms": result.end_to_end_ms})
    summaries = {}
    for name, selected in {"all": measurements, "at_most_2MP": [r for r in measurements if r["pixels"] <= 2_000_000],
                           "above_2MP": [r for r in measurements if r["pixels"] > 2_000_000]}.items():
        summaries[name] = {"images": len({r["filename"] for r in selected}), "measurements": len(selected)}
        if selected:
            summaries[name].update({key: {f"p{int(q * 100)}": percentile([r[key] for r in selected], q) for q in (0.5, 0.95)}
                                    for key in ("decode_ms", "predict_ms", "end_to_end_ms")})
    return {"summaries": summaries, "measurements": measurements,
            "scope": "file SHA pre-read; CPU single-image; startup/warmup excluded; no phone/full-pipeline claim"}
