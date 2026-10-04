"""Latency summaries including image decoding and preprocessing when recorded."""

import math
import statistics


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_timing(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    result = {}
    for key in ("decode_preprocess_ms", "gpu_transfer_forward_ms", "end_to_end_ms"):
        values = [float(row[key]) for row in rows]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"Invalid timing: {key}")
        result[key] = {
            "p50": statistics.median(values),
            "p95": percentile(values, 0.95),
        }
    return result
