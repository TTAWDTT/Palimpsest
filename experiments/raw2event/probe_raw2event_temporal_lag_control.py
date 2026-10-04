"""Test whether flat-region temporal residuals grow with frame lag.

Independent sensor read/shot noise alone predicts similar second-difference
scale at lag 1 and lag 5. A growth implicates movement/display/other temporal
variation and prevents treating the lag-1 residual as pure sensor noise.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from experiments.raw2event.probe_raw2event_temporal_residual import (
    ROOT,
    PREFIXES,
    INTENSITY_EDGES,
    MAX_ABS_SECOND_DIFF,
    frames,
)


OUT = Path("work/raw2event_temporal_lag_control.json")


def main() -> None:
    records = []
    for prefix in PREFIXES:
        data = list(frames(ROOT / f"{prefix}.mkv"))
        hists = {
            lag: np.zeros(
                (4, len(INTENSITY_EDGES) - 1, MAX_ABS_SECOND_DIFF + 1), dtype=np.int64
            )
            for lag in (1, 5)
        }
        centers = list(range(5, len(data) - 5, 5))
        for index in centers:
            current = data[index]
            for parity, (y, x) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                b = current[y::2, x::2].astype(np.float32)
                local_mean = cv2.GaussianBlur(b, (5, 5), 0)
                local_mean_sq = cv2.GaussianBlur(b * b, (5, 5), 0)
                local_std = np.sqrt(np.maximum(0, local_mean_sq - local_mean**2))
                eligible = local_std < 4
                eligible[:4] = eligible[-4:] = False
                eligible[:, :4] = eligible[:, -4:] = False
                intensity = np.digitize(b, INTENSITY_EDGES) - 1
                for lag in (1, 5):
                    a = data[index - lag][y::2, x::2].astype(np.float32)
                    c = data[index + lag][y::2, x::2].astype(np.float32)
                    second_abs = np.abs(a - 2 * b + c).astype(np.int32)
                    for band in range(len(INTENSITY_EDGES) - 1):
                        mask = eligible & (intensity == band)
                        if np.any(mask):
                            hists[lag][parity, band] += np.bincount(
                                np.clip(second_abs[mask], 0, MAX_ABS_SECOND_DIFF),
                                minlength=MAX_ABS_SECOND_DIFF + 1,
                            )
        record = {
            "prefix": prefix,
            "frames_decoded": len(data),
            "centers": len(centers),
            "comparison": [],
        }
        for parity in range(4):
            for band in range(len(INTENSITY_EDGES) - 1):
                row = {
                    "cfa_parity_yx": [parity // 2, parity % 2],
                    "intensity_interval": [
                        INTENSITY_EDGES[band],
                        INTENSITY_EDGES[band + 1],
                    ],
                }
                for lag in (1, 5):
                    hist = hists[lag][parity, band]
                    n = int(hist.sum())
                    row[f"lag_{lag}_n"] = n
                    row[f"lag_{lag}_median_abs_second_difference_counts"] = (
                        int(np.searchsorted(np.cumsum(hist), (n + 1) // 2))
                        if n
                        else None
                    )
                record["comparison"].append(row)
        records.append(record)
        print(prefix, len(centers), flush=True)
    report = {
        "method": "same center frames and same locally flat CFA masks; second difference at lag 1 and lag 5",
        "flatness_threshold_counts": 4,
        "qualification": "lag growth can implicate motion or display/illumination dynamics, but equal scales do not prove pure sensor noise",
        "records": records,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
