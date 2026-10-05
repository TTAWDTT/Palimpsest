"""Descriptive quantiles shared by paired-image diagnostics."""

import numpy as np


def quantile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def quantiles(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        **(
            {
                key: float(np.quantile(array, probability))
                for key, probability in (
                    ("p05", 0.05),
                    ("p25", 0.25),
                    ("p50", 0.5),
                    ("p75", 0.75),
                    ("p95", 0.95),
                )
            }
            if len(array)
            else {}
        ),
    }
