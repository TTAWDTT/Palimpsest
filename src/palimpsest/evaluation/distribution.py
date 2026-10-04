"""Descriptive quantiles shared by paired-image diagnostics."""

import numpy as np


def quantile_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values)
    return {
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }
