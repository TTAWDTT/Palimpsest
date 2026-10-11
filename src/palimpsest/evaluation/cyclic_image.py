"""Descriptive single-image cyclic cumulants, with no classifier or iid claim.

Fixed, aligned nonoverlapping patches form an empirical distribution. Their
dependence and content heterogeneity preclude interpreting a split as an iid CI.
"""

import numpy as np

from .cumulants import normalized_cumulant_magnitude, population_cumulant4


def cyclic_patch_summary(field, *, patch=16, margin=16) -> dict:
    """Measure period2 quartets in x/y, and four fixed spatial quadrants.

    Input is finite float luminance. No resize, window taper, variance whitening,
    clipping or label-dependent operation is performed here. Frequencies refer
    to the patch DFT; margins must be a multiple of the period under study.
    """
    data = np.asarray(field, dtype=np.float64)
    if (
        data.ndim != 2
        or not np.isfinite(data).all()
        or isinstance(patch, bool)
        or not isinstance(patch, int)
        or patch < 8
        or patch % 8
        or isinstance(margin, bool)
        or not isinstance(margin, int)
        or margin < 0
        or margin % 2
    ):
        raise ValueError("Require finite2D field, patch multiple8 and even margin")
    core = data[margin:-margin, margin:-margin] if margin else data
    if min(core.shape, default=0) < 2 * patch or any(n % patch for n in core.shape):
        raise ValueError("Interior must have at least2x2 complete patches")
    rows, columns = core.shape[0] // patch, core.shape[1] // patch
    patches = core.reshape(rows, patch, columns, patch).transpose(0, 2, 1, 3)
    spectra = np.fft.fft2(patches, axes=(-2, -1), norm="ortho")
    unit = patch // 8
    reports = {}
    for axis in ("x", "y"):
        for name, indices in (
            ("cross", (unit, 2 * unit, 3 * unit, 6 * unit)),
            ("pure", (unit, 7 * unit, unit, 7 * unit)),
        ):
            features = (
                spectra[:, :, 0, list(indices)]
                if axis == "x"
                else spectra[:, :, list(indices), 0]
            )
            summaries = []
            groups = {"all": features.reshape(-1, 4)}
            for r in (0, 1):
                for c in (0, 1):
                    section = features[
                        r * (rows // 2) : (r + 1) * (rows // 2) if r == 0 else rows,
                        c * (columns // 2) : (c + 1) * (columns // 2)
                        if c == 0
                        else columns,
                    ]
                    groups[f"quadrant_{r}{c}"] = section.reshape(-1, 4)
            for group, values in groups.items():
                weights = np.full(len(values), 1 / len(values))
                conjugated = values.copy()
                conjugated[:, (1, 3)] = conjugated[:, (1, 3)].conj()
                cumulant = population_cumulant4(conjugated, weights)
                marginal, strengths, powers = [], [], []
                for u in values.T:
                    auto = population_cumulant4(
                        np.column_stack((u, u.conj(), u, u.conj())), weights
                    )
                    centered = u - u.mean()
                    power = float(np.mean(abs(centered) ** 2))
                    marginal.append(auto)
                    powers.append(power)
                    strengths.append(
                        None if power == 0 else float(abs(auto) / power**2)
                    )
                # Relative numerical floor, explicitly not a statistical gate.
                floor = 1e-12 * max(powers, default=0) ** 2
                normalized = normalized_cumulant_magnitude(
                    cumulant, marginal, floor=floor
                )
                summaries.append(
                    {
                        "group": group,
                        "patch_count": len(values),
                        "cumulant": {"real": cumulant.real, "imag": cumulant.imag},
                        "normalized_magnitude": normalized,
                        "marginal_strengths": strengths,
                        "numerical_floor": floor,
                    }
                )
            reports[f"{axis}_{name}"] = summaries
    return {
        "patch": patch,
        "margin": margin,
        "patch_grid": [rows, columns],
        "measurements": reports,
        "scope": "Empirical patch cumulants and fixed quadrant sensitivity; no iid, CI, AI score or universal invariance claim",
    }


def cyclic_rgb_summary(rgb, *, patch=16, margin=16) -> dict:
    """RGB float in caller's units, converted by fixed encoded-RGB luma weights."""
    data = np.asarray(rgb, dtype=np.float64)
    if data.ndim != 3 or data.shape[2] != 3 or not np.isfinite(data).all():
        raise ValueError("Require finiteHxWx3 RGB")
    luma = data @ np.array([0.299, 0.587, 0.114])
    return cyclic_patch_summary(luma, patch=patch, margin=margin)
