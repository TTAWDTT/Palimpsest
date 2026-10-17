"""Descriptive single-image cyclic cumulants, with no classifier or iid claim.

Fixed, aligned nonoverlapping patches form an empirical distribution. Their
dependence and content heterogeneity preclude interpreting a split as an iid CI.
"""

import numpy as np

from .cumulants import normalized_cumulant_magnitude, population_cumulant4


def _patch_spectra(field, patch, margin, *, offset=(0, 0), complete=True, dc=False):
    """Common patch extraction; translated support is never a patch rotation."""
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
    interior = tuple(n - 2 * margin for n in data.shape)
    if min(interior) < 2 * patch or (complete and any(n % patch for n in interior)):
        raise ValueError("Interior must have at least2x2 complete patches")
    rows, columns = (n // patch for n in interior)
    starts = (margin + offset[0], margin + offset[1])
    stops = (starts[0] + rows * patch, starts[1] + columns * patch)
    if min(starts) < 0 or any(b > n for b, n in zip(stops, data.shape)):
        raise ValueError("Translated patch support exceeds image")
    core = data[starts[0] : stops[0], starts[1] : stops[1]]
    patches = core.reshape(rows, patch, columns, patch).transpose(0, 2, 1, 3)
    if dc:
        patches = patches - patches.mean(axis=(-2, -1), keepdims=True)
    spectra = np.fft.fft2(patches, axes=(-2, -1), norm="ortho")
    return data, spectra, rows, columns


def _quartets(patch):
    unit = patch // 8
    return (
        ("cross", (unit, 2 * unit, 3 * unit, 6 * unit)),
        ("pure", (unit, 7 * unit, unit, 7 * unit)),
    )


def _frequency_columns(spectra, axis, indices):
    return (
        spectra[:, :, 0, list(indices)]
        if axis == "x"
        else spectra[:, :, list(indices), 0]
    )


def cyclic_patch_summary(field, *, patch=16, margin=16) -> dict:
    """Measure period2 quartets in x/y, and four fixed spatial quadrants.

    Input is finite float luminance. No resize, window taper, variance whitening,
    clipping or label-dependent operation is performed here. Frequencies refer
    to the patch DFT; margins must be a multiple of the period under study.
    """
    _, spectra, rows, columns = _patch_spectra(field, patch, margin)
    reports = {}
    for axis in ("x", "y"):
        for name, indices in _quartets(patch):
            features = _frequency_columns(spectra, axis, indices)
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


def _luma(rgb):
    data = np.asarray(rgb, dtype=np.float64)
    if data.ndim != 3 or data.shape[2] != 3 or not np.isfinite(data).all():
        raise ValueError("Require finiteHxWx3 RGB")
    return data @ np.array([0.299, 0.587, 0.114])


def cyclic_rgb_summary(rgb, *, patch=16, margin=16) -> dict:
    """RGB float in caller's units, converted by fixed encoded-RGB luma weights."""
    return cyclic_patch_summary(_luma(rgb), patch=patch, margin=margin)


def window_phase_summary(field, *, patch=16, margin=16) -> dict:
    """Contrast actual window starts separated by one pixel along each axis.

    Averaged powers replace the potentially zero auto-cumulant normalizer.
    No iid, AI specificity, Gaussian-noise, JPEG or scale invariance follows.
    Unused partial interior edges are floored and reported, never resized.
    """
    data, first, rows, columns = _patch_spectra(
        field, patch, margin, complete=False, dc=True
    )
    reports = {}
    for axis, offset in (("x", (0, 1)), ("y", (1, 0))):
        _, second, _, _ = _patch_spectra(
            data, patch, margin, offset=offset, complete=False, dc=True
        )
        for name, indices in _quartets(patch):
            cumulants, powers = [], []
            for spectra in (first, second):
                values = _frequency_columns(spectra, axis, indices).reshape(-1, 4)
                centered = values - values.mean(axis=0)
                powers.append(np.mean(abs(centered) ** 2, axis=0))
                values = values.copy()
                values[:, (1, 3)] = values[:, (1, 3)].conj()
                cumulants.append(
                    population_cumulant4(values, np.full(len(values), 1 / len(values)))
                )
            averaged = np.mean(powers, axis=0)
            if not np.isfinite(averaged).all():
                raise ValueError("Nonfinite Fourier powers")
            floor = max(
                (1e-12 * patch * np.max(abs(data))) ** 2, 1e-12 * float(max(averaged))
            )
            denominator = (
                None
                if (averaged <= floor).any()
                else float(np.exp(0.5 * np.log(averaged).sum()))
            )
            contrast = (cumulants[0] - cumulants[1]) / 2
            common = (cumulants[0] + cumulants[1]) / 2
            reports[f"{axis}_{name}"] = {
                "q0": {"real": cumulants[0].real, "imag": cumulants[0].imag},
                "q1": {"real": cumulants[1].real, "imag": cumulants[1].imag},
                "contrast": {"real": contrast.real, "imag": contrast.imag},
                "common": {"real": common.real, "imag": common.imag},
                "powers0": powers[0].tolist(),
                "powers1": powers[1].tolist(),
                "power_floor": float(floor),
                "denominator": denominator,
                "normalized_contrast": (
                    None if denominator is None else float(abs(contrast) / denominator)
                ),
            }
    return {
        "patch": patch,
        "margin": margin,
        "patch_grid": [rows, columns],
        "patch_count": rows * columns,
        "unused_interior": [
            data.shape[0] - 2 * margin - rows * patch,
            data.shape[1] - 2 * margin - columns * patch,
        ],
        "measurements": reports,
        "scope": "Finite translated-window contrast; not AI probability or propagation invariant",
    }


def window_phase_rgb_summary(rgb, *, patch=16, margin=16) -> dict:
    """Same encoded-RGB luma conversion as the historical cyclic descriptor."""
    return window_phase_summary(_luma(rgb), patch=patch, margin=margin)
