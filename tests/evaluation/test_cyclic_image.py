"""Single-field cyclic measurement and degeneracy controls."""

from itertools import product

import numpy as np
import pytest

from palimpsest.evaluation.cyclic_image import cyclic_rgb_summary
from experiments.origin_detection.non_neural.generator_mechanisms import (
    audit_native_cumulants as instrument,
)


def sign_field(random_phases=False):
    rows = []
    for signs in product((-1, 1), repeat=8):
        row = np.repeat(signs, 2)
        rows.extend([np.roll(row, s) for s in range(16)] if random_phases else [row])
    side = 64 if random_phases else 16
    patches = np.broadcast_to(np.array(rows)[:, None, :], (len(rows), 16, 16))
    field = (
        patches.reshape(side, side, 16, 16)
        .transpose(0, 2, 1, 3)
        .reshape(side * 16, side * 16)
    )
    field = np.pad(field, 16, mode="edge")
    return np.repeat(field[:, :, None], 3, axis=2).astype(float)


@pytest.mark.parametrize(
    "random_phases,expected,count", ((False, 1, 256), (True, 0, 4096))
)
def test_known_phase_relation_and_unaligned_null(random_phases, expected, count):
    result = cyclic_rgb_summary(sign_field(random_phases))
    for name in ("x_cross", "x_pure"):
        row = result["measurements"][name][0]
        assert row["patch_count"] == count
        assert row["normalized_magnitude"] == pytest.approx(expected, abs=1e-10)


def test_zero_field_stays_undefined():
    result = cyclic_rgb_summary(np.zeros((256, 256, 3)))
    for group in result["measurements"].values():
        assert all(row["normalized_magnitude"] is None for row in group)
        assert all(row["cumulant"] == {"real": 0, "imag": 0} for row in group)


@pytest.mark.parametrize("shape", ((16, 16, 3), (257, 256, 3), (256, 256, 4)))
def test_bad_frame_shape_refused(shape):
    with pytest.raises(ValueError):
        cyclic_rgb_summary(np.zeros(shape))


def test_model_hash_guard_clean_and_broken_twins(tmp_path, monkeypatch):
    monkeypatch.setattr(instrument, "MODELS_ROOT", tmp_path)
    root = tmp_path / "stylegan_native"
    root.mkdir()
    (root / "fixture.pkl").write_bytes(b"abc")
    spec = {
        "file": "fixture.pkl",
        "bytes": 3,
        "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    }
    assert instrument.model_file(spec) == root / "fixture.pkl"
    with pytest.raises(ValueError, match="refusing beforepickleload"):
        instrument.model_file({**spec, "sha256": "0" * 64})


def test_pointwise_prediction_signed_and_clipped():
    u = np.array([-20, -1, 0, 1, 20], dtype=np.float32)
    np.testing.assert_allclose(
        instrument.predicted_lrelu(u, gain=2, clamp=8),
        [-8, -0.4, 0, 2, 8],
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        instrument.predicted_lrelu(u, alpha=1, gain=2, clamp=8), [-8, -2, 0, 2, 8]
    )
