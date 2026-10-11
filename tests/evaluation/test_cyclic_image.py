"""Single-field cyclic measurement and degeneracy controls."""

from itertools import product

import numpy as np
import pytest

from palimpsest.evaluation.cyclic_image import (
    cyclic_rgb_summary,
    window_phase_summary,
)
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


def test_translated_window_rejects_smooth_content_but_not_periodic_scene():
    y, x = np.indices((256, 256))
    quadratic = window_phase_summary(2 * (x / 255) ** 2 - 1)
    for name in ("cross", "pure"):
        assert quadratic["measurements"][f"x_{name}"][
            "normalized_contrast"
        ] == pytest.approx(0, abs=1e-9)
        assert quadratic["measurements"][f"y_{name}"]["normalized_contrast"] is None
    scene = window_phase_summary((0.5 + 0.5 * y / 255) * np.cos(2 * np.pi * x / 8))
    assert scene["measurements"]["x_pure"]["normalized_contrast"] == pytest.approx(
        394 / 325, abs=1e-9
    )
    assert scene["measurements"]["x_cross"]["normalized_contrast"] is None


@pytest.mark.parametrize("transpose", (False, True))
def test_real_nn_windows_include_finite_boundary_and_not_circular_shift(transpose):
    from experiments.origin_detection.non_neural.generator_mechanisms.audit_window_phase import (
        nn_field,
    )

    field = nn_field()
    result = window_phase_summary(field.T if transpose else field)
    assert result["patch_count"] == 512
    key = "y_pure" if transpose else "x_pure"
    row = result["measurements"][key]
    q0, q1 = (complex(row[k]["real"], row[k]["imag"]) for k in ("q0", "q1"))
    assert q0 == pytest.approx((96 + 64 * np.sqrt(2)) * 1j)
    assert q1 == pytest.approx(-7 / 8 * q0)
    assert abs(q1 + q0) > 1  # A patch rotation would instead force -q0.


def test_window_phase_zero_degeneracy_and_translated_support_bounds():
    result = window_phase_summary(np.zeros((256, 256)))
    assert all(
        r["normalized_contrast"] is None for r in result["measurements"].values()
    )
    with pytest.raises(ValueError, match="Translated patch support"):
        window_phase_summary(np.zeros((32, 32)), margin=0)


def test_optimized_execution_cannot_disable_audit_gates(tmp_path):
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-m",
            "experiments.origin_detection.non_neural.generator_mechanisms.audit_window_phase",
            "--mode",
            "controls",
            "--protocol-revision",
            "0" * 40,
            "--output",
            str(tmp_path / "forbidden.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "assertions enabled" in result.stderr
    assert not (tmp_path / "forbidden.json").exists()
