"""Known-answer translated-window controls and bounded cached-pixel audit.

No neural inference, new source images, classifier fitting or parameter scan.
The non-generator carrier is a required counterexample, not a detection plant.
"""

import argparse
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from palimpsest.evaluation.cumulants import population_cumulant4
from palimpsest.evaluation.cyclic_image import window_phase_rgb_summary
from palimpsest.io.hashing import file_sha256
from palimpsest.io.provenance import load_committed_json
from palimpsest.paths import REPO_ROOT

PROTOCOL = REPO_ROOT / "configs/evaluation/window_phase_contrast.json"


def complex_value(row):
    return complex(row["real"], row["imag"])


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-8)


def reject_wrong(actual, wrong):
    try:
        assert_close(actual, wrong)
    except AssertionError:
        return True
    raise AssertionError("Deliberately wrong answer was accepted")


def debruijn_binary(order):
    """Deterministic binary cycle; coverage is verified independently below."""
    state, sequence = [0] * (2 * order), []

    def visit(t, period):
        if t > order:
            if order % period == 0:
                sequence.extend(state[1 : period + 1])
            return
        state[t] = state[t - period]
        visit(t + 1, period)
        for value in range(state[t - period] + 1, 2):
            state[t] = value
            visit(t + 1, t)

    visit(1, 1)
    return np.array(sequence) * 2 - 1


def nn_field():
    cycle = debruijn_binary(9)
    assert len(cycle) == 512
    words = [
        tuple(cycle[(8 * j + r + np.arange(9)) % 512])
        for r in range(8)
        for j in range(64)
    ]
    assert len(set(words)) == 512 and set(words) == set(product((-1, 1), repeat=9))
    fine = np.stack(
        [np.repeat(cycle[(np.arange(513) + r) % 512], 2)[:1025] for r in range(8)]
    )
    return np.pad(np.repeat(fine, 16, axis=0), 16, mode="edge").astype(float)


def coefficient_and_population_check():
    """Linear coefficient propagation and all512 signs; closed formulas grade both.

    Both finite routes use NumPy but neither invokes the image/window evaluator.
    The population route shares the cumulant primitive; formulas are the oracle.
    """
    z = np.array(list(product((-1, 1), repeat=9)), dtype=float)
    predicted = {"cross": 32 * np.sqrt(2) * 1j, "pure": (96 + 64 * np.sqrt(2)) * 1j}
    checks = []
    for name, frequencies in (("cross", (2, 4, 6, 12)), ("pure", (2, 14, 2, 14))):
        for delta in (0, 1):
            selection = np.eye(9)[(np.arange(16) + delta) // 2]
            dft = np.exp(-2j * np.pi * np.outer(frequencies, np.arange(16)) / 16)
            coefficients = dft @ selection
            coefficients[(1, 3), :] = coefficients[(1, 3), :].conj()
            propagated = -2 * np.prod(coefficients, axis=0).sum()
            outputs = z @ coefficients.T
            enumerated = population_cumulant4(outputs, np.full(512, 1 / 512))
            expected = predicted[name] * (1 if delta == 0 else -7 / 8)
            assert_close(propagated, expected)
            assert_close(enumerated, expected)
            checks.append(
                {
                    "name": name,
                    "delta": delta,
                    "coefficient_error": float(abs(propagated - expected)),
                    "population_error": float(abs(enumerated - expected)),
                }
            )
            if name == "pure" and delta == 0:
                # Undo prescribed conjugation: it measures auto rather than pure.
                wrong = outputs.copy()
                wrong[:, (1, 3)] = wrong[:, (1, 3)].conj()
                assert reject_wrong(
                    population_cumulant4(wrong, np.full(512, 1 / 512)), expected
                )
    return checks


def read_rgb(path, digest):
    if file_sha256(path) != digest:
        raise ValueError("Pixel file hash mismatch before decode")
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=float) * (2 / 255) - 1


def controls(output):
    root = output.with_suffix("")
    root.mkdir(parents=True, exist_ok=False)
    all_rows = {}
    polynomial = 2 * (np.indices((256, 256))[1] / 255) ** 2 - 1
    y, x = np.indices((256, 256))
    carrier = (0.5 + 0.5 * y / 255) * np.cos(2 * np.pi * x / 8)
    fields = {
        "NN_x": nn_field(),
        "NN_y": nn_field().T,
        "quadratic": polynomial,
        "scene_carrier": carrier,
        "zeros": np.zeros((256, 256)),
    }
    for name, field in fields.items():
        path = root / f"{name}.npy"
        np.save(path, np.repeat(field[:, :, None], 3, axis=2), allow_pickle=False)
        digest = file_sha256(path)
        begin = perf_counter()
        measured = window_phase_rgb_summary(read_rgb(path, digest))
        all_rows[name] = {
            "file": str(path),
            "sha256": digest,
            "seconds": perf_counter() - begin,
            "metrics": measured,
        }
    s = np.sqrt(2)
    expected_q = {"cross": 32 * s * 1j, "pure": (96 + 64 * s) * 1j}
    expected_s = {
        "cross": 30 * s / (16 * np.sqrt(143.5)),
        "pure": (90 + 60 * s) / (16 + 7.5 * s) ** 2,
    }
    for axis in ("x", "y"):
        measured = all_rows[f"NN_{axis}"]["metrics"]
        assert measured["patch_count"] == 512
        for name in ("cross", "pure"):
            row = measured["measurements"][f"{axis}_{name}"]
            assert_close(complex_value(row["q0"]), expected_q[name])
            assert_close(complex_value(row["q1"]), -7 / 8 * expected_q[name])
            assert_close(row["normalized_contrast"], expected_s[name])
            assert reject_wrong(complex_value(row["q1"]), -expected_q[name])
    quadratic = all_rows["quadratic"]["metrics"]["measurements"]
    for name in ("cross", "pure"):
        assert_close(quadratic[f"x_{name}"]["normalized_contrast"], 0)
        assert reject_wrong(quadratic[f"x_{name}"]["normalized_contrast"], 1)
        assert quadratic[f"y_{name}"]["normalized_contrast"] is None
    scene = all_rows["scene_carrier"]["metrics"]["measurements"]
    assert_close(
        complex_value(scene["x_pure"]["q1"]), -complex_value(scene["x_pure"]["q0"])
    )
    assert_close(scene["x_pure"]["normalized_contrast"], 394 / 325)
    assert reject_wrong(scene["x_pure"]["normalized_contrast"], 0)
    assert scene["x_cross"]["normalized_contrast"] is None
    for row in all_rows["zeros"]["metrics"]["measurements"].values():
        assert row["normalized_contrast"] is None and row["denominator"] is None
        assert complex_value(row["q0"]) == complex_value(row["q1"]) == 0
        assert row["normalized_contrast"] != 0
    # Reader clean twin and deliberate fingerprint mismatch use identical path.
    try:
        read_rgb(root / "zeros.npy", "0" * 64)
    except ValueError:
        hash_corruption_caught = True
    else:
        raise AssertionError("Corrupted pixel fingerprint accepted")
    return {
        "controls_passed": True,
        "case_count": len(fields),
        "finite_routes": coefficient_and_population_check(),
        "expected_normalized": expected_s,
        "hash_corruption_caught": hash_corruption_caught,
        "rows": all_rows,
    }


def cached_views(protocol):
    root = REPO_ROOT / protocol["cached_data"]["root"]
    receipts = {}
    for name, digest in protocol["cached_data"]["receipts"].items():
        path = root / name
        if file_sha256(path) != digest:
            raise ValueError("Cached receipt changed")
        receipts[name] = json.loads(path.read_text(encoding="utf-8"))
    views = []
    for family in ("stylegan2", "stylegan3"):
        images = receipts[f"{family}_complete.json"]["images"]
        assert [image["seed"] for image in images] == list(range(8))
        for image in images:
            assert [b["branch"] for b in image["branches"]] == [
                "baseline",
                "linearized",
            ]
            for branch in image["branches"]:
                identity = {
                    "family": family,
                    "seed": image["seed"],
                    "branch": branch["branch"],
                }
                views.extend(
                    [
                        {
                            **identity,
                            "encoding": "float",
                            "file": branch["float_file"],
                            "sha256": branch["float_file_sha256"],
                        },
                        {
                            **identity,
                            "encoding": "PNG",
                            "file": branch["png"]["file"],
                            "sha256": branch["png"]["sha256"],
                        },
                    ]
                )
    photo = receipts["photos.json"]
    # The original photo packet's structure is fixed, not inferred from filenames.
    assert len(photo["photos"]) == 2
    for index, image in enumerate(photo["photos"]):
        assert [v["variant"] for v in image["variants"]] == [
            "baseline",
            "NN",
            "bicubic",
        ]
        for view in image["variants"]:
            views.append(
                {
                    "family": "photo",
                    "source": image["source"],
                    "branch": view["variant"],
                    "encoding": "PNG",
                    "file": str(
                        root / "photos" / f"photo{index}_{view['variant']}.png"
                    ),
                    "sha256": view["sha256"],
                }
            )
    assert len(views) == 70 and len({r["file"] for r in views}) == 70
    return views


def main():
    if not __debug__:
        raise RuntimeError(
            "Scientific audit requires assertions enabled; do not use -O"
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("controls", "pilot", "cached"), required=True
    )
    parser.add_argument("--protocol-revision", required=True)
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Receipt overwrite refused")
    protocol, digest = load_committed_json(PROTOCOL, args.protocol_revision, REPO_ROOT)
    identity = {
        "protocol_revision": args.protocol_revision,
        "protocol_sha256": digest,
        "code_sha256": file_sha256(Path(__file__)),
        "metric_sha256": file_sha256(
            REPO_ROOT / "src/palimpsest/evaluation/cyclic_image.py"
        ),
        "cumulant_sha256": file_sha256(
            REPO_ROOT / "src/palimpsest/evaluation/cumulants.py"
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
    }
    if args.mode == "controls":
        result = controls(args.output)
    else:
        if args.controls is None:
            raise ValueError("Require passed analytic control receipt")
        control = json.loads(args.controls.read_text(encoding="utf-8"))
        assert control["controls_passed"] and control["case_count"] == 5
        for key in (
            "protocol_sha256",
            "code_sha256",
            "metric_sha256",
            "cumulant_sha256",
        ):
            assert control[key] == identity[key]
        views = cached_views(protocol)
        rows = []
        if args.mode == "pilot":
            selected, reused = views[:3], []
        else:
            if args.pilot is None:
                raise ValueError("Require same-instrument three-view timing pilot")
            pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
            for key in (
                "protocol_sha256",
                "code_sha256",
                "metric_sha256",
                "cumulant_sha256",
            ):
                assert pilot[key] == identity[key]
            assert [r["view"] for r in pilot["rows"]] == views[:3]
            selected, reused = views[3:], pilot["rows"]
        for view in selected:
            path = Path(view["file"])
            # Hash precedes pixel decoding and is included in this audit timer.
            begin = perf_counter()
            rgb = read_rgb(path, view["sha256"])
            assert rgb.shape == (256, 256, 3)
            measured = window_phase_rgb_summary(rgb)
            rows.append(
                {
                    "view": view,
                    "metrics": measured,
                    "hash_decode_metric_seconds": perf_counter() - begin,
                }
            )
        combined = reused + rows
        result = {
            "rows": combined,
            "view_count": len(combined),
            "reused_pilot_count": len(reused),
            "controls_receipt_sha256": file_sha256(args.controls),
            "scope": "Exposed development cache; no source classification or external validation",
        }
        if args.mode == "pilot":
            costs = [r["hash_decode_metric_seconds"] for r in rows][1:]
            result["remaining_projection"] = {
                "remaining_views": 67,
                "seconds": 67 * float(np.mean(costs)),
                "basis": "last two of three same-code mixed float/PNG views; constant-cost assumption, excludes process startup and receipt loading",
            }
        else:
            assert len(combined) == 70
            result["pilot_sha256"] = file_sha256(args.pilot)
            result["projection_used"] = pilot["remaining_projection"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({**identity, **result}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "output": str(args.output),
                "controls_passed": result.get("controls_passed"),
                "view_count": result.get("view_count"),
                "projection": result.get("remaining_projection"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
