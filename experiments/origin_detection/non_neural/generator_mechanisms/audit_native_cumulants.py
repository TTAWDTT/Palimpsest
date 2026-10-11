"""Bounded native-generator/photography mechanism assay, not a classifier.

ExternalNVIDIA code stays outside this package. Only hash-bound official models
are loaded. Generation uses author referenceops; the pure metric uses NumPy.
"""

import argparse
from contextlib import contextmanager
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
from PIL import Image

from palimpsest.evaluation.cyclic_image import cyclic_rgb_summary
from palimpsest.io.hashing import file_sha256
from palimpsest.io.provenance import load_committed_json
from palimpsest.paths import DATA_ROOT, MODELS_ROOT, REPO_ROOT, WORKSPACE_ROOT


PROTOCOL = REPO_ROOT / "configs/evaluation/native_generator_cumulants.json"


def predicted_lrelu(x, bias=None, *, alpha=0.2, gain=np.sqrt(2), clamp=None):
    """NumPyfloat32 pointwise prediction; caller explicitly supplies bias."""
    u = np.asarray(x, dtype=np.float32)
    if bias is not None:
        u = u + np.asarray(bias, dtype=np.float32).reshape(1, -1, 1, 1)
    result = np.where(u >= 0, u, np.float32(alpha) * u) * np.float32(gain)
    return result if clamp is None else np.clip(result, -clamp, clamp)


def measured_pixels(rgb):
    return cyclic_rgb_summary(rgb, patch=16, margin=16)


def quantize(rgb):
    # Originalgen_images.py uses *127.5+128 anduint8 truncation, no rounding.
    return np.clip(rgb * 127.5 + 128, 0, 255).astype(np.uint8)


def png_measurement(path, rgb):
    pixels = quantize(rgb)
    Image.fromarray(pixels).save(path)
    started = perf_counter()
    with Image.open(path) as im:
        decoded = np.asarray(im.convert("RGB"))
    if not np.array_equal(decoded, pixels):
        raise ValueError("PNG roundtrip changed pixels")
    stats = measured_pixels(decoded.astype(float) * (2 / 255) - 1)
    measured_seconds = perf_counter() - started
    return {
        "file": str(path),
        "sha256": file_sha256(path),
        "metrics": stats,
        "decode_metric_seconds": measured_seconds,
    }


def controls(out):
    started = perf_counter()
    out.mkdir(parents=True, exist_ok=True)
    fixtures = []
    for phase_mixed, expected, count in ((False, 1, 256), (True, 0, 4096)):
        states = np.array(list(product((-1, 1), repeat=8)))
        vectors = np.repeat(states, 2, axis=1)
        if phase_mixed:
            vectors = np.stack(
                [np.roll(vectors, r, axis=1) for r in range(16)], axis=1
            ).reshape(-1, 16)
        grid = 64 if phase_mixed else 16
        patches = np.repeat(vectors[:, None, :], 16, axis=1)
        field = np.concatenate(
            [
                np.concatenate(patches[r * grid : (r + 1) * grid], axis=1)
                for r in range(grid)
            ],
            axis=0,
        )
        field = np.pad(field, 16, mode="edge")
        rgb = np.repeat(field[:, :, None], 3, axis=2)
        fixture = png_measurement(
            out / ("mixed.png" if phase_mixed else "coupled.png"), rgb
        )
        for key in ("x_cross", "x_pure"):
            row = fixture["metrics"]["measurements"][key][0]
            np.testing.assert_allclose(
                row["normalized_magnitude"], expected, rtol=0, atol=1e-10
            )
            assert row["patch_count"] == count
        fixture.update(expected=expected, count=count)
        fixtures.append(fixture)
    direct_zero = measured_pixels(np.zeros((256, 256, 3)))
    for group in direct_zero["measurements"].values():
        assert all(r["normalized_magnitude"] is None for r in group)
        assert all(r["cumulant"] == {"real": 0, "imag": 0} for r in group)
    u = np.array([-20, -1, 0, 1, 20], dtype=np.float32)
    np.testing.assert_allclose(
        predicted_lrelu(u, gain=2, clamp=8), [-8, -0.4, 0, 2, 8], rtol=0, atol=1e-7
    )
    np.testing.assert_array_equal(
        predicted_lrelu(u, alpha=1, gain=2, clamp=8), [-8, -2, 0, 2, 8]
    )
    caught = 0
    for actual, wrong in (
        (
            fixtures[0]["metrics"]["measurements"]["x_cross"][0][
                "normalized_magnitude"
            ],
            0,
        ),
        (predicted_lrelu(u), predicted_lrelu(u) + 0.1),
    ):
        try:
            np.testing.assert_allclose(actual, wrong, rtol=0, atol=1e-10)
        except AssertionError:
            caught += 1
    assert caught == 2
    return {
        "fixtures": fixtures,
        "zero": direct_zero,
        "corruptions_caught": caught,
        "elapsed_seconds": perf_counter() - started,
    }


def model_file(spec):
    path = MODELS_ROOT / "stylegan_native" / spec["file"]
    if path.stat().st_size != spec["bytes"] or file_sha256(path) != spec["sha256"]:
        raise ValueError("Model bytes/SHA mismatch; refusing beforepickleload")
    return path


def external_modules(spec):
    external = WORKSPACE_ROOT / "external/stylegan3"
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=external,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=external,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if revision != spec["source_commit"] or dirty:
        raise ValueError("Externalsource commit/tree mismatch")
    sys.path.insert(0, str(external))
    import legacy
    from torch_utils.ops import bias_act, filtered_lrelu, upfirdn2d

    return legacy, (bias_act, filtered_lrelu, upfirdn2d)


@contextmanager
def author_reference_ops(modules, state, spec):
    bias_act, filtered, upfirdn = modules
    originals = [
        (module, name, getattr(module, name))
        for module, name in (
            (bias_act, "bias_act"),
            (filtered, "filtered_lrelu"),
            (upfirdn, "upfirdn2d"),
        )
    ]
    for module, name, original in originals:

        def reference(*args, _original=original, **kwargs):
            kwargs["impl"] = "ref"
            return _original(*args, **kwargs)

        setattr(module, name, reference)
    bias_original = originals[0][2]

    def observed_bias(
        x, b=None, dim=1, act="linear", alpha=None, gain=None, clamp=None, impl="ref"
    ):
        active = state.get("active", False) and act == "lrelu"
        selected_alpha = 1.0 if active and state["linearized"] else alpha
        result = bias_original(
            x,
            b,
            dim=dim,
            act=act,
            alpha=selected_alpha,
            gain=gain,
            clamp=clamp,
            impl="ref",
        )
        if active:
            if dim != 1 or state.get("capture") is not None:
                raise ValueError("Unexpectedactivation axis orduplicate capture")
            raw = x.detach().cpu().numpy()
            bias = None if b is None else b.detach().cpu().numpy()
            a = float(selected_alpha if selected_alpha is not None else 0.2)
            g = float(gain if gain is not None else np.sqrt(2))
            channels = spec["intervention"]["checked_channels"]
            expected = predicted_lrelu(
                raw[:, :channels],
                None if bias is None else bias[:channels],
                alpha=a,
                gain=g,
                clamp=clamp,
            )
            observed = result[:, :channels].detach().cpu().numpy()
            np.testing.assert_allclose(
                observed,
                expected,
                rtol=spec["intervention"]["rtol"],
                atol=spec["intervention"]["atol"],
            )
            try:
                np.testing.assert_allclose(
                    observed + spec["intervention"]["corruption"],
                    expected,
                    rtol=spec["intervention"]["rtol"],
                    atol=spec["intervention"]["atol"],
                )
            except AssertionError:
                rejected = True
            else:
                raise ValueError("Plantedactivation corruptionescaped")
            state["capture"] = {
                "input_sha256": sha256(raw.tobytes()).hexdigest(),
                "bias_sha256": None
                if bias is None
                else sha256(bias.tobytes()).hexdigest(),
                "shape": list(raw.shape),
                "alpha": a,
                "gain": g,
                "clamp": clamp,
                "checked_channels": min(channels, observed.shape[1]),
                "max_prediction_error": float(abs(observed - expected).max()),
                "corruption_caught": rejected,
            }
        return result

    bias_act.bias_act = observed_bias
    try:
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


def generator_assay(spec, model_spec, seeds, out, metadata_only=False):
    import torch

    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    legacy, modules = external_modules(spec)
    path = model_file(model_spec)
    with path.open("rb") as stream:
        networks = legacy.load_network_pkl(stream)
    generator = networks["G_ema"].eval().requires_grad_(False)
    del networks
    if (
        generator.z_dim != 512
        or generator.c_dim != 0
        or generator.img_resolution != 256
    ):
        raise ValueError("Unregisteredgenerator dimensions")
    target_name = (
        model_spec["target"]
        if model_spec["family"] == "stylegan2"
        else generator.synthesis.layer_names[-2]
    )
    target = generator.synthesis.get_submodule(target_name)
    metadata = {
        "family": model_spec["family"],
        "target": target_name,
        "parameters": sum(p.numel() for p in generator.parameters()),
        "embedded_source_sha256": sha256(
            type(generator.synthesis)._orig_module_src.encode()
        ).hexdigest(),
    }
    if metadata_only:
        return {"metadata": metadata, "images": []}
    generator = generator.to("cuda")
    state = {"active": False, "linearized": False, "capture": None}
    hooks = [
        target.register_forward_pre_hook(lambda *_: state.update(active=True)),
        target.register_forward_hook(lambda *_: state.update(active=False)),
    ]
    target_outputs = []
    hooks.append(
        target.register_forward_hook(
            lambda _module, _args, output: target_outputs.append(
                output.detach().clone()
            )
        )
    )

    def version_signature():
        return [
            (name, t.data_ptr(), t._version)
            for name, t in list(generator.named_parameters())
            + list(generator.named_buffers())
        ]

    versions = version_signature()
    results = []
    out.mkdir(parents=True, exist_ok=True)
    try:
        with author_reference_ops(modules, state, spec), torch.inference_mode():
            for seed in seeds:
                started = perf_counter()
                z = torch.from_numpy(
                    np.random.RandomState(seed)
                    .randn(1, generator.z_dim)
                    .astype(np.float32)
                ).to("cuda")
                w = generator.mapping(z, None, truncation_psi=1)
                w_sha = sha256(w.cpu().numpy().tobytes()).hexdigest()
                branches = []
                arrays = []
                target_outputs.clear()
                for linearized in (False, True):
                    state.update(linearized=linearized, capture=None)
                    image = generator.synthesis(w, noise_mode="const", force_fp32=True)
                    rgb = image[0].permute(1, 2, 0).cpu().numpy()
                    if not np.isfinite(rgb).all() or state["capture"] is None:
                        raise ValueError("Nonfiniteoutput ormissingactivation capture")
                    branch = "linearized" if linearized else "baseline"
                    float_path = out / f"seed{seed}_{branch}.npy"
                    np.save(float_path, rgb)
                    encoded = png_measurement(out / f"seed{seed}_{branch}.png", rgb)
                    branches.append(
                        {
                            "branch": branch,
                            "capture": state["capture"],
                            "float_sha256": sha256(rgb.tobytes()).hexdigest(),
                            "float_file": str(float_path),
                            "float_file_sha256": file_sha256(float_path),
                            "float_range": [float(rgb.min()), float(rgb.max())],
                            "float_metrics": measured_pixels(rgb),
                            "png": encoded,
                        }
                    )
                    arrays.append(rgb)
                if (
                    branches[0]["capture"]["input_sha256"]
                    != branches[1]["capture"]["input_sha256"]
                    or branches[0]["capture"]["bias_sha256"]
                    != branches[1]["capture"]["bias_sha256"]
                ):
                    raise ValueError("Pairedactivation input/bias differed")
                if (
                    len(target_outputs) != 2
                    or version_signature() != versions
                    or sha256(w.cpu().numpy().tobytes()).hexdigest() != w_sha
                ):
                    raise ValueError(
                        "Targetoutputs/modelstate/style changed unexpectedly"
                    )
                delta = target_outputs[1] - target_outputs[0]
                zero = torch.zeros_like(delta)
                if model_spec["family"] == "stylegan2":
                    final = generator.synthesis.get_submodule("b256.torgb")
                    predicted_delta = final(delta, w[:, -1]) - final(zero, w[:, -1])
                else:
                    final = generator.synthesis.get_submodule(
                        generator.synthesis.layer_names[-1]
                    )
                    predicted_delta = (
                        final(delta, w[:, -1], force_fp32=True)
                        - final(zero, w[:, -1], force_fp32=True)
                    ) * generator.synthesis.output_scale
                predicted_delta = predicted_delta[0].permute(1, 2, 0).cpu().numpy()
                measured_delta = arrays[1] - arrays[0]
                np.testing.assert_allclose(
                    measured_delta,
                    predicted_delta,
                    rtol=spec["intervention"]["rtol"],
                    atol=spec["intervention"]["atol"],
                )
                try:
                    np.testing.assert_allclose(
                        measured_delta + 0.1,
                        predicted_delta,
                        rtol=spec["intervention"]["rtol"],
                        atol=spec["intervention"]["atol"],
                    )
                except AssertionError:
                    rgb_corruption_caught = True
                else:
                    raise ValueError("CorruptedRGBresponse predictionescaped")
                results.append(
                    {
                        "seed": seed,
                        "style_sha256": w_sha,
                        "branches": branches,
                        "rgb_mse": float(
                            np.mean((arrays[0].astype(float) - arrays[1]) ** 2)
                        ),
                        "elapsed_seconds": perf_counter() - started,
                        "rgb_delta_prediction_error": float(
                            abs(measured_delta - predicted_delta).max()
                        ),
                        "model_state_unchanged": True,
                        "rgb_corruption_caught": rgb_corruption_caught,
                    }
                )
                print(
                    json.dumps(
                        {
                            "family": model_spec["family"],
                            "seed": seed,
                            "seconds": results[-1]["elapsed_seconds"],
                        }
                    ),
                    flush=True,
                )
    finally:
        for hook in hooks:
            hook.remove()
    return {"metadata": metadata, "images": results}


def photography(spec, out):
    registry = REPO_ROOT / "configs/evaluation/decoder_chain_probe.json"
    if file_sha256(registry) != spec["registry_binding"]["decoder_chain_probe_sha256"]:
        raise ValueError("Photo registry differs fromregisteredbytes")
    old = json.loads(registry.read_text())
    rows = []
    out.mkdir(parents=True, exist_ok=True)
    for p in old["probes"]:
        if p["label"] != "REAL":
            continue
        path = DATA_ROOT / p["data_relative"]
        if p["role"] != "fit" or file_sha256(path) != p["sha256"]:
            raise ValueError("Photo role/SHA mismatch")
        with Image.open(path) as im:
            baseline = np.asarray(
                im.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
            )
        nearest = np.repeat(np.repeat(baseline[::2, ::2], 2, axis=0), 2, axis=1)
        bicubic = np.asarray(
            Image.fromarray(baseline)
            .resize((128, 128), Image.Resampling.BICUBIC)
            .resize((256, 256), Image.Resampling.BICUBIC)
        )
        variants = []
        for name, pixels in (
            ("baseline", baseline),
            ("NN", nearest),
            ("bicubic", bicubic),
        ):
            file = out / f"photo{len(rows)}_{name}.png"
            Image.fromarray(pixels).save(file)
            with Image.open(file) as im:
                decoded = np.asarray(im.convert("RGB"))
            assert np.array_equal(decoded, pixels)
            variants.append(
                {
                    "variant": name,
                    "sha256": file_sha256(file),
                    "metrics": measured_pixels(decoded.astype(float) * (2 / 255) - 1),
                }
            )
        rows.append(
            {
                "source": p["filename"],
                "source_sha256": p["sha256"],
                "variants": variants,
            }
        )
    assert len(rows) == 2
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-revision", required=True)
    parser.add_argument(
        "--mode",
        choices=("controls", "metadata", "pilot", "run", "photos"),
        required=True,
    )
    parser.add_argument("--family", choices=("stylegan2", "stylegan3"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec, digest = load_committed_json(PROTOCOL, args.protocol_revision, REPO_ROOT)
    artifact_dir = args.output.with_suffix("")
    if args.mode != "controls":
        controls_receipt = args.output.parent / "controls.json"
        gate = json.loads(controls_receipt.read_text())
        if gate["protocol_sha256"] != digest or gate["corruptions_caught"] != 2:
            raise ValueError("Currentprotocol controls missing ornotpassed")
    if args.mode == "controls":
        result = controls(artifact_dir)
    elif args.mode == "photos":
        result = {"photos": photography(spec, artifact_dir)}
    else:
        if not args.family:
            parser.error("Generator modes require --family")
        model_spec = next(m for m in spec["models"] if m["family"] == args.family)
        seeds = (
            spec["generation"]["pilot_seeds"]
            if args.mode == "pilot"
            else spec["generation"]["seeds"]
        )
        prior = None
        if args.mode == "run":
            pilot_path = args.output.parent / f"{args.family}_pilot.json"
            prior = json.loads(pilot_path.read_text())
            if prior["protocol_sha256"] != digest or prior[
                "code_sha256"
            ] != file_sha256(Path(__file__)):
                raise ValueError("Pilot protocol/code differed; no blindreuse")
            finished = [p["seed"] for p in prior["images"]]
            if finished != spec["generation"]["pilot_seeds"]:
                raise ValueError("Pilot seeds/coverage mismatch")
            for image in prior["images"]:
                for branch in image["branches"]:
                    if (
                        file_sha256(Path(branch["png"]["file"]))
                        != branch["png"]["sha256"]
                    ):
                        raise ValueError("Pilot PNG changed")
                    if (
                        file_sha256(Path(branch["float_file"]))
                        != branch["float_file_sha256"]
                    ):
                        raise ValueError("Pilot floatfile changed")
            seeds = [s for s in seeds if s not in finished]
        result = generator_assay(
            spec, model_spec, seeds, artifact_dir, metadata_only=args.mode == "metadata"
        )
        if prior is not None:
            result["images"] = prior["images"] + result["images"]
            if [r["seed"] for r in result["images"]] != spec["generation"]["seeds"]:
                raise ValueError("Final seed roster mismatch")
            result["reused_pilot_count"] = len(prior["images"])
    result.update(
        mode=args.mode,
        protocol_revision=args.protocol_revision,
        protocol_sha256=digest,
        code_sha256=file_sha256(Path(__file__)),
        scope=spec["scope"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "mode": args.mode}), flush=True)


if __name__ == "__main__":
    main()
