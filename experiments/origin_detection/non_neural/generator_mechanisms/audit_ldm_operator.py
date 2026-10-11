"""One-shot pinned LDM Upsample assay with hand-set kernels, no trained model.

This is an instrument experiment, not a neural detector. Only the inspected
Upsample class is executed; the full decoder and external imports are not run.
"""

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from palimpsest.evaluation.polyphase import periodic_upsampling_matrix


PINNED_SOURCE_SHA = "043ff43c4d4304f7259419d1bd4529445da71661b5362653ecd174bd4342c413"
SOURCE_COMMIT = "a506df5756472e2ebaf9078affdde2c4f1502cd4"
WEIGHTS_SHA = "735e4c3a447a3255760d7f86845f09f937809baa529c17370d83e4c3758f3c75"
WEIGHTS_BYTES = 334641190


def _pinned_class(source_path):
    data = source_path.read_bytes()
    if sha256(data).hexdigest() != PINNED_SOURCE_SHA:
        raise ValueError("Source SHA does not match inspected LDM revision")
    classes = [
        node
        for node in ast.parse(data).body
        if isinstance(node, ast.ClassDef) and node.name == "Upsample"
    ]
    if len(classes) != 1:
        raise ValueError("Expected exactly one pinned Upsample class")
    return ast.Module(body=classes, type_ignores=[])


def _periodic_operator(length, kernel):
    repeat = periodic_upsampling_matrix([1, 1], 2, length)
    operator = np.kron(repeat, repeat).reshape(2 * length, 2 * length, length * length)
    return sum(
        (
            kernel[r, c] * np.roll(operator, (1 - r, 1 - c), axis=(0, 1))
            for r in range(3)
            for c in range(3)
        ),
        np.zeros_like(operator),
    )


def assay(source_path, length):
    if length not in (3, 4):
        raise ValueError("Registered fixture length must be3 or4")
    class_ast = _pinned_class(source_path)
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(0)
    namespace = {"torch": torch, "nn": torch.nn}
    exec(compile(class_ast, str(source_path), "exec"), namespace)
    identity = np.zeros((3, 3))
    identity[1, 1] = 1
    box = np.zeros((3, 3))
    box[1:, 1:] = 0.25
    kernels = {"identity": identity, "forward_box_mean": box}
    basis = torch.eye(length * length, dtype=torch.float64).reshape(
        -1, 1, length, length
    )
    results = []
    start = perf_counter()
    with torch.inference_mode():
        for name, kernel in kernels.items():
            module = namespace["Upsample"](1, True).double().eval()
            module.conv.weight.copy_(torch.from_numpy(kernel).reshape(1, 1, 3, 3))
            module.conv.bias.zero_()
            observed = module(basis).numpy()[:, 0].transpose(1, 2, 0)
            predicted = _periodic_operator(length, kernel)
            np.testing.assert_allclose(
                observed[1:-1, 1:-1], predicted[1:-1, 1:-1], rtol=0, atol=0
            )
            ones = module(
                torch.ones((1, 1, length, length), dtype=torch.float64)
            ).numpy()[0, 0]
            np.testing.assert_array_equal(
                ones[1:-1, 1:-1], np.ones((2 * length - 2, 2 * length - 2))
            )
            boundary_difference = float(np.max(np.abs(ones - predicted.sum(axis=2))))
            expected_boundary = 0.0 if name == "identity" else 0.75
            if boundary_difference != expected_boundary:
                raise ValueError("Known zero-padding boundary effect changed")
            # Population covariance for iid unit-variance inputs; no sampled images.
            variance = np.sum(observed[1:-1, 1:-1] ** 2, axis=2)
            phase_variances = [
                [float(variance[r::2, c::2].mean()) for c in range(2)] for r in range(2)
            ]
            # Cropping begins at phase(1,1), so absolute-phase table is reversed.
            phase_variances = np.roll(
                np.array(phase_variances), (1, 1), axis=(0, 1)
            ).tolist()
            expected_variances = (
                [[1, 1], [1, 1]] if name == "identity" else [[1, 0.5], [0.5, 0.25]]
            )
            np.testing.assert_array_equal(phase_variances, expected_variances)
            wrong = predicted.copy()
            wrong[1, 1, 0] += 1
            try:
                np.testing.assert_array_equal(observed[1:-1, 1:-1], wrong[1:-1, 1:-1])
            except AssertionError:
                caught = True
            else:
                raise ValueError("Planted interior coefficient error survived")
            results.append(
                {
                    "kernel": name,
                    "absolute_phase_variances": phase_variances,
                    "interior_max_abs_error": 0.0,
                    "ones_boundary_max_abs_error": boundary_difference,
                    "planted_coefficient_error_caught": caught,
                }
            )
    return {
        "source_commit": SOURCE_COMMIT,
        "source_sha256": PINNED_SOURCE_SHA,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "latent_side": length,
        "impulses_per_kernel": length * length,
        "fixtures": results,
        "elapsed_seconds": perf_counter() - start,
        "scope": "Executed pinned Upsample class with manually assigned1-channel kernels, no released weights, complete decoder, real photo, classifier or physical simulation",
    }


def trained_footprint(weight, row_phase, column_phase, output_channels):
    """Fold3x3 correlation taps through nearest-neighbor indices, interior only."""
    weight = np.asarray(weight, dtype=np.float64)
    if (
        weight.ndim != 4
        or weight.shape[2:] != (3, 3)
        or not np.isfinite(weight).all()
        or row_phase not in (0, 1)
        or column_phase not in (0, 1)
        or not 1 <= output_channels <= weight.shape[0]
    ):
        raise ValueError("Invalid trained-layer footprint request")
    footprint = np.zeros((output_channels, weight.shape[1], 4, 4), dtype=np.float64)
    for kr in range(3):
        for kc in range(3):
            row = (2 + row_phase + kr - 1) // 2
            column = (2 + column_phase + kc - 1) // 2
            footprint[:, :, row, column] += weight[:output_channels, :, kr, kc]
    return footprint


def _checked_weights(path):
    if path.stat().st_size != WEIGHTS_BYTES:
        raise ValueError("Released weights byte length mismatch")
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != WEIGHTS_SHA:
        raise ValueError("Released weights SHA mismatch")
    return digest.hexdigest()


def trained_assay(source_path, weights_path, layer_count=3, probe_channels=4):
    """Check trained linear sublayers; hidden-input covariance remains unknown."""
    if layer_count not in (1, 3) or probe_channels not in (1, 4):
        raise ValueError("Unregistered trained-layer probe size")
    class_ast = _pinned_class(source_path)
    weights_sha = _checked_weights(weights_path)
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    torch.manual_seed(0)
    namespace = {"torch": torch, "nn": torch.nn}
    exec(compile(class_ast, str(source_path), "exec"), namespace)
    state = load_file(str(weights_path), device="cpu")
    results = []
    started = perf_counter()
    for level in (1, 2, 3)[:layer_count]:
        prefix = f"decoder.up.{level}.upsample.conv"
        weight, bias = state[prefix + ".weight"], state[prefix + ".bias"]
        channels = weight.shape[0]
        if weight.shape != (channels, channels, 3, 3) or bias.shape != (channels,):
            raise ValueError("Unexpected released convolution shape")
        module = namespace["Upsample"](channels, True).double().eval()
        module.load_state_dict(
            {"conv.weight": weight.double(), "conv.bias": bias.double()}, strict=True
        )
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        source = torch.zeros(
            (1, channels, 4, 4), dtype=torch.float64, requires_grad=True
        )
        output = module(source)
        np.testing.assert_allclose(
            output.detach().numpy()[0],
            np.broadcast_to(bias.numpy()[:, None, None], (channels, 8, 8)),
            rtol=1e-12,
            atol=1e-12,
        )
        variances = np.zeros((probe_channels, 2, 2))
        errors = []
        first_actual = first_expected = None
        for row_phase in (0, 1):
            for column_phase in (0, 1):
                expected = trained_footprint(
                    weight.numpy(), row_phase, column_phase, probe_channels
                )
                for channel in range(probe_channels):
                    gradient = torch.autograd.grad(
                        output[0, channel, 2 + row_phase, 2 + column_phase],
                        source,
                        retain_graph=True,
                    )[0][0].numpy()
                    np.testing.assert_allclose(
                        gradient, expected[channel], rtol=1e-12, atol=1e-12
                    )
                    errors.append(float(np.max(abs(gradient - expected[channel]))))
                    variances[channel, row_phase, column_phase] = np.sum(
                        expected[channel] ** 2
                    )
                    if first_actual is None:
                        first_actual, first_expected = gradient, expected[channel]
        corrupted = first_expected.copy()
        corrupted[0, 0, 0] += 0.1
        try:
            np.testing.assert_allclose(first_actual, corrupted, rtol=1e-12, atol=1e-12)
        except AssertionError:
            caught = True
        else:
            raise ValueError("Incorrect trained coefficient was not caught")
        results.append(
            {
                "layer": prefix,
                "input_channels": channels,
                "probe_channels": probe_channels,
                "jacobian_checks": len(errors),
                "max_abs_coefficient_error": max(errors),
                "phase_variances_under_iid_input": variances.tolist(),
                "planted_coefficient_error_caught": caught,
            }
        )
    return {
        "source_commit": SOURCE_COMMIT,
        "source_sha256": PINNED_SOURCE_SHA,
        "weights_sha256": weights_sha,
        "weights_bytes": WEIGHTS_BYTES,
        "weights_revision": "629b3ad3030ce36e15e70c5db7d91df0d60c627f",
        "torch_version": torch.__version__,
        "threads": torch.get_num_threads(),
        "fixtures": results,
        "elapsed_seconds": perf_counter() - started,
        "scope": "Released ft-MSE upsampling sublayer weights and actual pinned source Jacobians; iid unit-variance input is a hypothetical condition, not measured decoder features or final RGB/source discrimination",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--length", type=int, choices=(3, 4), default=4)
    parser.add_argument("--suite", choices=("hand", "trained"), default="hand")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--trained-layers", type=int, choices=(1, 3), default=3)
    parser.add_argument("--probe-channels", type=int, choices=(1, 4), default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.suite == "trained":
        if args.weights is None or args.length != 4:
            parser.error("Trained suite requires --weights and length4")
        result = trained_assay(
            args.source, args.weights, args.trained_layers, args.probe_channels
        )
    else:
        result = assay(args.source, args.length)
    result["code_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
