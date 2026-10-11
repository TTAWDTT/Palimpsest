"""One-shot full AE mechanism probe, not AEROBLADE or a source classifier.

Execute only pinned source definitions with complete strict parameter loads.
Bind the input protocol to an already committed blob before any pixel read.
"""

import argparse
import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter
from types import MethodType

import numpy as np
from PIL import Image

from palimpsest.data.images import resize_unit_float
from palimpsest.evaluation.polyphase import spatial_phase_moments
from palimpsest.io.hashing import file_sha256
from palimpsest.io.provenance import load_committed_json
from palimpsest.paths import DATA_ROOT, REPO_ROOT
from .audit_ldm_operator import _checked_weights, pinned_source_ast, trained_footprint


PROTOCOL_RELATIVE = "configs/evaluation/decoder_chain_probe.json"
SOURCE_DEFINITIONS = (
    "nonlinearity",
    "Normalize",
    "Upsample",
    "Downsample",
    "ResnetBlock",
    "AttnBlock",
    "make_attn",
    "Encoder",
    "Decoder",
)


def committed_protocol(path, revision):
    protocol, digest = load_committed_json(path, revision, REPO_ROOT)
    if protocol["schema_version"] != 1 or protocol["side"] != 128:
        raise ValueError("Unregistered decoder-chain schema or resolution")
    if any(
        p["role"] != "fit" or p["condition"] != "original" for p in protocol["probes"]
    ):
        raise ValueError("Only registered fit originals are eligible")
    return protocol, digest


def indexed_prediction(
    hidden, weight, bias, row_phase, column_phase, output_channels=4, *, mode="nearest"
):
    """Predict valid interior outputs using measured hidden inputs, no iid model."""
    hidden = np.asarray(hidden, dtype=np.float64)
    if (
        hidden.ndim != 3
        or hidden.shape[1] != hidden.shape[2]
        or hidden.shape[1] < 4
        or not np.isfinite(hidden).all()
    ):
        raise ValueError("Expected finite CxNxN hidden input with N>=4")
    footprint = trained_footprint(
        weight, row_phase, column_phase, output_channels, mode=mode
    )
    side = hidden.shape[1]
    prediction = np.broadcast_to(
        np.asarray(bias)[:output_channels, None, None],
        (output_channels, side - 2, side - 2),
    ).copy()
    for r in range(4):
        for c in range(4):
            coefficients = footprint[:, :, r, c]
            if not np.any(coefficients):
                continue
            patch = hidden[:, r : r + side - 2, c : c + side - 2]
            prediction += np.einsum("oc,cyx->oyx", coefficients, patch, optimize=True)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite predicted hidden response")
    return prediction


def load_components(source_path, weights_path, protocol):
    source_ast = pinned_source_ast(source_path, SOURCE_DEFINITIONS)
    _checked_weights(weights_path)
    import torch
    from safetensors.torch import load_file

    torch.set_num_threads(1)
    torch.manual_seed(0)
    namespace = {"torch": torch, "nn": torch.nn, "np": np}
    exec(compile(source_ast, str(source_path), "exec"), namespace)
    config = protocol["architecture"]
    components = {
        "encoder": namespace["Encoder"](**config, double_z=True),
        "decoder": namespace["Decoder"](**config),
        "quant_conv": torch.nn.Conv2d(8, 8, 1),
        "post_quant_conv": torch.nn.Conv2d(4, 4, 1),
    }
    state = load_file(str(weights_path), device="cpu")
    used = set()
    for name, module in components.items():
        selected = {
            key[len(name) + 1 :]: value
            for key, value in state.items()
            if key.startswith(name + ".")
        }
        module.load_state_dict(selected, strict=True)
        module.double().eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        used.update(name + "." + key for key in selected)
    declared = protocol.get("declared_non_forward_buffers", {})
    if set(state) - used != set(declared):
        raise ValueError("Released state has unused or unaccounted tensors")
    for key, expected in declared.items():
        observed = {
            "dtype": str(state[key].dtype).removeprefix("torch."),
            "shape": list(state[key].shape),
        }
        if observed != expected:
            raise ValueError("Declared non-forward state shape/dtype differs")
    return (
        torch,
        components,
        {
            "loaded_forward_tensors": len(used),
            "released_tensor_count": len(state),
            "accounted_non_forward_buffers": sorted(declared),
        },
    )


def field_metrics(rgb):
    """Raw model float RGB; no quantization or inference decision."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[0] != 3 or not np.isfinite(rgb).all():
        raise ValueError("Expected finite3xHxW model RGB")
    luminance = np.einsum("c,cyx->yx", np.array([0.299, 0.587, 0.114]), rgb)
    laplacian = (
        4 * luminance[1:-1, 1:-1]
        - luminance[:-2, 1:-1]
        - luminance[2:, 1:-1]
        - luminance[1:-1, :-2]
        - luminance[1:-1, 2:]
    )
    return {
        "luminance": spatial_phase_moments(luminance[2:-2, 2:-2]),
        "laplacian": spatial_phase_moments(laplacian[1:-1, 1:-1]),
        "rgb_range": [float(rgb.min()), float(rgb.max())],
        "scope": "Single raw normalized-float field with2-pixel boundary removal; spatial summaries, not ensemble orAI evidence",
    }


def run_probe(torch, components, pixels, protocol):
    side = protocol["side"]
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.shape != (side, side, 3) or not np.isfinite(pixels).all():
        raise ValueError("Probe pixels differ from registered shape")
    captures = {}
    handles = []
    for level in (1, 2, 3):

        def capture(module, inputs, output, level=level):
            captures[level] = (
                inputs[0].detach().numpy()[0].copy(),
                output.detach().numpy()[0].copy(),
            )

        handles.append(
            components["decoder"].up[level].upsample.register_forward_hook(capture)
        )
    started = perf_counter()
    try:
        with torch.inference_mode():
            image = torch.from_numpy(pixels.transpose(2, 0, 1)[None].copy())
            encoded = components["quant_conv"](components["encoder"](image))
            mean = encoded[:, :4]  # Source DiagonalGaussianDistribution.mode.
            decoded = components["decoder"](components["post_quant_conv"](mean))
            output = decoded.numpy()[0].copy()
    finally:
        for handle in handles:
            handle.remove()
    if set(captures) != {1, 2, 3} or output.shape != (3, side, side):
        raise ValueError("Incomplete decoder stages or wrongRGB shape")
    rows = []
    for level in (1, 2, 3):
        hidden, actual = captures[level]
        module = components["decoder"].up[level].upsample.conv
        weight, bias = module.weight.detach().numpy(), module.bias.detach().numpy()
        side_hidden = hidden.shape[1]
        errors = []
        for r in (0, 1):
            for c in (0, 1):
                predicted = indexed_prediction(hidden, weight, bias, r, c)
                observed = actual[
                    :4, 2 + r : 2 * side_hidden - 2 : 2, 2 + c : 2 * side_hidden - 2 : 2
                ]
                np.testing.assert_allclose(
                    observed,
                    predicted,
                    rtol=protocol["linear_response_rtol"],
                    atol=protocol["linear_response_atol"],
                )
                errors.append(float(np.max(abs(observed - predicted))))
                if r == c == 0:
                    corrupted = observed.copy()
                    corrupted[0, 0, 0] += 0.1
                    try:
                        np.testing.assert_allclose(
                            corrupted,
                            predicted,
                            rtol=protocol["linear_response_rtol"],
                            atol=protocol["linear_response_atol"],
                        )
                    except AssertionError:
                        caught = True
                    else:
                        raise ValueError("Corrupted real-hidden output survived")
        rows.append(
            {
                "level": level,
                "hidden_shape": list(hidden.shape),
                "max_abs_prediction_error": max(errors),
                "output_corruption_caught": caught,
                "output_phase_moments": [
                    spatial_phase_moments(v[2:-2, 2:-2]) for v in actual[:4]
                ],
            }
        )
    result = {
        "stages": rows,
        "input_pixel_metrics": field_metrics(pixels.transpose(2, 0, 1)),
        "decoder_float_metrics": field_metrics(output),
        "elapsed_seconds": perf_counter() - started,
    }
    intervention = protocol.get("interpolation_intervention")
    if intervention:
        level = intervention["level"]
        target = components["decoder"].up[level].upsample
        original_forward = target.forward
        alternative = {}

        def changed_forward(module, x):
            resized = torch.nn.functional.interpolate(
                x, scale_factor=2.0, mode="bilinear", align_corners=False
            )
            return module.conv(resized)

        def changed_capture(module, inputs, output):
            alternative["hidden"] = inputs[0].detach().numpy()[0].copy()
            alternative["output"] = output.detach().numpy()[0].copy()

        target.forward = MethodType(changed_forward, target)
        handle = target.register_forward_hook(changed_capture)
        try:
            with torch.inference_mode():
                changed = (
                    components["decoder"](components["post_quant_conv"](mean))
                    .numpy()[0]
                    .copy()
                )
        finally:
            handle.remove()
            target.forward = original_forward
        np.testing.assert_array_equal(alternative["hidden"], captures[level][0])
        weight = target.conv.weight.detach().numpy()
        bias = target.conv.bias.detach().numpy()
        side_hidden = alternative["hidden"].shape[1]
        errors = []
        for r in (0, 1):
            for c in (0, 1):
                expected = indexed_prediction(
                    alternative["hidden"], weight, bias, r, c, mode="bilinear"
                )
                observed = alternative["output"][
                    :4, 2 + r : 2 * side_hidden - 2 : 2, 2 + c : 2 * side_hidden - 2 : 2
                ]
                np.testing.assert_allclose(
                    observed,
                    expected,
                    rtol=protocol["linear_response_rtol"],
                    atol=protocol["linear_response_atol"],
                )
                errors.append(float(np.max(abs(observed - expected))))
        result["interpolation_intervention"] = {
            "level": level,
            "same_hidden_input": True,
            "same_weights": True,
            "max_abs_prediction_error": max(errors),
            "decoder_float_metrics": field_metrics(changed),
            "rgb_mse_full": float(np.mean((changed - output) ** 2)),
            "rgb_mse_interior": float(
                np.mean((changed[:, 2:-2, 2:-2] - output[:, 2:-2, 2:-2]) ** 2)
            ),
            "scope": "Single interpolation operation changed without retraining; rawfloat model output, not a detector or equal-quality generator comparison",
        }
        result["elapsed_seconds"] = perf_counter() - started
    ordinary = np.repeat(np.repeat(pixels[::2, ::2], 2, axis=0), 2, axis=1)
    result["ordinary_processing_control"] = field_metrics(ordinary.transpose(2, 0, 1))
    return result


def validated_probes(protocol):
    csv_path = REPO_ROOT / protocol["historical_metadata_csv"]
    if file_sha256(csv_path) != protocol["historical_metadata_sha256"]:
        raise ValueError("Historical metadata CSV SHA changed")
    selected = {p["filename"]: p for p in protocol["probes"]}
    found = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["filename"] not in selected or row["variant"] != "raw":
                continue
            expected = selected[row["filename"]]
            for key in ("sha256", "role", "condition", "domain", "label"):
                if row[key] != expected[key]:
                    raise ValueError("Selected image metadata differs")
            if row["filename"] in found or row["role"] != "fit":
                raise ValueError("Duplicate or non-fit probe")
            found[row["filename"]] = expected
    if set(found) != set(selected) or len(found) != 4:
        raise ValueError("Registered4-image probe coverage changed")
    return [found[p["filename"]] for p in protocol["probes"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=REPO_ROOT / PROTOCOL_RELATIVE)
    parser.add_argument("--protocol-revision", required=True)
    parser.add_argument("--mode", choices=("chart", "images"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, protocol_sha = committed_protocol(args.protocol, args.protocol_revision)
    torch, components, state_accounting = load_components(
        args.source, args.weights, protocol
    )
    if args.mode == "chart":
        axis = np.linspace(-1, 1, protocol["side"])
        y, x = np.meshgrid(axis, axis, indexing="ij")
        inputs = [("synthetic_chart", np.stack((x, y, (x + y) / 2), axis=-1))]
    else:
        inputs = []
        for probe in validated_probes(protocol):
            path = (DATA_ROOT / probe["data_relative"]).resolve()
            if not path.is_relative_to(DATA_ROOT.resolve()):
                raise ValueError("Probe image escapes registered data root")
            data = path.read_bytes()
            if sha256(data).hexdigest() != probe["sha256"]:
                raise ValueError("Probe file SHA changed")
            with Image.open(BytesIO(data)) as image:
                if image.size != (probe["width"], probe["height"]):
                    raise ValueError("Probe native dimensions changed")
                pixels = (
                    resize_unit_float(image.convert("RGB"), protocol["side"]).astype(
                        np.float64
                    )
                    * 2
                    - 1
                )
            inputs.append((probe["filename"], pixels))
    results = []
    for name, pixels in inputs:
        results.append(
            {"filename": name, **run_probe(torch, components, pixels, protocol)}
        )
        print(json.dumps({"completed": len(results), "total": len(inputs)}), flush=True)
    result = {
        "protocol_revision": args.protocol_revision,
        "protocol_sha256": protocol_sha,
        "code_sha256": file_sha256(Path(__file__)),
        "mode": args.mode,
        "state_accounting": state_accounting,
        "probes": results,
        "scope": protocol["purpose"],
        "limits": protocol["scientific_limits"],
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
