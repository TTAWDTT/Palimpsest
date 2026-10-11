"""Audit cached native mechanism receipts and summarize without model inference.

This is a packet-specific evaluator. Reuses pure metric; no fitted classifier,
significance test or independent validation is produced.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from palimpsest.evaluation.cyclic_image import cyclic_rgb_summary
from palimpsest.io.hashing import file_sha256


def verify_metrics(expected, measured):
    assert expected["patch_grid"] == measured["patch_grid"] == [14, 14]
    for key, groups in expected["measurements"].items():
        for old, new in zip(groups, measured["measurements"][key], strict=True):
            assert old["group"] == new["group"]
            assert (
                old["patch_count"]
                == new["patch_count"]
                == (196 if old["group"] == "all" else 49)
            )
            for name in ("real", "imag"):
                np.testing.assert_allclose(
                    old["cumulant"][name], new["cumulant"][name], rtol=1e-7, atol=1e-9
                )
            a, b = old["normalized_magnitude"], new["normalized_magnitude"]
            if a is None or b is None:
                assert a is b
            else:
                assert np.isfinite(a) and np.isfinite(b)
                np.testing.assert_allclose(a, b, rtol=1e-7, atol=1e-9)


def evaluate(root):
    groups = []
    all_rows = []
    response_errors = []
    metric_times = []
    protocol_hashes = set()
    summary = []
    for family in ("stylegan2", "stylegan3"):
        receipt = json.loads(
            (root / f"{family}_complete.json").read_text(encoding="utf-8")
        )
        protocol_hashes.add(receipt["protocol_sha256"])
        assert [r["seed"] for r in receipt["images"]] == list(range(8))
        assert receipt["reused_pilot_count"] == 3
        for image in receipt["images"]:
            assert image["model_state_unchanged"] and image["rgb_corruption_caught"]
            response_errors.append(image["rgb_delta_prediction_error"])
            a, b = image["branches"]
            assert [a["branch"], b["branch"]] == ["baseline", "linearized"]
            assert a["capture"]["input_sha256"] == b["capture"]["input_sha256"]
            assert a["capture"]["bias_sha256"] == b["capture"]["bias_sha256"]
            for branch in image["branches"]:
                assert branch["capture"]["checked_channels"] == 4
                assert branch["capture"]["corruption_caught"]
                fp = Path(branch["float_file"])
                assert file_sha256(fp) == branch["float_file_sha256"]
                rgb = np.load(fp, allow_pickle=False)
                assert rgb.shape == (256, 256, 3) and rgb.dtype == np.float32
                assert (
                    file_sha256(Path(branch["png"]["file"])) == branch["png"]["sha256"]
                )
                assert np.isfinite(rgb).all()
                verify_metrics(branch["float_metrics"], cyclic_rgb_summary(rgb))
                with Image.open(branch["png"]["file"]) as im:
                    encoded = np.asarray(im.convert("RGB"))
                verify_metrics(
                    branch["png"]["metrics"],
                    cyclic_rgb_summary(encoded.astype(float) * (2 / 255) - 1),
                )
                metric_times.append(branch["png"]["decode_metric_seconds"])
                for encoding, metrics in (
                    ("float", branch["float_metrics"]),
                    ("PNG", branch["png"]["metrics"]),
                ):
                    for key, rows in metrics["measurements"].items():
                        groups.append(
                            (
                                family,
                                image["seed"],
                                branch["branch"],
                                encoding,
                                key,
                                rows,
                            )
                        )
                        all_rows.extend(
                            {
                                "family": family,
                                "seed": image["seed"],
                                "branch": branch["branch"],
                                "encoding": encoding,
                                "measure": key,
                                **row,
                            }
                            for row in rows
                        )
        for key in ("x_cross", "x_pure", "y_cross", "y_pure"):
            pairs = [
                [
                    b["png"]["metrics"]["measurements"][key][0]["normalized_magnitude"]
                    for b in i["branches"]
                ]
                for i in receipt["images"]
            ]
            data = np.array(pairs, dtype=float)
            summary.append(
                {
                    "family": family,
                    "measure": key,
                    "n_pairs": 8,
                    "baseline_median": float(np.median(data[:, 0])),
                    "linearized_median": float(np.median(data[:, 1])),
                    "increased_pairs": int(np.sum(data[:, 1] > data[:, 0])),
                    "decreased_pairs": int(np.sum(data[:, 1] < data[:, 0])),
                    "baseline_range": [
                        float(data[:, 0].min()),
                        float(data[:, 0].max()),
                    ],
                }
            )
    assert len(protocol_hashes) == 1 and len(all_rows) == 1280
    largest = max(
        (r for r in all_rows if r["normalized_magnitude"] is not None),
        key=lambda r: r["normalized_magnitude"],
    )
    photos = json.loads((root / "photos.json").read_text(encoding="utf-8"))
    assert photos["protocol_sha256"] in protocol_hashes and len(photos["photos"]) == 2
    for index, photo in enumerate(photos["photos"]):
        assert [v["variant"] for v in photo["variants"]] == [
            "baseline",
            "NN",
            "bicubic",
        ]
        for variant in photo["variants"]:
            path = root / "photos" / f"photo{index}_{variant['variant']}.png"
            assert file_sha256(path) == variant["sha256"]
            with Image.open(path) as im:
                rgb = np.asarray(im.convert("RGB"))
            verify_metrics(
                variant["metrics"],
                cyclic_rgb_summary(rgb.astype(float) * (2 / 255) - 1),
            )
    photo_table = [
        {
            "source": p["source"],
            "variant": v["variant"],
            **{
                k: rows[0]["normalized_magnitude"]
                for k, rows in v["metrics"]["measurements"].items()
            },
        }
        for p in photos["photos"]
        for v in p["variants"]
    ]
    return {
        "native_pairs": 16,
        "native_branches": 32,
        "native_metric_rows": len(all_rows),
        "recomputed_encodings": 64,
        "max_rgb_response_error": max(response_errors),
        "summary": summary,
        "photo_table": photo_table,
        "largest_descriptor": largest,
        "decode_metric_seconds": {
            "count": len(metric_times),
            "p50": float(np.median(metric_times)),
            "p95": float(np.percentile(metric_times, 95)),
            "scope": "Recordednative256PNG decode, pixelcheck,luma anddescriptor; not AI detection or devicebenchmark",
        },
        "scope": "Cacheddevelopment mechanism summary; content unmatched, no BA/AUC or formal CI",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.root)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "native_pairs",
                    "native_branches",
                    "recomputed_encodings",
                    "max_rgb_response_error",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
