"""Test chart-only projector forward models on held-out textured inputs.

This is a deliberately restricted, effective RGB experiment on the author's
warped 256px CompenNet PNGs. It does not recover physical projector, camera or
surface parameters individually. No test image is used to fit parameters."""

from __future__ import annotations
import argparse
import json
import time
import zipfile

from experiments.projection_capture.chart_response.compennet.protocol import (
    ARCHIVE,
    AUDIT,
    OUT,
    run_setup,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setups",
        nargs="+",
        default=[
            "light1/pos1/stripes",
            "light2/pos1/curves",
            "light3/pos3/squares",
            "light4/pos2/curves",
        ],
    )
    parser.add_argument(
        "--all", action="store_true", help="evaluate every CRC-audited setup"
    )
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--test-count", type=int, default=200)
    args = parser.parse_args()
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if args.all:
        args.setups = sorted(audit["setups"])
    if not audit["all_member_crc_passed"] or any(
        s not in audit["setups"] for s in args.setups
    ):
        raise ValueError("archive/setup is not signed by full CRC audit")
    started = time.perf_counter()
    results = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for setup in args.setups:
            result = run_setup(archive, setup, args.train_count, args.test_count)
            results.append(result)
            print(
                setup,
                result["chosen_sigma"],
                {k: round(v["mae"], 4) for k, v in result["test_aggregate"].items()},
                flush=True,
            )
            OUT.write_text(
                json.dumps(
                    {
                        "archive_sha256": audit["archive_sha256"],
                        "elapsed_sec": time.perf_counter() - started,
                        "results": results,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
