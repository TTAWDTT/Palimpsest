"""Evaluate whether five paired features distinguish RR 4:2:0 from color simulation."""

from __future__ import annotations

from palimpsest.paths import REPO_ROOT

import argparse
import hashlib
import json
from pathlib import Path

from experiments.origin_detection.platform_statistics.rr.evaluate_simulator_critic import (
    evaluate,
)


BASE = REPO_ROOT
INPUT = BASE / "work" / "rr_redigital_420_color_pilot.json"
OUTPUT = BASE / "work" / "rr_redigital_420_color_critic.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data["per_source_features"]
    if len(rows) != data["sources"] or len({row["source"] for row in rows}) != len(
        rows
    ):
        raise ValueError("source coverage mismatch")
    result = {
        "scope": "5-feature source-disjoint two-sample critic on RR internal 4:2:0 development stratum",
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "variants": {
            variant: evaluate(
                [
                    {
                        "source": row["source"],
                        "real": row["real"],
                        "simulated": row[variant],
                    }
                    for row in rows
                ],
                False,
            )
            for variant in ("baseline", "color")
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
