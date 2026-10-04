"""Decode one real projector-camera Gray-code capture sequence.

The source pattern stack itself defines 10-bit codebooks. No assumption about
Gray-code polarity/order is needed. Output coordinates are projector *pixels*,
not world depth. Only run after 84 selected members pass ZIP CRC."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

from experiments.projection_capture.structured_light.compennetpp.protocol import (
    FOLDER,
    MANIFEST,
    OUT,
    VIS,
    decode_setup,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--folder", type=Path, default=FOLDER)
    parser.add_argument("--output-json", type=Path, default=OUT)
    parser.add_argument("--preview", type=Path, default=VIS)
    args = parser.parse_args()
    result = decode_setup(args.manifest, args.folder, args.output_json, args.preview)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("source_axis_x", "source_axis_y")
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
