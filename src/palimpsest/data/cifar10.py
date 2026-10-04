"""Read source pixels from a trusted official CIFAR-10 Python archive.

This format uses pickle: only use a checksum-verified, trusted archive.
The decoder preserves planar RGB ordering and the experiment prefix keys.
"""

import io
import pickle
import tarfile
import warnings
from pathlib import Path

import numpy as np

from palimpsest.paths import DATA_ROOT

CIFAR_ARCHIVE = DATA_ROOT / "raw/cifar10_official/cifar-10-python.tar.gz"


def load_originals(
    rows: list[dict], archive_path: Path = CIFAR_ARCHIVE
) -> dict[str, np.ndarray]:
    by_batch = {}
    with tarfile.open(archive_path, "r:gz") as archive:
        for batch in sorted({row["cifar_batch"] for row in rows}):
            with archive.extractfile(f"cifar-10-batches-py/{batch}") as stream:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=r"dtype\(\): align should be passed.*"
                    )
                    payload = pickle.load(io.BytesIO(stream.read()), encoding="bytes")
            by_batch[batch] = np.asarray(payload[b"data"], dtype=np.uint8)
    return {
        row["prefix"]: by_batch[row["cifar_batch"]][int(row["row"])]
        .reshape(3, 32, 32)
        .transpose(1, 2, 0)
        for row in rows
    }
