"""Model source fingerprints used to invalidate simulation caches."""

import hashlib
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parents[1] / "simulation"


def simulation_code_files() -> tuple[Path, ...]:
    return tuple(sorted(MODEL_ROOT.rglob("*.py")))


def simulation_code_sha256() -> str:
    digest = hashlib.sha256()
    for path in simulation_code_files():
        digest.update(path.relative_to(MODEL_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
