"""Source fingerprints and local Git protocol bindings."""

import hashlib
import json
from pathlib import Path
import re
import subprocess

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


def load_committed_json(
    path: Path, revision: str, repository: Path
) -> tuple[dict, str]:
    """Require a JSON protocol to match an explicit local Git blob.

    This establishes content identity, not an external immutable timestamp.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Protocol revision must be a full commit SHA")
    relative = path.resolve().relative_to(repository.resolve()).as_posix()
    data = path.read_bytes()
    archived = subprocess.run(
        ["git", "show", f"{revision}:{relative}"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    digest = hashlib.sha256(data)
    if digest.digest() != hashlib.sha256(archived).digest():
        raise ValueError("Working protocol differs from committed blob")
    protocol = json.loads(data)
    if not isinstance(protocol, dict):
        raise ValueError("Protocol must be a JSON object")
    return protocol, digest.hexdigest()
