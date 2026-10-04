"""Repository paths shared by historical experiments.

Source code lives in experiments/. Existing manifests, scores and audit files
stay in work/; raw data and model weights retain their recorded E: paths.
Run experiments from the repository root. Importing this module creates nothing.
"""

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = REPO_ROOT / "work"


def simulation_code_files() -> tuple[Path, ...]:
    """Include every model stage in experiment cache invalidation."""
    return tuple(sorted((REPO_ROOT / "origin_simulation").rglob("*.py")))


def simulation_code_sha256() -> str:
    """Hash relative filenames and content, independently of checkout location."""
    digest = hashlib.sha256()
    for path in simulation_code_files():
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
