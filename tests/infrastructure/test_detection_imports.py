"""Core consumers must not require weights, datasets, torch or sklearn at import."""

import os
from pathlib import Path
import subprocess
import sys


def test_detection_foundation_imports_are_idle_and_heavy_dependencies_optional(
    tmp_path,
):
    root = Path(__file__).resolve().parents[2]
    modules = (
        "palimpsest.contracts",
        "palimpsest.localization.interfaces",
        "palimpsest.localization.baselines.quadrilateral",
        "palimpsest.localization.baselines.sam2",
        "palimpsest.detection.interfaces",
        "palimpsest.detection.baselines.bfree",
        "palimpsest.detection.baselines.d3",
        "palimpsest.detection.baselines.benford",
        "palimpsest.pipelines.image",
        "palimpsest.pipelines.cli",
        "palimpsest.training.interfaces",
    )
    code = "import importlib,sys\n" + "\n".join(
        f"importlib.import_module({m!r})" for m in modules
    )
    code += "\nassert 'torch' not in sys.modules\nassert 'sklearn' not in sys.modules\n"
    env = dict(
        os.environ,
        PYTHONPATH=str(root / "src"),
        PALIMPSEST_WORKSPACE_ROOT=str(tmp_path),
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "" and result.stderr == ""
    assert not list(tmp_path.iterdir())
