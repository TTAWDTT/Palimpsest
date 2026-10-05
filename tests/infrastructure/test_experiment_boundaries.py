import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "layout_check", ROOT / "tools/check_layout.py"
)
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


@pytest.mark.parametrize(
    "source",
    [
        'x = ROOT / "audit.json".read_text()',
        'x = Image.open(ROOT / "photo.png")',
        "with zipfile.ZipFile(ARCHIVE) as z: pass",
        "x = simulate_print_scan(source, parameters)",
        'def helper(value=Path("data.csv").read_text()): pass',
    ],
)
def test_checker_rejects_import_time_research_or_path_precedence_errors(source):
    assert layout.experiment_issues(ast.parse(source))


def test_checker_allows_declared_paths_and_explicit_entry():
    source = 'OUT = ROOT / "audit.json"\ndef main():\n    OUT.write_text("result")\nif __name__ == "__main__":\n    main()'
    assert not layout.experiment_issues(ast.parse(source))


def test_export_check_does_not_mistake_function_locals_for_module_exports():
    tree = ast.parse(
        "from pathlib import Path\nROOT = Path('.')\ndef main():\n    hidden = 1\n    def helper(): pass\nclass Settings:\n    internal = 2"
    )
    assert layout.module_bindings(tree) == {"Path", "ROOT", "main", "Settings"}


def test_previously_executing_modules_and_shared_protocols_import_without_data(
    tmp_path,
):
    inventory = json.loads(
        (ROOT / "docs/maintenance/inventory_before.json").read_text(encoding="utf-8")
    )
    migration = json.loads(
        (ROOT / "docs/maintenance/experiment_migration.json").read_text(
            encoding="utf-8"
        )
    )
    modules = {
        migration[row["file"]][:-3].replace("/", ".")
        for row in inventory["files"]
        if row["top_level_io"] and row["file"] in migration
    }
    modules.update(
        p.relative_to(ROOT).with_suffix("").as_posix().replace("/", ".")
        for p in (ROOT / "experiments").rglob("protocol.py")
    )
    modules.add(
        "experiments.print_capture.integration.synthetic.audit_print_scan_convergence"
    )
    modules.add("experiments.origin_detection.baselines.run_d3_rr")
    code = "import importlib\n" + "\n".join(
        f"importlib.import_module({name!r})" for name in sorted(modules)
    )
    env = os.environ | {
        "PALIMPSEST_WORKSPACE_ROOT": str(tmp_path),
        "PALIMPSEST_WORK_DIR": str(tmp_path / "results"),
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert not list(tmp_path.iterdir())


def test_migrated_writer_performs_real_file_write_with_path_division(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.syspath_prepend(str(ROOT))
    from experiments.data_preparation.csgc import audit_additional_pairs as workflow

    paths = {}
    source = (np.indices((100, 100)).sum(axis=0) % 2 * 255).astype(np.uint8)
    for spi, factor in (("2400", 4), ("4800", 8), ("9600", 16)):
        path = tmp_path / f"{spi}.png"
        Image.fromarray(source.repeat(factor, axis=0).repeat(factor, axis=1)).save(path)
        paths[spi] = path
    monkeypatch.setattr(
        workflow,
        "extract",
        lambda spi, folder, item: (paths[spi], {"sha256": "synthetic-fixture"}),
    )
    monkeypatch.setattr(workflow, "WORK_DIR", tmp_path)
    workflow.main()
    report = json.loads((tmp_path / "csgc_cross_resolution_more_ids.json").read_text())
    assert len(report) == 3
    assert all(row["all_templates_equal"] for row in report.values())
    capsys.readouterr()


def test_extracted_protocol_is_in_cache_fingerprint(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    from experiments.screen_capture.spectral_response.raw2event import protocol

    assert Path(protocol.__file__) in protocol.CODE
    assert any(
        p.name == "protocol.py" and "source_to_raw" in p.parts for p in protocol.CODE
    )
    dependency = tmp_path / "protocol.py"
    dependency.write_text("first revision")
    monkeypatch.setattr(protocol, "CODE", (dependency,))
    first = protocol.code_hash()
    dependency.write_text("second revision")
    assert protocol.code_hash() != first
