from pathlib import Path

import pytest

from palimpsest.paths import ResearchPaths


def test_default_paths_follow_checkout_without_creating_data(tmp_path):
    repo = tmp_path / "checkout"
    paths = ResearchPaths.load(repo=repo, environ={})
    assert paths.data == tmp_path / "data"
    assert paths.work == repo / "work"
    assert not repo.exists()


def test_relative_config_and_environment_override_are_independent_of_cwd(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "local.toml"
    config.write_text(
        '[paths]\ndata = "../inputs"\nwork = "results"\n', encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    paths = ResearchPaths.load(
        repo, Path("local.toml"), {"PALIMPSEST_DATA_ROOT": "../override"}
    )
    assert paths.data == tmp_path / "override"
    assert paths.work == repo / "results"


def test_configured_workspace_changes_default_storage_roots(tmp_path):
    paths = ResearchPaths.load(
        repo=tmp_path / "repo", environ={"PALIMPSEST_WORKSPACE_ROOT": "../storage"}
    )
    assert paths.models == tmp_path / "storage/models"
    assert paths.data == tmp_path / "storage/data"


def test_explicit_missing_config_is_not_silently_ignored(tmp_path):
    with pytest.raises(FileNotFoundError):
        ResearchPaths.load(repo=tmp_path, config=Path("missing.toml"), environ={})


def test_unknown_config_key_is_reported(tmp_path):
    config = tmp_path / "local.toml"
    config.write_text('[paths]\ndtaa = "inputs"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="dtaa"):
        ResearchPaths.load(repo=tmp_path, config=config, environ={})
