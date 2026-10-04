"""Research paths with environment > local TOML > sibling-directory defaults.

Loading paths never creates directories or touches datasets. Relative settings
are resolved against the repository so experiments do not depend on shell cwd.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib
from typing import Mapping


def repository_root() -> Path:
    override = os.environ.get("PALIMPSEST_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    for start in (Path(__file__).resolve(), Path.cwd()):
        for candidate in (start, *start.parents):
            if (candidate / "pyproject.toml").is_file() and (
                candidate / "experiments"
            ).is_dir():
                return candidate
    return Path.cwd().resolve()


@dataclass(frozen=True)
class ResearchPaths:
    repo: Path
    workspace: Path
    data: Path
    models: Path
    envs: Path
    cache: Path
    work: Path
    reports: Path

    @classmethod
    def load(
        cls,
        repo: Path | None = None,
        config: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "ResearchPaths":
        env = os.environ if environ is None else environ
        root = (repo if repo is not None else repository_root()).resolve()
        config_value = env.get("PALIMPSEST_CONFIG")
        config_path = config or (
            Path(config_value)
            if config_value
            else root / "configs/workspace.local.toml"
        )
        if not config_path.is_absolute():
            config_path = root / config_path
        explicit = config is not None or bool(config_value)
        if explicit and not config_path.is_file():
            raise FileNotFoundError(config_path)
        settings = {}
        if config_path.is_file():
            with config_path.open("rb") as stream:
                settings = tomllib.load(stream).get("paths", {})
            allowed = {
                "workspace",
                "data",
                "models",
                "envs",
                "cache",
                "work",
                "reports",
            }
            unknown = set(settings) - allowed
            if unknown:
                raise ValueError(f"Unknown path settings: {sorted(unknown)}")

        def resolve(name: str, default: Path) -> Path:
            value = env.get(
                f"PALIMPSEST_{name.upper()}_ROOT", settings.get(name, default)
            )
            if not isinstance(value, (str, Path)) or not str(value):
                raise ValueError(f"Invalid path setting: {name}")
            path = Path(value).expanduser()
            return (path if path.is_absolute() else root / path).resolve()

        workspace = resolve("workspace", root.parent)
        return cls(
            repo=root,
            workspace=workspace,
            data=resolve("data", workspace / "data"),
            models=resolve("models", workspace / "models"),
            envs=resolve("envs", workspace / "envs"),
            cache=resolve("cache", workspace / "cache"),
            work=resolve("work", root / "work"),
            reports=resolve("reports", root / "reports"),
        )


PATHS = ResearchPaths.load()
REPO_ROOT = PATHS.repo
WORKSPACE_ROOT = PATHS.workspace
DATA_ROOT = PATHS.data
MODELS_ROOT = PATHS.models
ENVS_ROOT = PATHS.envs
CACHE_ROOT = PATHS.cache
WORK_DIR = PATHS.work
REPORTS_DIR = PATHS.reports
