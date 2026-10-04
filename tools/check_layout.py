"""Check local imports and documentation links without running experiments.

Run from any directory with Python's standard library. No data, model weights,
network access, GPU or E: drive is needed.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SOURCE_TREES = ("src", "experiments", "tests")
PACKAGES = ("palimpsest", "origin_simulation", "experiments", "tests")
LINK = re.compile(r"!?\[[^\]\n]*\]\(([^\s)]+)\)")


def module_name(path: Path) -> str:
    parts = list(path.relative_to(ROOT).with_suffix("").parts)
    if parts[0] == "src":
        parts.pop(0)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def main() -> None:
    problems: list[str] = []
    paths = [path for folder in SOURCE_TREES for path in (ROOT / folder).rglob("*.py")]
    trees = {
        module_name(path): ast.parse(path.read_text(encoding="utf-8")) for path in paths
    }
    # Catch imports of helpers accidentally removed while extracting utilities.
    bindings = {}
    for module, tree in trees.items():
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.add(node.id)
            elif isinstance(node, ast.Import):
                names.update(
                    alias.asname or alias.name.split(".")[0] for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                names.update(alias.asname or alias.name for alias in node.names)
        bindings[module] = names
    for path in paths:
        module = module_name(path)
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        for node in ast.walk(trees[module]):
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    prefix = package.split(".")[
                        : len(package.split(".")) - node.level + 1
                    ]
                    base = ".".join(prefix + ([base] if base else []))
                if base.split(".")[0] in PACKAGES and base not in trees:
                    problems.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: missing module {base}"
                    )
                if base in trees and "*" not in bindings[base]:
                    for alias in node.names:
                        if (
                            alias.name != "*"
                            and alias.name not in bindings[base]
                            and f"{base}.{alias.name}" not in trees
                        ):
                            problems.append(
                                f"{path.relative_to(ROOT)}:{node.lineno}: missing symbol {base}.{alias.name}"
                            )
                if module.startswith("palimpsest.") and base.startswith("experiments"):
                    problems.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}: library imports experiment {base}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if module.startswith("palimpsest.") and alias.name.startswith(
                        "experiments"
                    ):
                        problems.append(
                            f"{path.relative_to(ROOT)}:{node.lineno}: library imports experiment {alias.name}"
                        )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.replace("\\", "/")
                if value.startswith("experiments/") and value.endswith((".py", ".ps1")):
                    if not (ROOT / value).is_file():
                        problems.append(
                            f"{path.relative_to(ROOT)}:{node.lineno}: missing source {value}"
                        )

    documents = [ROOT / "README.md", ROOT / "work/README.md"]
    for folder in ("docs", "reports", "sources", "experiments", "configs"):
        documents.extend((ROOT / folder).rglob("*.md"))
    links = 0
    for path in documents:
        for match in LINK.finditer(path.read_text(encoding="utf-8")):
            url = unquote(match[1])
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url) or url.startswith(
                ("#", "/")
            ):
                continue
            relative = url.split("#", 1)[0]
            destination = (path.parent / relative).resolve()
            if not destination.is_relative_to(ROOT):
                problems.append(
                    f"{path.relative_to(ROOT)}: link escapes repository: {url}"
                )
            elif not destination.exists():
                problems.append(f"{path.relative_to(ROOT)}: missing link {url}")
            links += 1
    if problems:
        print("\n".join(problems))
        raise SystemExit(1)
    print(
        f"OK: {len(paths)} Python files, {len(documents)} documents, {links} local links."
    )


if __name__ == "__main__":
    main()
