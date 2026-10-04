"""Check local imports and documentation links without running experiments.

Run from any directory with Python's standard library. No data, model weights,
network access, GPU or E: drive is needed.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
SOURCE_TREES = ("src", "experiments", "tests")
PACKAGES = ("palimpsest", "origin_simulation", "experiments", "tests")
LINK = re.compile(r"!?\[[^\]\n]*\]\(([^\s)]+)\)")
DOMAINS = {
    "origin_detection",
    "screen_capture",
    "print_capture",
    "projection_capture",
    "data_preparation",
}
ROLES = {"prepare", "audit", "run", "fit", "evaluate", "plot", "benchmark", "resume"}
PATH_METHODS = {
    "open",
    "read_text",
    "read_bytes",
    "write_text",
    "write_bytes",
    "exists",
    "mkdir",
    "resolve",
    "with_suffix",
    "with_name",
    "stat",
    "unlink",
    "is_file",
    "is_dir",
    "glob",
    "rglob",
}


def experiment_issues(tree: ast.AST) -> list[str]:
    """Inspect executable module scope without executing research code."""
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in PATH_METHODS:
            receiver = node.value
            if isinstance(receiver, ast.JoinedStr) or (
                isinstance(receiver, ast.Constant) and isinstance(receiver.value, str)
            ):
                issues.append(
                    f"line {node.lineno}: path operation on a string; check division parentheses"
                )

    class ModuleScope(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            # Defaults and decorators run on import; bodies do not.
            for expr in (
                *node.decorator_list,
                *node.args.defaults,
                *node.args.kw_defaults,
            ):
                if expr is not None:
                    self.visit(expr)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            self.generic_visit(node)

        def visit_Lambda(self, node):
            self.visit(node.args)

        def visit_If(self, node):
            if ast.unparse(node.test) in (
                "__name__ == '__main__'",
                "'__main__' == __name__",
            ):
                return
            self.generic_visit(node)

        def visit_Call(self, node):
            name = ast.unparse(node.func)
            leaf = name.rsplit(".", 1)[-1]
            if (
                leaf
                in {
                    "open",
                    "read_text",
                    "read_bytes",
                    "write_text",
                    "write_bytes",
                    "exists",
                    "mkdir",
                    "stat",
                    "unlink",
                    "glob",
                    "rglob",
                }
                or leaf
                in {
                    "ZipFile",
                    "urlopen",
                    "run",
                    "Popen",
                    "print",
                    "load_pairs",
                    "load_originals",
                    "read_rgb",
                    "load_rgb",
                    "extract_frame",
                    "simulate_print_scan",
                    "render_screen_capture",
                }
                or name in {"np.load", "sys.path.insert"}
            ):
                issues.append(
                    f"line {node.lineno}: research operation at module scope: {name}"
                )
            self.generic_visit(node)

    ModuleScope().visit(tree)
    return issues


def module_bindings(tree: ast.AST) -> set[str]:
    """Collect module exports, excluding names local to functions/classes."""
    names = set()

    class Bindings(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            names.add(node.name)

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                names.add(node.id)

        def visit_Import(self, node):
            names.update(
                alias.asname or alias.name.split(".")[0] for alias in node.names
            )

        def visit_ImportFrom(self, node):
            names.update(alias.asname or alias.name for alias in node.names)

        def visit_If(self, node):
            if ast.unparse(node.test) not in (
                "__name__ == '__main__'",
                "'__main__' == __name__",
            ):
                self.generic_visit(node)

    Bindings().visit(tree)
    return names


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
    bindings = {module: module_bindings(tree) for module, tree in trees.items()}
    for path in paths:
        module = module_name(path)
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        if path.is_relative_to(ROOT / "experiments") and path.name != "__init__.py":
            relative = path.relative_to(ROOT / "experiments")
            if relative.parts[0] not in DOMAINS:
                problems.append(f"{path.relative_to(ROOT)}: unknown experiment domain")
            if path.stem != "protocol" and path.stem.split("_", 1)[0] not in ROLES:
                problems.append(f"{path.relative_to(ROOT)}: missing execution role")
            if path.stem == "protocol" and any(
                isinstance(n, ast.FunctionDef) and n.name == "main"
                for n in trees[module].body
            ):
                problems.append(
                    f"{path.relative_to(ROOT)}: protocol provider contains a runner"
                )
            for issue in experiment_issues(trees[module]):
                problems.append(f"{path.relative_to(ROOT)}: {issue}")
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

    migration = json.loads(
        (ROOT / "docs/maintenance/experiment_migration.json").read_text(
            encoding="utf-8"
        )
    )
    extractions = json.loads(
        (ROOT / "docs/maintenance/experiment_extractions.json").read_text(
            encoding="utf-8"
        )
    )
    targets = set(migration.values()) | {
        target for group in extractions.values() for target in group
    }
    for target in targets:
        if not (ROOT / target).is_file():
            problems.append(f"migration target missing: {target}")
    for old in migration:
        if (ROOT / old).exists():
            problems.append(f"obsolete experiment still present: {old}")
    active = {
        p.relative_to(ROOT).as_posix()
        for p in paths
        if p.is_relative_to(ROOT / "experiments") and p.name != "__init__.py"
    }
    # New experiments are allowed, but every migrated/extracted source must remain accounted for.
    missing = {
        target
        for target in targets
        if target.startswith("experiments/") and target.endswith(".py")
    } - active
    problems.extend(
        f"migration coverage missing: {target}" for target in sorted(missing)
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
