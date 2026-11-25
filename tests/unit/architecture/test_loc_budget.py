"""Architecture invariant tests: LOC budgets, shadow-tree shims, and LLM-free HCIR core.

Enforces the rescoped gates from ``reports/audit/2026-11-18_full_project_audit.md`` §8.4:

1. Total lines across ``hbllm/**/*.py`` must stay at or below 300,000 LOC.
2. Total lines across ``hbllm/hcir/**/*.py`` must stay at or below 150,000 LOC.
3. ``__path__.append`` shadow-tree shims into ``experiments/`` may only decrease (ratchet).
4. ``hbllm/hcir/**`` must never import LLM clients or the LLM scaffold plugin.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
HBLLM_DIR = ROOT_DIR / "hbllm"
HCIR_DIR = HBLLM_DIR / "hcir"

LOC_BUDGET = 300_000
HCIR_LOC_BUDGET = 150_000

# Ratchet: number of ``__path__.append`` shims may only go down. Target is 0.
MAX_PATH_APPEND_SHIMS = 0

FORBIDDEN_HCIR_IMPORT_PREFIXES = (
    "openai",
    "anthropic",
    "litellm",
    "transformers",
    "plugins.llm_scaffold",
    "hbllm.serving.provider",
)


def _iter_python_files(base: Path) -> list[Path]:
    return [p for p in sorted(base.rglob("*.py")) if "__pycache__" not in p.parts]


def _count_lines(base: Path) -> tuple[int, int]:
    total = 0
    files = _iter_python_files(base)
    for path in files:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            total += sum(1 for _ in handle)
    return total, len(files)


def test_hbllm_python_loc_within_budget() -> None:
    """Assert total line count across ``hbllm/**/*.py`` is <= 300,000."""
    total_lines, file_count = _count_lines(HBLLM_DIR)
    assert file_count > 0
    assert total_lines <= LOC_BUDGET, (
        f"hbllm/**/*.py total LOC {total_lines:,} exceeds budget {LOC_BUDGET:,} "
        f"across {file_count} files"
    )


def test_hcir_python_loc_within_budget() -> None:
    """Assert total line count across ``hbllm/hcir/**/*.py`` is <= 150,000."""
    total_lines, file_count = _count_lines(HCIR_DIR)
    assert file_count > 0
    assert total_lines <= HCIR_LOC_BUDGET, (
        f"hbllm/hcir/**/*.py total LOC {total_lines:,} exceeds budget {HCIR_LOC_BUDGET:,} "
        f"across {file_count} files"
    )


def test_path_append_shim_ratchet() -> None:
    """``__path__.append`` shadow-tree shims may only decrease (target: zero)."""
    shims: list[str] = []
    for path in _iter_python_files(HBLLM_DIR):
        src = path.read_text(encoding="utf-8", errors="ignore")
        if "__path__.append" in src:
            shims.append(str(path.relative_to(ROOT_DIR)))
    assert len(shims) <= MAX_PATH_APPEND_SHIMS, (
        f"{len(shims)} __path__.append shims exceed ratchet {MAX_PATH_APPEND_SHIMS}:\n"
        + "\n".join(shims)
    )


def test_zero_llm_imports_in_hcir() -> None:
    """``hbllm/hcir/**`` must never import LLM clients or the LLM scaffold."""
    violations: list[str] = []
    for path in _iter_python_files(HCIR_DIR):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if any(
                    name == p or name.startswith(f"{p}.") for p in FORBIDDEN_HCIR_IMPORT_PREFIXES
                ):
                    violations.append(f"{path.relative_to(ROOT_DIR)}:{node.lineno} imports {name}")
    assert not violations, "LLM imports inside hbllm/hcir:\n" + "\n".join(violations)


def test_zero_shimmed_imports_in_hcir_and_runtime() -> None:
    """``hbllm/hcir/**`` and ``hbllm/runtime/**`` must not import ``hbllm.actions`` or experiments shims."""
    shimmed_packages: list[str] = []
    for path in _iter_python_files(HBLLM_DIR):
        if path.name != "__init__.py":
            continue
        src = path.read_text(encoding="utf-8", errors="ignore")
        if "__path__.append" in src:
            rel_parts = path.relative_to(ROOT_DIR).parent.parts
            shimmed_packages.append(".".join(rel_parts))

    # P2 perception provider consolidation still bridges legacy audio/speech/vision adapters
    _p2_perception_allowlist: set[str] = set()

    violations: list[str] = []
    for base_dir in (HCIR_DIR, HBLLM_DIR / "runtime"):
        for path in _iter_python_files(base_dir):
            rel_str = str(path.relative_to(ROOT_DIR))
            if rel_str in _p2_perception_allowlist:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    matched = next(
                        (
                            pkg
                            for pkg in shimmed_packages
                            if name == pkg or name.startswith(f"{pkg}.")
                        ),
                        None,
                    )
                    if matched is None:
                        continue
                    # Allow real non-experiment modules directly inside hbllm/<pkg>/
                    mod_rel = Path(*name.split("."))
                    real_mod_exists = (ROOT_DIR / mod_rel).with_suffix(".py").exists() or (
                        ROOT_DIR / mod_rel / "__init__.py"
                    ).exists()
                    if matched != "hbllm.actions" and real_mod_exists:
                        continue
                    violations.append(f"{rel_str}:{node.lineno} imports shimmed {name}")

    assert not violations, "Shimmed experiment imports inside hcir/runtime:\n" + "\n".join(
        violations
    )


def _package_exports_symbol(pkg_init: Path, symbol: str, seen: set[Path] | None = None) -> bool:
    """Check whether a package __init__.py defines, imports, or star-reexports ``symbol``."""
    if seen is None:
        seen = set()
    if pkg_init in seen or not pkg_init.exists():
        return False
    seen.add(pkg_init)
    pkg_text = pkg_init.read_text(encoding="utf-8", errors="ignore")
    if symbol in pkg_text:
        return True
    try:
        pkg_tree = ast.parse(pkg_text, filename=str(pkg_init))
    except SyntaxError:
        return False
    for node in ast.walk(pkg_tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and any(a.name == "*" for a in node.names)
        ):
            target_rel = Path(*node.module.split("."))
            for cand in (
                (ROOT_DIR / target_rel).with_suffix(".py"),
                ROOT_DIR / target_rel / "__init__.py",
            ):
                if cand.exists() and _package_exports_symbol(cand, symbol, seen):
                    return True
    return False


def test_all_hbllm_internal_imports_resolve_to_shipped_modules() -> None:
    """Every ``hbllm.*`` import inside ``hbllm/`` must resolve to a shipped module (no missing experiments)."""
    missing: list[str] = []
    for path in _iter_python_files(HBLLM_DIR):
        rel_str = str(path.relative_to(ROOT_DIR))
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith("hbllm."):
                        continue
                    mod_rel = Path(*alias.name.split("."))
                    exists = (ROOT_DIR / mod_rel).with_suffix(".py").exists() or (
                        ROOT_DIR / mod_rel / "__init__.py"
                    ).exists()
                    if not exists:
                        missing.append(f"{rel_str}:{node.lineno} imports missing {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                if not (node.module == "hbllm" or node.module.startswith("hbllm.")):
                    continue
                mod_rel = Path(*node.module.split("."))
                mod_py = (ROOT_DIR / mod_rel).with_suffix(".py")
                mod_pkg = ROOT_DIR / mod_rel / "__init__.py"
                if not (mod_py.exists() or mod_pkg.exists()):
                    missing.append(f"{rel_str}:{node.lineno} imports missing {node.module}")
                    continue
                # If importing from a package __init__.py (not a .py module), check submodule fallback
                if mod_pkg.exists() and not mod_py.exists():
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        sub_rel = ROOT_DIR / mod_rel / alias.name
                        if (
                            sub_rel.with_suffix(".py").exists()
                            or (sub_rel / "__init__.py").exists()
                        ):
                            continue
                        if not _package_exports_symbol(mod_pkg, alias.name):
                            missing.append(
                                f"{rel_str}:{node.lineno} imports missing {node.module}.{alias.name}"
                            )
    assert not missing, "Unshipped hbllm.* imports inside hbllm/:\n" + "\n".join(missing)
