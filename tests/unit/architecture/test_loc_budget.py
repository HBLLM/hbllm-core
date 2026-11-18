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
MAX_PATH_APPEND_SHIMS = 17

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
