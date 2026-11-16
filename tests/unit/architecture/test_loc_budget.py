"""Architecture invariant test: total Python LOC under ``hbllm/`` stays within budget.

Enforces the R5 subtraction gate from ``reports/audit/2026-11-15_phase2_generalization_review.md``
§5: total lines across ``hbllm/**/*.py`` must stay at or below 250,000 LOC.
"""

from __future__ import annotations

from pathlib import Path

LOC_BUDGET = 250_000


def test_hbllm_python_loc_within_budget() -> None:
    """Assert total line count across ``hbllm/**/*.py`` is <= 250,000."""
    root_dir = Path(__file__).resolve().parents[3] / "hbllm"
    total_lines = 0
    file_count = 0

    for path in sorted(root_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        file_count += 1
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            total_lines += sum(1 for _ in handle)

    assert file_count > 0
    assert total_lines <= LOC_BUDGET, (
        f"hbllm/**/*.py total LOC {total_lines:,} exceeds budget {LOC_BUDGET:,} "
        f"across {file_count} files"
    )
