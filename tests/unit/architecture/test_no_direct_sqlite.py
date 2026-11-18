"""Architecture invariant test: zero direct SQLite connections outside sanctioned stores.

Enforces Anti-Pattern 2 / architectural invariant:
All database access must be routed through sanctioned storage layers:
- `hbllm/hcir/stores/` (HCIR storage backends)
- `hbllm/persistence/` (Persistence service & profiles)
- `hbllm/security/` (Security, audit, and tenant registry)
"""

from __future__ import annotations

import re
from pathlib import Path


def test_zero_direct_sqlite_connect_outside_sanctioned_stores() -> None:
    """Verify that no module outside sanctioned layers calls sqlite3.connect or aiosqlite.connect."""
    root_dir = Path(__file__).resolve().parents[3] / "hbllm"

    sanctioned_prefixes = (
        "hbllm/hcir/stores/",
        "hbllm/persistence/",
        "hbllm/security/",
    )

    connect_pattern = re.compile(r"\b(sqlite3|aiosqlite)\.connect\(")

    violations: list[str] = []

    for path in root_dir.rglob("*.py"):
        rel_path = path.relative_to(root_dir.parent).as_posix()
        if any(rel_path.startswith(prefix) for prefix in sanctioned_prefixes):
            continue

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            for line_no, line in enumerate(content.splitlines(), start=1):
                if connect_pattern.search(line):
                    violations.append(f"{rel_path}:{line_no}: {line.strip()}")
        except Exception as exc:
            violations.append(f"{rel_path}: failed to read: {exc}")

    assert not violations, (
        f"Found {len(violations)} unsanctioned direct SQLite connect() calls:\n"
        + "\n".join(f"  - {v}" for v in violations)
    )
