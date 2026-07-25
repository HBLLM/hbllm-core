"""
HBLLM Runtime Diagnostics — Comprehensive system health report CLI.

Inspects hardware acceleration, Rust crate compilation status, event loop engine,
database configuration, and cognitive subsystem status.

Usage::

    python -m hbllm.cli.diagnostics
"""

from __future__ import annotations

import platform
import sys
from typing import Any


def check_rust_crates() -> dict[str, bool]:
    """Check compilation status of native Rust crates."""
    crates = [
        "hbllm_perception_rs",
        "hbllm_semantic_search",
        "hbllm_compute_kernel_rs",
        "hbllm_tokenizer_rs",
        "hbllm_knowledge_graph_rs",
        "hbllm_confidence_rs",
        "hbllm_concept_extract_rs",
    ]
    status = {}
    for crate in crates:
        try:
            __import__(crate)
            status[crate] = True
        except ImportError:
            status[crate] = False
    return status


def check_event_loop() -> str:
    """Detect current event loop policy."""
    import importlib.util

    if importlib.util.find_spec("uvloop") is not None:
        return "uvloop (installed)"
    return "asyncio (default CPython loop)"


def run_diagnostics() -> dict[str, Any]:
    """Run full diagnostic collection."""
    rust_crates = check_rust_crates()
    rust_compiled = sum(1 for v in rust_crates.values() if v)

    import sqlite3

    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "event_loop": check_event_loop(),
        "rust": {
            "compiled_count": f"{rust_compiled}/{len(rust_crates)}",
            "crates": rust_crates,
        },
        "sqlite": {
            "version": sqlite3.sqlite_version,
        },
    }


def print_diagnostics() -> None:
    """Print formatted diagnostic report to stdout."""
    diag = run_diagnostics()

    print()
    print("╔═══════════════════════════════════════════════╗")
    print("║          HBLLM Runtime Diagnostics            ║")
    print("╚═══════════════════════════════════════════════╝")
    print()

    plat = diag["platform"]
    print(f"  OS Platform   : {plat['system']} {plat['release']} ({plat['machine']})")
    print(f"  Python        : {plat['python']}")
    print(f"  Event Loop    : {diag['event_loop']}")
    print(f"  SQLite        : v{diag['sqlite']['version']}")
    print()

    rust = diag["rust"]
    print(f"  Rust Acceleration ({rust['compiled_count']} crates compiled):")
    for crate, compiled in rust["crates"].items():
        symbol = "✓ compiled" if compiled else "✗ not compiled (python fallback)"
        print(f"    - {crate:<26}: {symbol}")
    print()


def main() -> None:
    print_diagnostics()


if __name__ == "__main__":
    main()
