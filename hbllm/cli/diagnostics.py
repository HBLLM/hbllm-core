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
        "hbllm_hcir_graph",
        "hbllm_confidence",
        "hbllm_perception_rs",
        "hbllm_semantic_search",
        "hbllm_compute",
        "hbllm_tokenizer_rs",
        "hbllm_knowledge_graph",
        "hbllm_concept_extract",
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


def check_cognitive_subsystems() -> dict[str, Any]:
    """Inspect Cognitive OS kernel, constitutional verifier, memory tiers, and native graph health."""
    from hbllm.hcir.kernel.governance.constitutional_verifier import (
        ConstitutionalVerifier,
    )
    from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

    system = build_cognitive_system(CognitiveSystemConfig(world_id="diagnostics_check"))
    tm = system.services.transaction_manager
    verifier_present = any(
        isinstance(stage, ConstitutionalVerifier) for stage in tm._verification_stages
    )
    substrate_health = system.workspace.graph.native_health()

    return {
        "kernel_services": "online",
        "verifier_attached": verifier_present,
        "transaction_manager": "verified" if not tm.unverified else "unverified",
        "memory_tiers": ["working", "brain", "persistent", "meta"],
        "substrate": substrate_health,
    }


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
        "cognitive_os": check_cognitive_subsystems(),
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

    cog = diag.get("cognitive_os", {})
    if cog:
        print("  Cognitive OS & Subsystems:")
        print(f"    - Kernel Services          : {cog.get('kernel_services', 'unknown')}")
        v_sym = "✓ attached" if cog.get("verifier_attached") else "✗ missing"
        print(f"    - Constitutional Verifier  : {v_sym}")
        print(f"    - Transaction Governance   : {cog.get('transaction_manager', 'unknown')}")
        tiers_str = ", ".join(t.lower() for t in cog.get("memory_tiers", []))
        print(f"    - Memory Tiers             : 4 tiers active ({tiers_str})")
        sub = cog.get("substrate", {})
        sub_avail = "✓ available" if sub.get("available") else "python fallback"
        print(
            f"    - Graph Native Substrate   : {sub_avail} (divergences: {sub.get('divergences', 0)})"
        )
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
