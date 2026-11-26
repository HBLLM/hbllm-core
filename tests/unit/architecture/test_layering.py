"""Architectural layering invariants test.

Enforces strict one-way dependency boundaries across HBLLM layers:
1. Legacy ``hbllm.brain`` and ``experiments/legacy_brain`` packages are retired.
2. Zero imports of ``hbllm.brain`` remain across core, plugins, benchmarks, or tests.
3. Core cognitive OS must not import from external Plugins (core ↛ plugins).
4. Governance policies canonical location in HCIR.
5. Domain-agnostic lifecycle boundaries and context identification.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _get_imports(py_path: Path) -> list[str]:
    """Extract all imported module names from a Python source file using AST."""
    imports: list[str] = []
    try:
        content = py_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(py_path))
    except Exception:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def _find_violations(base_dir: Path, forbidden_prefixes: tuple[str, ...]) -> list[str]:
    """Find all files in base_dir importing modules matching forbidden_prefixes."""
    violations: list[str] = []
    for py_file in base_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        imported = _get_imports(py_file)
        for imp in imported:
            if any(imp == prefix or imp.startswith(f"{prefix}.") for prefix in forbidden_prefixes):
                violations.append(f"{py_file.relative_to(base_dir.parent.parent)}: imports {imp}")
    return violations


def test_legacy_brain_directory_retired() -> None:
    """Invariant: Legacy Gen-1 hbllm/brain and experiments/legacy_brain are completely retired."""
    root = Path(__file__).resolve().parents[3]
    assert not (root / "hbllm" / "brain").exists(), "hbllm/brain must be retired"
    assert not (root / "experiments" / "legacy_brain").exists(), (
        "experiments/legacy_brain must be retired"
    )


def test_zero_hbllm_brain_imports_across_codebase() -> None:
    """Invariant: Zero active code across hbllm, plugins, benchmarks, scripts, experiments, examples, or tests imports hbllm.brain."""
    root = Path(__file__).resolve().parents[3]
    violations: list[str] = []
    for sub in ("hbllm", "plugins", "benchmarks", "scripts", "experiments", "examples", "tests"):
        base = root / sub
        if not base.is_dir():
            continue
        for py_file in base.rglob("*.py"):
            if "__pycache__" in py_file.parts:
                continue
            if py_file.name == "test_no_duplicate_faculties.py":
                continue
            for imp in _get_imports(py_file):
                if imp == "hbllm.brain" or imp.startswith("hbllm.brain."):
                    violations.append(f"{py_file.relative_to(root)}: imports {imp}")
    assert not violations, "Legacy hbllm.brain imports detected:\n" + "\n".join(violations)


def test_core_does_not_import_plugins() -> None:
    """Invariant: Core system (hbllm) must never depend on external plugin adapters."""
    root = Path(__file__).resolve().parents[3]
    hbllm_dir = root / "hbllm"
    assert hbllm_dir.is_dir()

    violations = _find_violations(hbllm_dir, ("plugins",))
    assert not violations, "Core subsystem violates architectural layering:\n" + "\n".join(
        violations
    )


def test_governance_policies_canonical_location() -> None:
    """Governance policies are canonically located inside HCIR Kernel."""
    from hbllm.hcir.kernel.governance.policies.owner_rules import OwnerRuleStore
    from hbllm.hcir.kernel.governance.policies.policy_engine import PolicyEngine

    assert OwnerRuleStore is not None
    assert PolicyEngine is not None


def test_vocabulary_domain_agnostic_boundaries() -> None:
    """Lifecycle boundaries and context identifiers are domain-agnostic."""
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
    from hbllm.hcir.world.percept import EpisodeBoundaryKind

    # Boundary kinds support both domain-agnostic and legacy terms
    assert hasattr(EpisodeBoundaryKind, "GOAL_REACHED")
    assert hasattr(EpisodeBoundaryKind, "LEVEL_COMPLETE")

    engine = AutonomousEpistemicEngine()
    assert hasattr(engine, "current_context_id")
    assert hasattr(engine, "current_level")

    engine.current_context_id = "42"
    assert engine.current_context_id == "42"
    assert engine.current_level == 42


def test_sqlite_connections_confined_to_infrastructure() -> None:
    """Invariant: sqlite3.connect and aiosqlite.connect are strictly confined to infrastructure.

    Permitted directories:
    - hbllm/hcir/stores/
    - hbllm/persistence/
    - hbllm/security/
    All other cognitive and operational modules must route through persistence infrastructure.
    """
    root = Path(__file__).resolve().parents[3]
    hbllm_dir = root / "hbllm"
    assert hbllm_dir.is_dir()

    allowed_prefixes = (
        str((hbllm_dir / "hcir" / "stores").resolve()),
        str((hbllm_dir / "persistence").resolve()),
        str((hbllm_dir / "security").resolve()),
    )

    violations: list[str] = []
    for py_file in hbllm_dir.rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue
        file_resolved = str(py_file.resolve())
        if any(file_resolved.startswith(p) for p in allowed_prefixes):
            continue

        try:
            content = py_file.read_text(encoding="utf-8")
            tree = ast.parse(content, filename=str(py_file))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "connect":
                    if isinstance(node.func.value, ast.Name) and node.func.value.id in (
                        "sqlite3",
                        "aiosqlite",
                    ):
                        violations.append(
                            f"{py_file.relative_to(hbllm_dir.parent)}:{node.lineno} calls {node.func.value.id}.connect"
                        )

    assert not violations, (
        "Direct SQLite connections detected outside infrastructure boundaries:\n"
        + "\n".join(violations)
    )
