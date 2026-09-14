"""Architectural dependency boundary tests.

Verifies that:
1. Core runtime submodules (continual learning, developmental adapter plugins, graph IR)
   do not transitively load heavy server or enterprise database modules.
2. PEP 562 lazy loading in hbllm.brain and hbllm.memory resolves attributes on demand
   without leaking dependencies at package import time.
3. pyproject.toml maintains clean segregation between core dependencies and optional extras.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

HEAVY_MODULES = [
    "fastapi",
    "uvicorn",
    "redis",
    "qdrant_client",
    "asyncpg",
    "pgvector",
    "cryptography",
]


def _run_in_fresh_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    """Runs a Python snippet in a clean subprocess with current PYTHONPATH."""
    repo_root = str(Path(__file__).resolve().parents[3])
    env = dict(os.environ)
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{repo_root}:{current_pythonpath}" if current_pythonpath else repo_root

    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_developmental_adapter_import_cleanliness() -> None:
    """Importing plugins.developmental_adapter must not load heavy server/db modules."""
    code = f"""
import sys
import plugins.developmental_adapter

heavy_targets = {HEAVY_MODULES!r}
loaded = set(sys.modules.keys())
leaked = [
    target for target in heavy_targets
    if any(k == target or k.startswith(target + ".") for k in loaded)
]
if leaked:
    print(f"LEAKED: {{leaked}}", file=sys.stderr)
    sys.exit(1)
"""
    result = _run_in_fresh_subprocess(code)
    assert result.returncode == 0, (
        f"Developmental adapter leaked heavy dependencies:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_continual_store_import_cleanliness() -> None:
    """Importing hbllm.brain.continual.store must not load heavy server/db modules."""
    code = f"""
import sys
import hbllm.brain.continual.store

heavy_targets = {HEAVY_MODULES!r}
loaded = set(sys.modules.keys())
leaked = [
    target for target in heavy_targets
    if any(k == target or k.startswith(target + ".") for k in loaded)
]
if leaked:
    print(f"LEAKED: {{leaked}}", file=sys.stderr)
    sys.exit(1)
"""
    result = _run_in_fresh_subprocess(code)
    assert result.returncode == 0, (
        f"Continual store leaked heavy dependencies:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_brain_lazy_loading_and_resolution() -> None:
    """Importing hbllm.brain must not eagerly load heavy modules, but must resolve exports on demand."""
    code = f"""
import sys
import hbllm.brain as brain

heavy_targets = {HEAVY_MODULES!r}
loaded = set(sys.modules.keys())
leaked = [
    target for target in heavy_targets
    if any(k == target or k.startswith(target + ".") for k in loaded)
]
assert not leaked, f"Heavy modules loaded on raw brain import: {{leaked}}"

# Verify dir() contains exports
exports = dir(brain)
assert "BrainFactory" in exports
assert "RouterNode" in exports
assert "GoalManager" in exports
assert "PolicyEngine" in exports

# Verify on-demand attribute resolution works
factory = brain.BrainFactory
assert factory is not None
assert hasattr(factory, "create")

# Verify non-existent attribute raises AttributeError
try:
    _ = brain.NonExistentSymbol
    raise AssertionError("Expected AttributeError for NonExistentSymbol")
except AttributeError:
    pass
"""
    result = _run_in_fresh_subprocess(code)
    assert result.returncode == 0, (
        f"Brain lazy loading verification failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_memory_lazy_loading_and_resolution() -> None:
    """Importing hbllm.memory must provide unified interface and resolve MemoryNode on demand."""
    code = """
import sys
import hbllm.memory as mem

# Interface and types must be immediately available
assert hasattr(mem, "UnifiedMemoryInterface")
assert hasattr(mem, "MemoryType")
assert hasattr(mem, "SearchResult")

# dir() contains both eager and lazy exports
exports = dir(mem)
assert "MemoryNode" in exports
assert "ConceptExtractor" in exports
assert "UnifiedMemoryInterface" in exports

# On-demand resolution
node_cls = mem.MemoryNode
assert node_cls is not None
"""
    result = _run_in_fresh_subprocess(code)
    assert result.returncode == 0, (
        f"Memory lazy loading verification failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_pyproject_dependency_discipline() -> None:
    """pyproject.toml must enforce clean boundaries between core dependencies and extras."""
    repo_root = Path(__file__).resolve().parents[3]
    pyproject_path = repo_root / "pyproject.toml"

    with pyproject_path.open("rb") as f:
        config = tomllib.load(f)

    project = config.get("project", {})
    dependencies = project.get("dependencies", [])
    optional_deps = project.get("optional-dependencies", {})

    forbidden_in_core = [
        "fastapi",
        "uvicorn",
        "redis",
        "qdrant-client",
        "asyncpg",
        "pgvector",
        "cryptography",
    ]

    for dep in dependencies:
        for forbidden in forbidden_in_core:
            assert not dep.startswith(forbidden), (
                f"Core dependencies must not contain '{forbidden}'. Found: '{dep}'"
            )

    # Verify optional extras exist
    assert "server" in optional_deps, (
        "Optional dependency extra 'server' must exist in pyproject.toml"
    )
    assert "storage" in optional_deps, (
        "Optional dependency extra 'storage' must exist in pyproject.toml"
    )

    # Verify server contains fastapi and uvicorn
    server_deps = optional_deps["server"]
    assert any("fastapi" in d for d in server_deps)
    assert any("uvicorn" in d for d in server_deps)

    # Verify storage contains vector and db backends
    storage_deps = optional_deps["storage"]
    assert any("qdrant-client" in d for d in storage_deps)
    assert any("asyncpg" in d for d in storage_deps)
