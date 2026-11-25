"""Architecture tests enforcing State Ownership and Fail-Closed TransactionManager invariants (H-7)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from hbllm.hcir.kernel.governance.constitutional_verifier import ConstitutionalVerifier
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.workspace import HCIRWorkspaceState

ROOT_DIR = Path(__file__).resolve().parents[3]
HBLLM_DIR = ROOT_DIR / "hbllm"
HCIR_DIR = HBLLM_DIR / "hcir"

_RAW_MUTATION_METHODS = frozenset(
    {"add_node", "upsert_node", "add_edge", "remove_node", "remove_edge"}
)
_SANCTIONED_PREFIXES = (
    "hbllm/hcir/graph.py",
    "hbllm/hcir/native_adapter.py",
    "hbllm/hcir/workspace.py",
    "hbllm/hcir/snapshot.py",
    "hbllm/hcir/stores/",
    "hbllm/hcir/algorithms/",
    "hbllm/hcir/reasoning/operators/",
    "hbllm/hcir/grounding/concepts/relational_graph_matcher.py",
)
_WORKSPACE_ABSENT_FALLBACK_ALLOWLIST: frozenset[str] = frozenset()
MAX_WORKSPACE_ABSENT_FALLBACK_CALLS = 0


def test_transaction_manager_fails_closed_by_default() -> None:
    """Default TransactionManager attaches ConstitutionalVerifier; empty stages require opt-in."""
    ws = HCIRWorkspaceState()
    tm_default = TransactionManager(ws)
    assert tm_default.unverified is False
    assert any(isinstance(s, ConstitutionalVerifier) for s in tm_default._verification_stages)

    ws_empty = HCIRWorkspaceState()
    with pytest.raises(ValueError, match="verification_stages"):
        TransactionManager(ws_empty, verification_stages=[])

    tm_opt_in = TransactionManager(
        ws_empty,
        verification_stages=[],
        unsafe_allow_unverified=True,
    )
    assert tm_opt_in.unverified is True


def test_no_unsafe_unverified_in_production_hbllm() -> None:
    """No production module under ``hbllm/`` may opt out via unsafe_allow_unverified=True."""
    hits: list[str] = []
    for path in sorted(HBLLM_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for kw in node.keywords:
                if (
                    kw.arg == "unsafe_allow_unverified"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                ):
                    hits.append(f"{path.relative_to(ROOT_DIR)}:{node.lineno}")
    assert not hits, f"Found unsafe_allow_unverified=True calls in production code: {hits}"


def test_raw_graph_mutation_ratchet_in_hcir() -> None:
    """Raw graph mutation primitives outside kernel/stores/scratch are disallowed or ratcheted."""
    unexpected: list[str] = []
    fallback_calls: list[str] = []

    for path in sorted(HCIR_DIR.rglob("*.py")):
        rel = str(path.relative_to(ROOT_DIR))
        if any(rel == p or rel.startswith(p) for p in _SANCTIONED_PREFIXES):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _RAW_MUTATION_METHODS
            ):
                site = f"{rel}:{node.lineno}:.{node.func.attr}()"
                if rel in _WORKSPACE_ABSENT_FALLBACK_ALLOWLIST:
                    fallback_calls.append(site)
                else:
                    unexpected.append(site)

    assert not unexpected, "Unexpected raw graph mutations inside hbllm/hcir:\n" + "\n".join(
        unexpected
    )
    assert len(fallback_calls) <= MAX_WORKSPACE_ABSENT_FALLBACK_CALLS, (
        f"Workspace-absent fallback raw calls ({len(fallback_calls)}) exceed ratchet "
        f"{MAX_WORKSPACE_ABSENT_FALLBACK_CALLS}:\n" + "\n".join(fallback_calls)
    )
