"""Architecture ratchet tests: prevent structural debt growth in the cognitive core.

Enforces R6 from ``reports/audit/2026-11-15_phase2_generalization_review.md`` §5:
counts of dynamic attribute introspection (``hasattr`` / 3-arg ``getattr``), broad silent
``except: pass`` handlers, and direct SQLite connection sites may only decrease over time.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
HBLLM_DIR = ROOT_DIR / "hbllm"
WORLD_DIR = HBLLM_DIR / "hcir" / "world"
HCIR_DIR = HBLLM_DIR / "hcir"

# R6/R7/R8 ratchets (tightened monotonically downward after R8 bounded state & latency)
MAX_WORLD_HASATTR = 115
MAX_WORLD_GETATTR3 = 357
MAX_HCIR_BROAD_SILENT_EXCEPT = 0
MAX_HBLLM_BROAD_SILENT_EXCEPT = 46
MAX_SQLITE_CONNECT_CALLS = 22
MAX_SQLITE_CONNECT_FILES = 8


def _iter_python_files(base: Path) -> list[Path]:
    return [p for p in sorted(base.rglob("*.py")) if "__pycache__" not in p.parts]


def _is_broad_silent_handler(node: ast.ExceptHandler) -> bool:
    if len(node.body) != 1 or not isinstance(node.body[0], ast.Pass):
        return False
    if node.type is None:
        return True
    return isinstance(node.type, ast.Name) and node.type.id in ("Exception", "BaseException")


def _collect_ast_debt() -> dict[str, int]:
    counts = {
        "world_hasattr": 0,
        "world_getattr3": 0,
        "hcir_broad_silent": 0,
        "hbllm_broad_silent": 0,
        "sqlite_calls": 0,
        "sqlite_files": 0,
    }
    for path in _iter_python_files(HBLLM_DIR):
        src = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src, filename=str(path))
        is_world = path.is_relative_to(WORLD_DIR)
        is_hcir = path.is_relative_to(HCIR_DIR)
        file_has_sqlite = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if is_world and node.func.id == "hasattr":
                    counts["world_hasattr"] += 1
                elif is_world and node.func.id == "getattr" and len(node.args) >= 3:
                    counts["world_getattr3"] += 1
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    node.func.attr == "connect"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in ("sqlite3", "aiosqlite")
                ):
                    counts["sqlite_calls"] += 1
                    file_has_sqlite = True
            elif isinstance(node, ast.ExceptHandler) and _is_broad_silent_handler(node):
                if is_hcir:
                    counts["hcir_broad_silent"] += 1
                counts["hbllm_broad_silent"] += 1
        if file_has_sqlite:
            counts["sqlite_files"] += 1
    return counts


def test_world_dynamic_attribute_ratchet() -> None:
    """Enforce downward ratchet on hasattr and 3-arg getattr inside hbllm/hcir/world/."""
    counts = _collect_ast_debt()
    assert counts["world_hasattr"] <= MAX_WORLD_HASATTR, (
        f"hbllm/hcir/world hasattr count {counts['world_hasattr']} > ratchet {MAX_WORLD_HASATTR}"
    )
    assert counts["world_getattr3"] <= MAX_WORLD_GETATTR3, (
        f"hbllm/hcir/world 3-arg getattr count {counts['world_getattr3']} > ratchet {MAX_WORLD_GETATTR3}"
    )


def test_silent_broad_except_pass_ratchet() -> None:
    """Enforce zero broad silent except:pass in hbllm/hcir/ and ratchet across hbllm/."""
    counts = _collect_ast_debt()
    assert counts["hcir_broad_silent"] <= MAX_HCIR_BROAD_SILENT_EXCEPT, (
        f"hbllm/hcir broad silent except:pass count {counts['hcir_broad_silent']} > "
        f"ratchet {MAX_HCIR_BROAD_SILENT_EXCEPT}"
    )
    assert counts["hbllm_broad_silent"] <= MAX_HBLLM_BROAD_SILENT_EXCEPT, (
        f"hbllm broad silent except:pass count {counts['hbllm_broad_silent']} > "
        f"ratchet {MAX_HBLLM_BROAD_SILENT_EXCEPT}"
    )


def test_sqlite_connect_count_ratchet() -> None:
    """Enforce downward ratchet on direct sqlite3/aiosqlite.connect() call sites."""
    counts = _collect_ast_debt()
    assert counts["sqlite_files"] <= MAX_SQLITE_CONNECT_FILES
    assert counts["sqlite_calls"] <= MAX_SQLITE_CONNECT_CALLS


def test_no_get_sub_attr_or_blacklisted_engine_attributes() -> None:
    """Enforce zero get_sub_attr helpers and zero blacklisted ad-hoc engine attrs in world/."""
    blacklisted = (
        "_feedback_assimilated",
        "_prev_oriented_threats",
        "_consecutive_null_actions",
        "_consecutive_quiescent_actions",
        "_structural_goals_cache",
    )
    violations: list[str] = []
    for path in _iter_python_files(WORLD_DIR):
        src = path.read_text(encoding="utf-8", errors="ignore")
        rel = str(path.relative_to(ROOT_DIR))
        if "get_sub_attr" in src:
            violations.append(f"{rel}: contains get_sub_attr")
        for attr in blacklisted:
            if attr in src:
                violations.append(f"{rel}: contains blacklisted {attr}")
    assert not violations, "Structural debt evasion / blacklisted attrs:\n" + "\n".join(violations)


def test_transition_kb_and_topology_ontology_free_ratchet() -> None:
    """Ensure TransitionKB & topology router contain no hardcoded domain ontology labels or deleted private sets."""
    forbidden_literals = frozenset({"barrier", "walkable", "cargo", "receptacle", "key", "door"})
    ontology_free_files = (
        WORLD_DIR / "causal" / "transition_kb.py",
        WORLD_DIR / "topological_graph_induction.py",
    )
    for target in ontology_free_files:
        src = target.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(target))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value.strip().lower() not in forbidden_literals, (
                    f"Forbidden domain ontology literal {node.value!r} found in {target}"
                )

    deleted_private_re = re.compile(
        r"\._(goal|barrier|walkable|cargo|receptacle|actuator)_features\b"
    )
    for path in _iter_python_files(HBLLM_DIR):
        src = path.read_text(encoding="utf-8", errors="ignore")
        match = deleted_private_re.search(src)
        assert match is None, (
            f"Deleted private ontology attribute {match.group(0)!r} referenced in {path}"
        )
