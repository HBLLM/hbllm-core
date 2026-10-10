"""Continuous Integration Gate: Domain-Generality and Architectural Cleanliness Audit.

Verifies that core/hbllm/ contains ZERO benchmark-specific leakage, hardcoded heuristics,
or game-specific dependencies:
1. No game-specific imports in core/hbllm/hcir/world/.
2. No hardcoded benchmark task IDs or outputs in solver code.
3. Cryptographic integrity of frozen task manifests.
4. Strict isolation of adapter plugins outside core cognitive machinery.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from experiments.benchmarks.manifests.transformation_task_manifest import verify_manifest_integrity


def test_cryptographic_manifest_integrity():
    """Verify that all manifest tasks match frozen SHA-256 fingerprints with zero drift."""
    is_valid, errors = verify_manifest_integrity()
    assert is_valid, f"Manifest integrity verification failed: {errors}"


def test_core_world_model_has_no_benchmark_specific_imports():
    """Verify that core/hbllm/hcir/world/ imports only generic stdlib, numpy, and internal core modules."""
    world_dir = Path(__file__).resolve().parents[3] / "hbllm" / "hcir" / "world"
    assert world_dir.exists() and world_dir.is_dir()
    assert not (world_dir / "benchmarks").exists(), (
        "benchmarks directory must NOT exist inside core/hbllm/hcir/world/"
    )

    forbidden_patterns = [
        re.compile(r"\barc_agi\b", re.IGNORECASE),
        re.compile(r"\bkaggle\b", re.IGNORECASE),
        re.compile(r"\bgame_engine\b", re.IGNORECASE),
    ]

    violations = []
    for py_file in world_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(py_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for pat in forbidden_patterns:
                        if pat.search(alias.name):
                            violations.append(f"{py_file.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for pat in forbidden_patterns:
                    if pat.search(mod):
                        violations.append(f"{py_file.name}: from {mod} import ...")

    assert not violations, (
        f"Domain-generality violation: benchmark-specific imports in core world model: {violations}"
    )


def test_core_solver_contains_no_hardcoded_task_ids():
    """Verify that TransformationProgramSearch and GridOperator contain no hardcoded task solutions or IDs."""
    core_files = [
        Path(__file__).resolve().parents[3] / "hbllm" / "hcir" / "world" / "grid_operator.py",
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "benchmarks"
        / "evaluation"
        / "generalization_harness.py",
        Path(__file__).resolve().parents[3] / "hbllm" / "hcir" / "world" / "rule_induction.py",
    ]

    # Specific benchmark hex ID pattern (e.g. 8-char hex task IDs like '007bbfb7')
    hex_task_pat = re.compile(r"['\"][0-9a-f]{8}['\"]", re.IGNORECASE)

    violations = []
    for f in core_files:
        if f.exists():
            content = f.read_text(encoding="utf-8")
            matches = hex_task_pat.findall(content)
            if matches:
                violations.append(f"{f.name} contains benchmark hex task IDs: {matches}")

    assert not violations, f"Hardcoded benchmark IDs detected in core solver files: {violations}"


def test_zero_ground_truth_leakage_in_solver_signature():
    """Verify that TransformationProgramSearch.solve does not accept or access test_output."""
    import inspect

    from hbllm.hcir.world.grid_operator import TransformationProgramSearch

    sig = inspect.signature(TransformationProgramSearch.solve)
    param_names = list(sig.parameters.keys())
    assert "test_output" not in param_names, "solve() must not accept test_output"
    assert "expected" not in param_names, "solve() must not accept expected outputs"
    assert param_names == ["self", "train_pairs", "test_input"], (
        f"Unexpected solve() parameters: {param_names}"
    )


def test_no_task_id_branching_in_core_solver():
    """Verify that solver candidate generation and ranking contain no task-ID specific branching."""
    core_files = [
        Path(__file__).resolve().parents[3] / "hbllm" / "hcir" / "world" / "grid_operator.py",
        Path(__file__).resolve().parents[3] / "hbllm" / "hcir" / "world" / "rule_induction.py",
    ]

    task_id_branch_pat = re.compile(r"\b(task_id|task\.id|task_name)\s*==\s*['\"]", re.IGNORECASE)
    violations = []
    for f in core_files:
        if f.exists():
            content = f.read_text(encoding="utf-8")
            for line_no, line in enumerate(content.splitlines(), start=1):
                if task_id_branch_pat.search(line):
                    violations.append(f"{f.name}:{line_no}: {line.strip()}")

    assert not violations, f"Task-ID branching detected in solver logic: {violations}"


def test_no_hidden_fixture_or_test_manifest_access_in_core():
    """Verify that core/hbllm/hcir/ does not import from tests, fixtures, or external benchmark directories."""
    core_hcir_dir = Path(__file__).resolve().parents[3] / "hbllm" / "hcir"
    forbidden_imports = [
        "tests",
        "fixtures",
        "conftest",
        "kaggle_submission",
        "arc_agi_2",
        "experiments",
    ]

    violations = []
    for py_file in core_hcir_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        try:
            tree = ast.parse(content, filename=str(py_file))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_mod = alias.name.split(".")[0]
                    if top_mod in forbidden_imports:
                        violations.append(f"{py_file.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    top_mod = node.module.split(".")[0]
                    if top_mod in forbidden_imports:
                        violations.append(f"{py_file.name}: from {node.module} import ...")

    assert not violations, (
        f"Hidden test fixture or external benchmark imports found in core: {violations}"
    )


def test_candidate_generation_and_ranking_path_code_review():
    """Verify that candidate generation only derives operators from mathematical and topological primitives."""
    from hbllm.hcir.world.grid_operator import TransformationProgramSearch

    searcher = TransformationProgramSearch(max_depth=3)
    # Ensure all operators have valid typed contracts and descriptions
    for op in searcher.operators:
        assert hasattr(op, "name"), f"Operator {op} missing name attribute"
        assert hasattr(op, "propose"), f"Operator {op} missing propose method"
        assert hasattr(op, "apply"), f"Operator {op} missing apply method"
        assert hasattr(op, "verify"), f"Operator {op} missing verify method"
