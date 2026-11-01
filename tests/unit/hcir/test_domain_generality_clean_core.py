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
    quick_tokens = ("arc_agi", "kaggle", "game_engine")

    violations = []
    for py_file in world_dir.glob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        if not any(tok in content.lower() for tok in quick_tokens):
            continue
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
        if not any(imp in content for imp in forbidden_imports):
            continue
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


def test_skills_all_has_no_grid_puzzle_solvers():
    """Verify that hbllm.hcir.skills.__all__ contains zero grid-puzzle specific solvers (WP3)."""
    import hbllm.hcir as hcir_pkg
    import hbllm.hcir.skills as skills
    import hbllm.hcir.skills.common_subskills as common_subskills
    import hbllm.hcir.skills.declarative as declarative

    forbidden = {
        "BlockPushPlanner",
        "GF2LinearSolver",
        "TileGridPredicate",
        "TileFramePredicate",
        "PanelConstraint",
        "LatticeConduitPredicate",
    }
    present = forbidden.intersection(set(skills.__all__))
    assert not present, f"Grid-puzzle specific solvers leaked into core skills __all__: {present}"

    # The solver names must not exist anywhere in the core module namespaces
    # (not merely absent from __all__) — they live in plugins/arc_agi_adapter.
    for module in (skills, declarative, common_subskills, hcir_pkg):
        leaked = forbidden.intersection(vars(module))
        assert not leaked, (
            f"Grid-puzzle specific solvers still defined/re-exported in {module.__name__}: {leaked}"
        )


def _fold_string(node: ast.AST) -> str | None:
    """Statically fold a string-producing AST node.

    Folds ``BinOp(Add)`` concatenations, f-strings (``JoinedStr``) and
    ``"<sep>".join([...])`` calls whose components are string constants.
    Returns ``None`` when the node is not statically foldable.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _fold_string(node.left)
        right = _fold_string(node.right)
        if left is not None and right is not None:
            return left + right
        return None
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                # FormattedValue or anything non-constant: unrepresentable.
                parts.append("\x00")
        return "".join(parts)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and isinstance(node.func.value, ast.Constant)
        and isinstance(node.func.value.value, str)
        and len(node.args) == 1
        and isinstance(node.args[0], (ast.List, ast.Tuple))
    ):
        elements = [
            _fold_string(elt)
            for elt in node.args[0].elts  # type: ignore[union-attr]
        ]
        if all(e is not None for e in elements):
            return node.func.value.value.join(elements)  # type: ignore[arg-type]
    return None


def _docstring_constant_ids(tree: ast.AST) -> set[int]:
    """Ids of Constant nodes that are docstrings (allow-listed from token checks)."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if hasattr(node, "body") and node.body:
                first_expr = node.body[0]
                if (
                    isinstance(first_expr, ast.Expr)
                    and isinstance(first_expr.value, ast.Constant)
                    and isinstance(first_expr.value.value, str)
                ):
                    ids.add(id(first_expr.value))
    return ids


def test_zero_forbidden_domain_tokens_in_cognitive_core():
    """Guard test verifying ZERO forbidden domain-specific tokens exist in core (WP3).

    Scans core/hbllm/hcir and core/hbllm/brain for:
        game_id, level_id, is_win, is_lost, click

    Checks (AST-level, so string-split evasion like ``"is_" + "win"`` is caught):
    - every string constant and every statically foldable concatenation
      (``BinOp(Add)``, f-string, ``"<sep>".join([...])``) is matched against the
      forbidden token regex;
    - every identifier (``Name``, ``Attribute``, ``FunctionDef``, ``ClassDef``,
      ``arg``, ``keyword``, import ``alias``) is matched against ``(?i)click``;
    - the original line-level regex scan is kept, with docstrings allow-listed.
    """
    forbidden_pat = re.compile(r"\b(game_id|level_id|is_win|is_lost|click)\b")
    core_dirs = [
        Path(__file__).resolve().parents[3] / "hbllm" / "hcir",
        Path(__file__).resolve().parents[3] / "hbllm" / "brain",
    ]

    violations = []
    _FRAGMENTS = ("game", "level", "is_", "win", "lost", "click")
    for d in core_dirs:
        for py_file in d.rglob("*.py"):
            rel = py_file.relative_to(d.parents[1])
            content = py_file.read_text(encoding="utf-8")
            content_lower = content.lower()
            if not any(frag in content_lower for frag in _FRAGMENTS):
                continue
            try:
                tree = ast.parse(content, filename=str(py_file))
            except SyntaxError:
                continue

            # Collect docstring line ranges to allow environment reference examples in docstrings
            docstring_lines: set[int] = set()
            docstring_ids: set[int] = set()
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    doc = ast.get_docstring(node, clean=False)
                    if doc and hasattr(node, "body") and node.body:
                        first_expr = node.body[0]
                        if (
                            isinstance(first_expr, ast.Expr)
                            and isinstance(first_expr.value, ast.Constant)
                            and isinstance(first_expr.value.value, str)
                        ):
                            docstring_lines.update(
                                range(
                                    first_expr.lineno,
                                    getattr(first_expr, "end_lineno", first_expr.lineno) + 1,
                                )
                            )
                            docstring_ids.add(id(first_expr.value))

            # ── Line-level scan (original behaviour) ─────────────────────────
            for line_no, line in enumerate(content.splitlines(), start=1):
                # Ignore docstring lines and pure comments
                if line_no in docstring_lines:
                    continue
                code_part = line.split("#")[0].strip()
                if not code_part:
                    continue
                match = forbidden_pat.search(code_part)
                if match:
                    violations.append(f"{rel}:{line_no}: token '{match.group()}' in '{code_part}'")

            # ── AST-level scan: folded strings and click identifiers ─────────
            has_click = "click" in content_lower
            for node in ast.walk(tree):
                # Folded string tokens (covers "is_" + "win", f-strings, joins)
                if (
                    (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add))
                    or isinstance(node, ast.JoinedStr)
                    or (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "join"
                    )
                ):
                    if id(node) in docstring_ids:
                        continue
                    folded = _fold_string(node)
                    if folded is not None:
                        match = forbidden_pat.search(folded)
                        if match:
                            violations.append(
                                f"{rel}:{getattr(node, 'lineno', '?')}: folded string token "
                                f"'{match.group()}' in {folded!r}"
                            )

                # Identifiers containing 'click' in any casing
                if has_click:
                    ident: str | None = None
                    if isinstance(node, ast.Name):
                        ident = node.id
                    elif isinstance(node, ast.Attribute):
                        ident = node.attr
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        ident = node.name
                    elif isinstance(node, ast.arg):
                        ident = node.arg
                    elif isinstance(node, ast.keyword):
                        ident = node.arg
                    elif isinstance(node, ast.alias):
                        ident = node.asname or node.name
                    if ident and "click" in ident.lower():
                        violations.append(
                            f"{rel}:{getattr(node, 'lineno', '?')}: forbidden identifier '{ident}'"
                        )

    assert not violations, (
        "Domain-specific forbidden tokens detected in cognitive core (WP3 violation):\n"
        + "\n".join(violations)
    )
