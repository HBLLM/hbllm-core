"""Static AST-based Control-Flow Auditor for Cognitive Reasoning Operators.

Audits source code at the Abstract Syntax Tree (AST) level to ensure:
1. No hardcoded domain branching (e.g. `if object == 'mepo': ...`) exists in reasoning kernels.
2. No task-specific ground truth tokens or labels are baked into control flow or constants.
3. Cognitive operators remain strictly domain-general and invariant to task entity names.
4. Detects hardcoded entity ID pattern comparisons (e.g. `if base == 'obj_support': ...`).
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from hbllm.experiment.leakage_audit import PROHIBITED_TASK_PRELOADS, LeakageAuditReport

logger = logging.getLogger(__name__)

# Variable names commonly used to refer to entity IDs or objects in reasoning routines
ENTITY_VAR_NAMES: frozenset[str] = frozenset(
    {
        "base",
        "base_id",
        "item",
        "item_id",
        "obj",
        "obj_id",
        "obj_name",
        "entity",
        "entity_id",
        "target",
        "target_id",
        "token",
        "concept_name",
    }
)

# Regex patterns that match task-specific scenario entity identifiers rather than generic types
HARDCODED_ENTITY_PATTERN = re.compile(
    r"^(obj_support|support_flat|support_curved|support_soft|support_stable|support_dome|"
    r"box_\d+|block_\d+|e\d+|tr_\d+|ts_\d+|target_\d+|obj_\d+|item_\d+|placed_block|"
    r"mepo|dax|hollow_cube|magnetic_cylinder|fragile_glass|nucleus_electron|star_comet)$",
    re.IGNORECASE,
)


@dataclass
class ASTViolation:
    """Detailed AST violation indicating hardcoded control-flow leakage."""

    filename: str
    line_number: int
    prohibited_token: str
    node_type: str
    snippet: str

    def format_message(self) -> str:
        return (
            f"[{self.filename}:{self.line_number}] Prohibited hardcoded token/pattern '{self.prohibited_token}' "
            f"found in {self.node_type}: '{self.snippet}'"
        )


class ASTLeakageAuditor:
    """Audits Python source files using AST parsing for hardcoded task-specific heuristics."""

    def __init__(self, prohibited_tokens: set[str] | None = None) -> None:
        self.prohibited_tokens: set[str] = (
            set(prohibited_tokens)
            if prohibited_tokens is not None
            else set(PROHIBITED_TASK_PRELOADS)
        )

    def audit_source(self, source_code: str, filename: str = "<string>") -> list[str]:
        """Audit a Python source string via AST traversal."""
        violations: list[str] = []
        try:
            tree = ast.parse(source_code, filename=filename)
        except SyntaxError as e:
            violations.append(f"[{filename}] SyntaxError during AST parsing: {e}")
            return violations

        docstrings = self._collect_docstrings(tree)

        for node in ast.walk(tree):
            if node in docstrings:
                continue
            found_token, node_type, snippet = self._inspect_node(node)
            if found_token:
                lineno = getattr(node, "lineno", 0)
                v = ASTViolation(
                    filename=filename,
                    line_number=lineno,
                    prohibited_token=found_token,
                    node_type=node_type,
                    snippet=snippet,
                )
                violations.append(v.format_message())

        return violations

    def _collect_docstrings(self, tree: ast.AST) -> set[ast.AST]:
        """Collect all docstring Constant/Expr nodes across modules, classes, and functions."""
        docstrings: set[ast.AST] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.body and isinstance(node.body[0], ast.Expr):
                    expr_val = node.body[0].value
                    if isinstance(expr_val, ast.Constant) and isinstance(expr_val.value, str):
                        docstrings.add(expr_val)
                        docstrings.add(node.body[0])
        return docstrings

    def audit_file(self, file_path: str | Path) -> list[str]:
        """Audit a single Python file for AST leakage."""
        path = Path(file_path)
        if not path.exists():
            return [f"File not found: {path}"]
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            return [f"Error reading {path}: {e}"]
        return self.audit_source(content, filename=str(path))

    def audit_directory(
        self, dir_path: str | Path, pattern: str = "*.py", exclude_tests: bool = True
    ) -> list[str]:
        """Recursively audit all Python files in a directory."""
        path = Path(dir_path)
        all_violations: list[str] = []
        for file in path.rglob(pattern):
            if exclude_tests and ("test" in file.name or "tests" in file.parts):
                continue
            all_violations.extend(self.audit_file(file))
        return all_violations

    def audit_reasoning_codebase(self, root_dir: str | Path) -> list[str]:
        """Audit only cognitive reasoning kernels and cohorts, excluding task scenario definitions."""
        root = Path(root_dir)
        reasoning_targets: list[Path] = []

        # 1. Target hbllm/brain cognitive engines
        brain_dir = root / "hbllm" / "brain"
        if brain_dir.is_dir():
            for f in brain_dir.rglob("*.py"):
                if "tests" not in f.parts and not f.name.startswith("test_"):
                    reasoning_targets.append(f)

        # 2. Target experiment cohorts (e.g. HBLLMCoreCohort)
        cohorts_file = root / "hbllm" / "experiment" / "cohorts.py"
        if cohorts_file.is_file():
            reasoning_targets.append(cohorts_file)

        # Audit collected targets
        all_violations: list[str] = []
        for target in reasoning_targets:
            all_violations.extend(self.audit_file(target))
        return all_violations

    def _inspect_node(self, node: ast.AST) -> tuple[str | None, str, str]:
        """Inspect an AST node for prohibited literals in comparisons and constants."""
        # 1. String constants matching prohibited tokens
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val_lower = node.value.lower().strip()
            for token in self.prohibited_tokens:
                if token.lower() in val_lower:
                    return token, "Constant", str(node.value)

        # 2. Compare nodes (e.g. if x == "mepo" or if base == "obj_support")
        if isinstance(node, ast.Compare):
            left_name = self._get_name_or_attr(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    comp_val = comparator.value.strip()
                    comp_lower = comp_val.lower()

                    # Check for prohibited task preloads
                    for token in self.prohibited_tokens:
                        if token.lower() in comp_lower:
                            return token, "Comparison", f"== '{comparator.value}'"

                    # Check for hardcoded entity comparisons (e.g. if base == "obj_support")
                    if left_name and (
                        left_name.lower() in ENTITY_VAR_NAMES
                        or any(
                            k in left_name.lower()
                            for k in ("entity", "obj", "base", "item", "target")
                        )
                    ):
                        if HARDCODED_ENTITY_PATTERN.match(comp_val):
                            return comp_val, "EntityComparison", f"{left_name} == '{comp_val}'"

                # Check reverse comparison: "obj_support" == base
                if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                    left_val = node.left.value.strip()
                    comp_name = self._get_name_or_attr(comparator)
                    if comp_name and (
                        comp_name.lower() in ENTITY_VAR_NAMES
                        or any(
                            k in comp_name.lower()
                            for k in ("entity", "obj", "base", "item", "target")
                        )
                    ):
                        if HARDCODED_ENTITY_PATTERN.match(left_val):
                            return left_val, "EntityComparison", f"'{left_val}' == {comp_name}"

        return None, "", ""

    def _get_name_or_attr(self, node: ast.AST) -> str | None:
        """Extract a readable identifier string from a Name, Attribute, or Subscript."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Subscript):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                return node.slice.value
        return None

    def run_full_static_audit(self, target_paths: list[str | Path]) -> LeakageAuditReport:
        """Run complete static AST audit over a set of files or directories."""
        all_violations: list[str] = []
        audited_files: list[str] = []

        for p in target_paths:
            path = Path(p)
            if path.is_dir():
                for f in path.rglob("*.py"):
                    if "tests" not in f.parts and not f.name.startswith("test_"):
                        audited_files.append(str(f))
                        all_violations.extend(self.audit_file(f))
            elif path.is_file():
                audited_files.append(str(path))
                all_violations.extend(self.audit_file(path))

        combined_hash = hashlib.sha256("".join(sorted(audited_files)).encode("utf-8")).hexdigest()

        is_clean = len(all_violations) == 0
        if not is_clean:
            logger.warning(
                "AST static audit found %d violations: %s", len(all_violations), all_violations
            )

        return LeakageAuditReport(
            is_clean=is_clean,
            initial_knowledge_hash=combined_hash,
            violations=all_violations,
            audited_cohorts=audited_files,
        )
