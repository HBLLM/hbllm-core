"""Static AST-based Control-Flow Auditor for Cognitive Reasoning Operators.

Audits source code at the Abstract Syntax Tree (AST) level to ensure:
1. No hardcoded domain branching (e.g. `if object == 'mepo': ...`) exists in reasoning kernels.
2. No task-specific ground truth tokens or labels are baked into control flow or constants.
3. Cognitive operators remain strictly domain-general and invariant to task entity names.
"""

from __future__ import annotations

import ast
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from hbllm.experiment.leakage_audit import PROHIBITED_TASK_PRELOADS, LeakageAuditReport

logger = logging.getLogger(__name__)


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
            f"[{self.filename}:{self.line_number}] Hardcoded task token '{self.prohibited_token}' "
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

        for node in ast.walk(tree):
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

    def _inspect_node(self, node: ast.AST) -> tuple[str | None, str, str]:
        """Inspect an AST node for prohibited literals in comparisons and constants."""
        # 1. String constants
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            val_lower = node.value.lower().strip()
            for token in self.prohibited_tokens:
                if token.lower() in val_lower:
                    return token, "Constant", str(node.value)

        # 2. Compare nodes (e.g. if x == "mepo")
        if isinstance(node, ast.Compare):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    val_lower = comparator.value.lower().strip()
                    for token in self.prohibited_tokens:
                        if token.lower() in val_lower:
                            return token, "Comparison", f"== '{comparator.value}'"

        return None, "", ""

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
