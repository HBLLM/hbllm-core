"""Ban raw workspace.add_node/upsert_node/remove_node/add_edge/remove_edge calls.

External callers must go through ``commit_operations`` / TransactionManager
(AGENTS.md Pattern A). Kernel-internal uses in transaction_manager.py and the
deprecated wrappers/docstrings in workspace.py are exempt.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCAN_DIRS = ["hbllm", "plugins", "kaggle_submission"]
BANNED = ("add_node", "upsert_node", "remove_node", "add_edge", "remove_edge")
EXEMPT_FILES = {
    ROOT / "hbllm/hcir/kernel/transaction_manager.py",
}
# deprecated public wrappers inside workspace.py itself
EXEMPT_DEFS = {"add_node", "upsert_node", "remove_node", "add_edge", "remove_edge"}
RECEIVER_PAT = re.compile(r"workspace|ws", re.IGNORECASE)


def _receiver_name(call: ast.Call) -> str:
    fn = call.func
    if isinstance(fn, ast.Attribute):
        v = fn.value
        while isinstance(v, ast.Attribute):
            v = v.value
        if isinstance(v, ast.Name):
            parts = []
            n = call.func.value
            while isinstance(n, ast.Attribute):
                parts.append(n.attr)
                n = n.value
            if isinstance(n, ast.Name):
                parts.append(n.id)
            return ".".join(reversed(parts)) or v.id
    return ""


def test_no_raw_workspace_graph_primitives() -> None:
    violations: list[str] = []
    for d in SCAN_DIRS:
        for py in (ROOT / d).rglob("*.py"):
            if py in EXEMPT_FILES or "__pycache__" in str(py):
                continue
            content = py.read_text()
            if not any(b in content for b in BANNED):
                continue
            try:
                tree = ast.parse(content, filename=str(py))
            except SyntaxError:
                continue

            parents: dict[int, ast.AST] = {}
            if py.name == "workspace.py":
                for node in ast.walk(tree):
                    for child in ast.iter_child_nodes(node):
                        parents[id(child)] = node

            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in BANNED
                ):
                    continue
                # allow the deprecated wrapper bodies in workspace.py
                if py.name == "workspace.py":
                    anc = node
                    inside_wrapper = False
                    while id(anc) in parents:
                        anc = parents[id(anc)]
                        if isinstance(anc, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if anc.name in EXEMPT_DEFS:
                                inside_wrapper = True
                            break
                    if inside_wrapper:
                        continue
                recv = _receiver_name(node)
                if RECEIVER_PAT.search(recv):
                    violations.append(
                        f"{py.relative_to(ROOT)}:{node.lineno}: {recv}.{node.func.attr}(...)"
                    )
    assert not violations, (
        "Raw workspace graph primitives bypass the transaction path:\n" + "\n".join(violations)
    )
