"""
Federation Firewall — Sandbox, Pydantic, and Cognitive Prompt Injection shields.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class FederationSecurityError(Exception):
    """Raised when a federated envelope payload violates security filters."""
    pass


class FederatedFirewall:
    """
    Quarantine filter gateway for untrusted external intent envelopes.

    Implements:
      - AST analysis for embedded code payloads (RCE prevention).
      - Regex-based heuristics scanning for Prompt Injections.
      - Pydantic-like structural key sanitization.
    """

    # Common prompt injection phrases
    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
        re.compile(r"system\s+prompt\s+override", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+a\s+different\s+agent", re.IGNORECASE),
        re.compile(r"new\s+system\s+directive", re.IGNORECASE),
        re.compile(r"bypass\s+(?:all\s+)?tenant\s+guards", re.IGNORECASE),
    ]

    # Dangerous AST nodes that are strictly forbidden in code parameters
    FORBIDDEN_AST_NODES = (
        ast.Import,
        ast.ImportFrom,
        ast.Delete,
        ast.Global,
        ast.Nonlocal,
    )

    # Allowed safe builtins in AST-evaluated strings
    ALLOWED_BUILTINS = {"abs", "all", "any", "bool", "dict", "float", "int", "len", "list", "map", "max", "min", "range", "set", "str", "sum", "tuple"}

    @classmethod
    def sanitize_payload_structure(cls, raw_data: dict[str, Any], schema_keys: set[str]) -> dict[str, Any]:
        """
        Sanitize and filter external payloads.

        Strips out any keys not explicitly defined in the schema to block
        prototype pollution, dictionary injection, or header overflow.
        """
        sanitized = {}
        for key in schema_keys:
            if key in raw_data:
                sanitized[key] = raw_data[key]
        return sanitized

    @classmethod
    def audit_text_field(cls, field_name: str, text: str) -> None:
        """
        Scan a text field for potential Prompt Injection or SQL command boundaries.

        Raises:
            FederationSecurityError: If injection vectors are detected.
        """
        if not text or not isinstance(text, str):
            return

        # 1. Check prompt injection heuristics
        for pattern in cls.INJECTION_PATTERNS:
            if pattern.search(text):
                logger.error("Federation security violation: Prompt Injection pattern caught in '%s'", field_name)
                raise FederationSecurityError(f"Security Alert: Malicious prompt injection pattern in field '{field_name}'.")

        # 2. Block command separators and path traversals in raw input
        if any(char in text for char in (";", "|", "&")) and ("sudo" in text or "rm" in text or "cat" in text):
            logger.error("Federation security violation: Command injection indicators in '%s'", field_name)
            raise FederationSecurityError(f"Security Alert: Shell separator characters combined with system commands are forbidden.")
            
        if "../" in text:
            logger.error("Federation security violation: Path traversal indicators in '%s'", field_name)
            raise FederationSecurityError("Security Alert: Path traversal queries are forbidden.")

    @classmethod
    def audit_python_code(cls, field_name: str, code_string: str) -> None:
        """
        Analyze Python code strings using AST parsing.

        Ensures no dynamic imports, file accesses, or os-commands exist.

        Raises:
            FederationSecurityError: If forbidden AST patterns are detected.
        """
        if not code_string or not isinstance(code_string, str):
            return

        try:
            tree = ast.parse(code_string)
        except SyntaxError as se:
            logger.warning("Code parameter in '%s' has syntax errors: %s", field_name, se)
            raise FederationSecurityError(f"Security Alert: Syntactically invalid Python payload in '{field_name}'.")

        for node in ast.walk(tree):
            # 1. Block imports, deletes, global statements
            if isinstance(node, cls.FORBIDDEN_AST_NODES):
                logger.error("Federation security violation: Forbidden AST node '%s' found", type(node).__name__)
                raise FederationSecurityError(f"Security Alert: Forbidden AST structure '{type(node).__name__}' in payload.")

            # 2. Block calls to dangerous builtins or modules (eval, exec, open)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name not in cls.ALLOWED_BUILTINS:
                    logger.error("Federation security violation: Forbidden callable '%s' in AST", func_name)
                    raise FederationSecurityError(f"Security Alert: Execution of builtin '{func_name}' is forbidden.")
                    
            # 3. Block attribute access to dangerous builtins
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                logger.error("Federation security violation: Private attribute access '%s' in AST", node.attr)
                raise FederationSecurityError("Security Alert: Accessing private or magic attributes is forbidden.")
