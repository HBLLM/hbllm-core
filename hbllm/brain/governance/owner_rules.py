"""
Owner Rules — natural language → structured policy conversion.

Re-exports canonical owner rule parser and storage definitions from
hbllm.hcir.kernel.governance.policies.owner_rules for backward compatibility.
"""

from __future__ import annotations

from hbllm.hcir.kernel.governance.policies.owner_rules import (
    OwnerRuleStore,
    ParsedRule,
    parse_owner_rule,
)

__all__ = [
    "OwnerRuleStore",
    "ParsedRule",
    "parse_owner_rule",
]
