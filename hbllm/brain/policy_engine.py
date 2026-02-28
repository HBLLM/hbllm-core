"""
Policy Engine — governance layer for HBLLM responses.

Enforces configurable rules (laws/policies/constraints) on every response
before delivery. Acts as a constitutional framework for the AI.

Policy types:
  DENY      — block responses matching a pattern
  REQUIRE   — ensure responses include required elements
  TRANSFORM — modify responses (e.g., add disclaimers)
  SCOPE     — restrict which domains/tools a tenant can access
  RATE      — per-tenant limits on specific capabilities
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class PolicyType(str, Enum):
    DENY = "deny"
    REQUIRE = "require"
    TRANSFORM = "transform"
    SCOPE = "scope"
    RATE = "rate"


class PolicyAction(str, Enum):
    BLOCK = "block"           # Block the response entirely
    WARN = "warn"             # Allow but log a warning
    APPEND = "append"         # Append text to response
    PREPEND = "prepend"       # Prepend text to response
    REPLACE = "replace"       # Replace matched content
    RESTRICT = "restrict"     # Restrict access to domains/tools


@dataclass
class Policy:
    """A single governance policy/rule."""
    name: str
    type: PolicyType
    action: PolicyAction = PolicyAction.BLOCK
    description: str = ""
    pattern: str = ""          # Regex pattern for DENY/REQUIRE
    content: str = ""          # Content for TRANSFORM (append/prepend text)
    domains: list[str] = field(default_factory=list)  # For SCOPE policies
    tenant_ids: list[str] = field(default_factory=lambda: ["*"])  # Which tenants
    priority: int = 0          # Higher = evaluated first
    enabled: bool = True
    severity: str = "medium"   # low, medium, high, critical

    def applies_to(self, tenant_id: str) -> bool:
        """Check if this policy applies to a given tenant."""
        return "*" in self.tenant_ids or tenant_id in self.tenant_ids


@dataclass
class PolicyResult:
    """Result of policy evaluation."""
    passed: bool
    original_text: str
    modified_text: str
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    applied_policies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": self.violations,
            "warnings": self.warnings,
            "applied_policies": self.applied_policies,
        }


class PolicyEngine:
    """
    Evaluates responses against a set of governance policies.

    Usage:
        engine = PolicyEngine()
        engine.load_from_yaml("config/policies.yaml")
        result = engine.evaluate("response text", tenant_id="t1")
        if not result.passed:
            # Response was blocked
    """

    def __init__(self) -> None:
        self._policies: list[Policy] = []

    @property
    def policy_count(self) -> int:
        return len(self._policies)

    def add_policy(self, policy: Policy) -> None:
        """Add a policy and re-sort by priority."""
        self._policies.append(policy)
        self._policies.sort(key=lambda p: p.priority, reverse=True)

    def remove_policy(self, name: str) -> bool:
        """Remove a policy by name."""
        before = len(self._policies)
        self._policies = [p for p in self._policies if p.name != name]
        return len(self._policies) < before

    def get_policy(self, name: str) -> Policy | None:
        """Get a single policy by name."""
        for p in self._policies:
            if p.name == name:
                return p
        return None

    def load_from_yaml(self, path: str | Path) -> int:
        """
        Load policies from a YAML file.

        Returns the number of policies loaded.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Policy file not found: %s", path)
            return 0

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        policies = data.get("policies", [])
        loaded = 0

        for p_data in policies:
            try:
                policy = Policy(
                    name=p_data["name"],
                    type=PolicyType(p_data["type"]),
                    action=PolicyAction(p_data.get("action", "block")),
                    description=p_data.get("description", ""),
                    pattern=p_data.get("pattern", ""),
                    content=p_data.get("content", ""),
                    domains=p_data.get("domains", []),
                    tenant_ids=p_data.get("tenant_ids", ["*"]),
                    priority=p_data.get("priority", 0),
                    enabled=p_data.get("enabled", True),
                    severity=p_data.get("severity", "medium"),
                )
                self.add_policy(policy)
                loaded += 1
            except (KeyError, ValueError) as e:
                logger.warning("Invalid policy entry: %s — %s", p_data.get("name", "?"), e)

        logger.info("Loaded %d policies from %s", loaded, path)
        return loaded

    def evaluate(
        self,
        text: str,
        tenant_id: str = "default",
        domain: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> PolicyResult:
        """
        Evaluate a response against all applicable policies.

        Args:
            text: The response text to evaluate
            tenant_id: Tenant for scope filtering
            domain: Domain the response was generated from
            metadata: Additional context

        Returns:
            PolicyResult with pass/fail, modified text, and violation details.
        """
        result = PolicyResult(
            passed=True,
            original_text=text,
            modified_text=text,
        )

        for policy in self._policies:
            if not policy.enabled:
                continue
            if not policy.applies_to(tenant_id):
                continue

            if policy.type == PolicyType.DENY:
                self._eval_deny(policy, result)
            elif policy.type == PolicyType.REQUIRE:
                self._eval_require(policy, result)
            elif policy.type == PolicyType.TRANSFORM:
                self._eval_transform(policy, result)
            elif policy.type == PolicyType.SCOPE:
                self._eval_scope(policy, result, domain)

            # Stop on first blocking violation
            if not result.passed and policy.action == PolicyAction.BLOCK:
                break

        return result

    def _eval_deny(self, policy: Policy, result: PolicyResult) -> None:
        """Check if response contains denied content."""
        if not policy.pattern:
            return

        try:
            if re.search(policy.pattern, result.modified_text, re.IGNORECASE):
                if policy.action == PolicyAction.BLOCK:
                    result.passed = False
                    result.violations.append(
                        f"[{policy.severity.upper()}] {policy.name}: {policy.description}"
                    )
                    result.modified_text = (
                        f"I cannot provide this response due to policy: {policy.name}. "
                        f"{policy.description}"
                    )
                elif policy.action == PolicyAction.WARN:
                    result.warnings.append(f"{policy.name}: {policy.description}")
                elif policy.action == PolicyAction.REPLACE:
                    result.modified_text = re.sub(
                        policy.pattern, policy.content or "[REDACTED]",
                        result.modified_text, flags=re.IGNORECASE
                    )

                result.applied_policies.append(policy.name)
        except re.error as e:
            logger.warning("Invalid regex in policy %s: %s", policy.name, e)

    def _eval_require(self, policy: Policy, result: PolicyResult) -> None:
        """Check if response includes required elements."""
        if not policy.pattern:
            return

        try:
            if not re.search(policy.pattern, result.modified_text, re.IGNORECASE):
                if policy.action == PolicyAction.BLOCK:
                    result.passed = False
                    result.violations.append(
                        f"[{policy.severity.upper()}] {policy.name}: Missing required element"
                    )
                elif policy.action == PolicyAction.APPEND:
                    result.modified_text += f"\n\n{policy.content}"
                elif policy.action == PolicyAction.WARN:
                    result.warnings.append(f"{policy.name}: required element missing")

                result.applied_policies.append(policy.name)
        except re.error as e:
            logger.warning("Invalid regex in policy %s: %s", policy.name, e)

    def _eval_transform(self, policy: Policy, result: PolicyResult) -> None:
        """Apply transformations to the response."""
        if policy.action == PolicyAction.APPEND:
            result.modified_text += f"\n\n{policy.content}"
            result.applied_policies.append(policy.name)
        elif policy.action == PolicyAction.PREPEND:
            result.modified_text = f"{policy.content}\n\n{result.modified_text}"
            result.applied_policies.append(policy.name)
        elif policy.action == PolicyAction.REPLACE and policy.pattern:
            try:
                new_text = re.sub(
                    policy.pattern, policy.content,
                    result.modified_text, flags=re.IGNORECASE
                )
                if new_text != result.modified_text:
                    result.modified_text = new_text
                    result.applied_policies.append(policy.name)
            except re.error:
                pass

    def _eval_scope(self, policy: Policy, result: PolicyResult, domain: str) -> None:
        """Check if the domain is allowed for this tenant."""
        if not policy.domains:
            return

        if policy.action == PolicyAction.RESTRICT and domain:
            if domain not in policy.domains:
                result.passed = False
                result.violations.append(
                    f"[{policy.severity.upper()}] {policy.name}: "
                    f"Domain '{domain}' not allowed (allowed: {', '.join(policy.domains)})"
                )
                result.modified_text = (
                    f"Access to the '{domain}' domain is restricted by policy: {policy.name}."
                )
                result.applied_policies.append(policy.name)

    def list_policies(self) -> list[dict[str, Any]]:
        """Return all policies as dicts."""
        return [
            {
                "name": p.name,
                "type": p.type.value,
                "action": p.action.value,
                "description": p.description,
                "enabled": p.enabled,
                "priority": p.priority,
                "severity": p.severity,
                "tenant_ids": p.tenant_ids,
            }
            for p in self._policies
        ]
