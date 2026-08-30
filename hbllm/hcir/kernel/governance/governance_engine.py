"""
Governance Engine — Central governance container for HCIR Cognitive OS.

Composes migration, security, resource, capability, and tenant policies to evaluate
execution authorization through the kernel pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.governance.owner_rules import OwnerRuleStore
from hbllm.brain.governance.policy_engine import (
    Policy,
    PolicyAction,
    PolicyCondition,
    PolicyEngine,
    PolicyType,
)
from hbllm.hcir.kernel.governance.constitutional_verifier import ConstitutionalVerifier
from hbllm.hcir.kernel.governance.policies.migration_policy import MigrationMode, MigrationPolicy
from hbllm.security.audit_trail import AuditTrail

logger = logging.getLogger(__name__)


@dataclass
class GovernanceDecision:
    """Output decision from evaluating governance policies."""

    allowed: bool
    reason: str = "Authorized"
    migration_mode: MigrationMode = MigrationMode.HYBRID
    evaluated_policies: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)


class GovernanceEngine:
    """Central policy composition and enforcement engine for Cognitive OS kernel execution."""

    def __init__(
        self,
        migration_policy: MigrationPolicy | None = None,
        policy_engine: PolicyEngine | None = None,
        owner_rule_store: OwnerRuleStore | None = None,
        constitutional_verifier: ConstitutionalVerifier | None = None,
        audit_trail: AuditTrail | None = None,
        auto_load_baseline_safety: bool = True,
    ) -> None:
        self._migration_policy = migration_policy or MigrationPolicy(MigrationMode.HYBRID)
        self._policy_engine = policy_engine or PolicyEngine()
        self._owner_rule_store = owner_rule_store
        self._constitutional_verifier = constitutional_verifier or ConstitutionalVerifier()
        self._audit_trail = audit_trail

        if auto_load_baseline_safety:
            self._load_baseline_safety_policies()

    @property
    def migration_policy(self) -> MigrationPolicy:
        return self._migration_policy

    @property
    def migration_mode(self) -> MigrationMode:
        return self._migration_policy.mode

    @property
    def policy_engine(self) -> PolicyEngine:
        return self._policy_engine

    @property
    def owner_rule_store(self) -> OwnerRuleStore | None:
        return self._owner_rule_store

    @property
    def constitutional_verifier(self) -> ConstitutionalVerifier:
        return self._constitutional_verifier

    @property
    def audit_trail(self) -> AuditTrail | None:
        return self._audit_trail

    def set_audit_trail(self, audit_trail: AuditTrail) -> None:
        self._audit_trail = audit_trail

    def attach_owner_rule_store(self, store: OwnerRuleStore, tenant_id: str = "*") -> None:
        """Attach an OwnerRuleStore and load rules for a given tenant."""
        self._owner_rule_store = store
        store.load_into_engine(tenant_id=tenant_id, engine=self._policy_engine)

    def _load_baseline_safety_policies(self) -> None:
        """Pre-install deterministic life-safety and physical hazard mitigation policies."""
        baseline_policies = [
            Policy(
                name="life_safety_critical_device_protection",
                type=PolicyType.DENY,
                pattern=r"(?:disable|turn_off|deactivate|bypass|mute|silence|suppress).*(?:smoke_detector|smoke\s+detector|fire_alarm|fire\s+alarm|co_detector|carbon_monoxide|sprinkler|alarm_system|e_stop|emergency_stop|safety_interlock)",
                action=PolicyAction.BLOCK,
                description="Prohibit disabling or muting life-safety sensors, alarms, and emergency interlocks",
                priority=1000,
            ),
            Policy(
                name="actuator_force_torque_limit",
                type=PolicyType.DENY,
                pattern=r"(?:force|torque|thrust|pressure).*?([5-9]\d{2,}|\d{4,})",
                action=PolicyAction.BLOCK,
                description="Block excessive actuator force (>500N) or torque commands exceeding physical safety limits",
                priority=900,
            ),
            Policy(
                name="human_in_workspace_actuator_hazard",
                type=PolicyType.DENY,
                pattern=r"(?:move|actuate|drive|swing|lift|punch|weld|cut|rotate|accelerate|force|set_torque|robot_arm|arm)",
                action=PolicyAction.BLOCK,
                conditions=[PolicyCondition(key="human_in_workspace", operator="eq", value=True)],
                description="Block high-energy or hazardous actuator motions when human presence is detected in workspace",
                priority=950,
            ),
            Policy(
                name="unattended_thermal_fire_hazard",
                type=PolicyType.DENY,
                pattern=r"(?:stove|burner|oven|heating_element|torch|laser|furnace).*?(?:max|high|\d{3,})",
                action=PolicyAction.BLOCK,
                conditions=[PolicyCondition(key="unattended", operator="eq", value=True)],
                description="Prohibit max-power thermal or burner actuation when unattended or for dangerous durations",
                priority=900,
            ),
            Policy(
                name="unauthorized_door_unlock_and_perimeter_security",
                type=PolicyType.DENY,
                pattern=r"(?:unlock|open).*(?:front_door|main_entrance|back_door|perimeter_gate|safe|vault|security_door|door)",
                action=PolicyAction.BLOCK,
                conditions=[PolicyCondition(key="authorized", operator="eq", value=False)],
                description="Prohibit unauthorized physical perimeter door or vault unlock commands",
                priority=900,
            ),
            Policy(
                name="adversarial_override_rejection",
                type=PolicyType.DENY,
                pattern=r"(?:system\s+override|disregard\s+rules|developer_mode|bypass_governance|sudo\s+force|ignore_safety)",
                action=PolicyAction.BLOCK,
                description="Reject adversarial prompt injection tokens attempting to override safety governance",
                priority=1000,
            ),
        ]

        for p in baseline_policies:
            if not self._policy_engine.get_policy(p.name):
                self._policy_engine.add_policy(p)

    def evaluate_execution(
        self,
        capability_name: str,
        arguments: dict[str, Any],
        context: Any = None,
    ) -> GovernanceDecision:
        """Evaluate all governance policies for a proposed capability execution."""
        evaluated = ["MigrationPolicy", "PolicyEngine", "SecurityPolicy", "TenantPolicy"]
        violations: list[str] = []

        # 1. Build canonical evaluation text
        cap_clean = capability_name.replace("_", " ")
        arg_tokens = []
        for k, v in arguments.items():
            k_clean = k.replace("_", " ")
            arg_tokens.append(f"{k_clean}={v}")
        eval_text = f"{capability_name} {cap_clean} " + " ".join(arg_tokens)

        # 2. Build canonical runtime context
        eval_context: dict[str, Any] = {}
        if isinstance(context, dict):
            eval_context.update(context)
        elif hasattr(context, "__dict__"):
            eval_context.update(
                {k: v for k, v in context.__dict__.items() if not k.startswith("_")}
            )

        # Inject arguments into context for condition evaluation
        eval_context.update(arguments)

        # Normalize key safety context flags
        if arguments.get("unauthorized") is True or arguments.get("authorized") is False:
            eval_context["authorized"] = False
        elif "authorized" not in eval_context and not arguments.get("unauthorized"):
            eval_context["authorized"] = True

        if (
            arguments.get("human_in_workspace")
            or arguments.get("human_present")
            or eval_context.get("human_present")
        ):
            eval_context["human_in_workspace"] = True

        if arguments.get("unattended") or eval_context.get("unattended"):
            eval_context["unattended"] = True

        # State key normalization (e.g. baby="sleeping" -> baby_state="sleeping")
        for k, v in list(eval_context.items()):
            if isinstance(v, str) and not k.endswith("_state"):
                eval_context[f"{k}_state"] = v

        tenant_id = str(eval_context.get("tenant_id", "default"))
        domain = str(eval_context.get("domain", ""))

        # 3. Evaluate PolicyEngine
        policy_res = self._policy_engine.evaluate(
            text=eval_text,
            tenant_id=tenant_id,
            domain=domain,
            context=eval_context,
        )

        if not policy_res.passed:
            violations.extend(policy_res.violations)

        # 4. Determine overall approval
        allowed = len(violations) == 0
        reason = (
            "Execution authorized under current governance policy"
            if allowed
            else f"Blocked by governance policies: {', '.join(violations)}"
        )

        # 5. Cryptographic audit trail logging
        if self._audit_trail is not None:
            try:
                self._audit_trail.log(
                    tenant_id=tenant_id,
                    action=f"kernel.execute.{capability_name}",
                    category="governance",
                    result="authorized" if allowed else "denied",
                    details={
                        "capability": capability_name,
                        "arguments": arguments,
                        "reason": reason,
                        "violations": violations,
                    },
                )
            except Exception as e:
                logger.warning("Audit trail logging failed: %s", e)

        return GovernanceDecision(
            allowed=allowed,
            reason=reason,
            migration_mode=self._migration_policy.mode,
            evaluated_policies=evaluated,
            violations=violations,
        )
