"""
Governance Engine — Central governance container for HCIR Cognitive OS.

Composes migration, security, resource, capability, and tenant policies to evaluate
execution authorization through the kernel pipeline. Employs structured intent analysis,
semantic synonym normalization, order-independent invariant checks, and fail-closed
authorization defaults for physical and life-safety operations.
"""

from __future__ import annotations

import logging
import re
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

# ── Semantic Synonym & Domain Dictionaries ────────────────────────────────────

_VERB_ARGUMENT_KEYS = {
    "action",
    "operation",
    "op",
    "command",
    "verb",
    "task",
    "method",
    "mode",
    "instruction",
    "function",
    "intent",
    "step",
    "activity",
    "subaction",
    "procedure",
    "directive",
}

_TARGET_ARGUMENT_KEYS = {
    "target",
    "device",
    "component",
    "object",
    "item",
    "entity",
    "resource",
    "system",
    "appliance",
    "fixture",
    "asset",
    "element",
    "subject",
    "unit",
    "subsystem",
    "module",
    "actuator_target",
}

_HUMAN_PROXIMITY_PATTERNS = (
    "human",
    "person",
    "occupant",
    "bystander",
    "operator",
    "worker",
    "pedestrian",
    "user_present",
    "presence",
    "proximity",
    "body",
    "face",
    "motion_detected",
    "obstacle",
    "living_being",
    "someone_near",
    "people",
)

_UNATTENDED_PATTERNS = (
    "unattended",
    "unmonitored",
    "alone",
    "no_supervision",
    "unsupervised",
    "no_human_supervision",
    "without_supervision",
)

_AUTH_POSITIVE_PATTERNS = (
    "authorized",
    "is_authorized",
    "authorized_user",
    "authenticated",
    "auth",
    "is_authenticated",
    "has_permission",
    "permitted",
    "allow_access",
    "access_granted",
    "has_access",
)

_AUTH_NEGATIVE_PATTERNS = (
    "unauthorized",
    "is_unauthorized",
    "unauthenticated",
    "not_authorized",
    "disallowed",
    "access_denied",
    "no_access",
)

_OVERRIDE_PATTERNS = (
    "override",
    "force",
    "bypass",
    "sudo",
    "developer_mode",
    "disregard_rules",
    "ignore_safety",
)

_LIFE_SAFETY_TARGETS = {
    "smoke_detector",
    "smoke_alarm",
    "fire_alarm",
    "co_detector",
    "carbon_monoxide",
    "sprinkler",
    "alarm_system",
    "e_stop",
    "emergency_stop",
    "safety_interlock",
    "gas_leak_detector",
    "hazard_sensor",
}

_PERIMETER_TARGETS = {
    "front_door",
    "main_entrance",
    "back_door",
    "perimeter_gate",
    "gate",
    "safe",
    "vault",
    "security_door",
    "door",
    "lock",
    "exterior_door",
    "patio_door",
    "entrance",
    "barrier",
    "turnstile",
    "portal",
    "dock",
    "entry",
}

_ROBOTIC_ACTUATOR_TARGETS = {
    "robot_arm",
    "arm",
    "manipulator",
    "actuator",
    "gripper",
    "linear_stage",
    "joint",
    "servo",
    "motor",
    "robot",
    "arm_assembly",
    "cutter",
    "welder",
    "press",
}

_THERMAL_TARGETS = {
    "stove",
    "burner",
    "stove_burner",
    "oven",
    "heating_element",
    "torch",
    "laser",
    "furnace",
    "heater",
    "cooktop",
}

_UNLOCK_ACTION_SYNONYMS = {
    "unlock",
    "open",
    "disarm",
    "grant",
    "unlatch",
    "release",
    "access",
    "permit_entry",
    "allow_entry",
}

_ACTUATE_ACTION_SYNONYMS = {
    "move",
    "actuate",
    "drive",
    "swing",
    "lift",
    "punch",
    "weld",
    "cut",
    "rotate",
    "accelerate",
    "force",
    "set_torque",
    "extend",
    "retract",
    "grip",
    "close_gripper",
    "open_gripper",
    "robot_arm",
    "operate",
    "swing_robot_arm",
}


def _flatten_mapping(data: Any, prefix: str = "", max_depth: int = 10) -> dict[str, Any]:
    """Recursively flatten arbitrary nested dictionaries, lists, and dataclasses into key-value map."""
    flattened: dict[str, Any] = {}
    if max_depth <= 0 or data is None:
        return flattened

    if isinstance(data, dict):
        for k, v in data.items():
            k_str = str(k).lower()
            flattened[k_str] = v
            if prefix:
                flattened[f"{prefix}.{k_str}"] = v
            if isinstance(v, (dict, list, tuple, set)) or hasattr(v, "__dict__"):
                flattened.update(
                    _flatten_mapping(
                        v, prefix=f"{prefix}.{k_str}" if prefix else k_str, max_depth=max_depth - 1
                    )
                )
    elif isinstance(data, (list, tuple, set)):
        for idx, item in enumerate(data):
            if isinstance(item, (dict, list, tuple, set)) or hasattr(item, "__dict__"):
                flattened.update(
                    _flatten_mapping(
                        item,
                        prefix=f"{prefix}[{idx}]" if prefix else f"[{idx}]",
                        max_depth=max_depth - 1,
                    )
                )
    elif hasattr(data, "__dict__"):
        flattened.update(
            _flatten_mapping(
                {k: v for k, v in data.__dict__.items() if not k.startswith("_")},
                prefix=prefix,
                max_depth=max_depth - 1,
            )
        )
    return flattened


@dataclass
class StructuredIntent:
    """Normalized semantic intent extracted from capability execution requests."""

    action: str = ""
    target: str = ""
    force_torque_value: float = 0.0
    is_unattended: bool = False
    human_in_workspace: bool = False
    is_override_attempt: bool = False
    is_authorized: bool | None = None  # None = not explicitly specified
    power_level: str = "normal"
    duration_minutes: float = 0.0


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
                pattern=r"(?:disable|turn_off|deactivate|bypass|mute|silence|suppress).*(?:smoke_detector|smoke\s+detector|fire_alarm|fire\s+alarm|co_detector|carbon_monoxide|sprinkler|alarm_system|e_stop|emergency_stop|safety_interlock|gas_leak_detector)",
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
                pattern=r"(?:move|actuate|drive|swing|lift|punch|weld|cut|rotate|accelerate|force|set_torque|robot_arm|arm|manipulator|gripper)",
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
                pattern=r"(?:unlock|open).*(?:front_door|main_entrance|back_door|perimeter_gate|safe|vault|security_door|door|gate|entrance)",
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

    def _extract_intent(
        self,
        capability_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
    ) -> StructuredIntent:
        """Extract and normalize structured semantic intent recursively from arbitrary capability calls."""
        intent = StructuredIntent()

        # Build comprehensive recursive argument map
        flat_args = _flatten_mapping(arguments)
        flat_ctx = _flatten_mapping(context)
        all_flattened = {**flat_ctx, **flat_args}

        # 1. Action Verb resolution (recursively scanning verb keys)
        detected_action = ""
        for verb_key in _VERB_ARGUMENT_KEYS:
            if verb_key in all_flattened and all_flattened[verb_key]:
                val = str(all_flattened[verb_key]).lower().strip()
                if val:
                    detected_action = val
                    break

        if not detected_action:
            parts = capability_name.lower().split(".")[-1].split("_")
            detected_action = parts[0] if parts else capability_name.lower()

        intent.action = detected_action

        # 2. Target resolution (recursively scanning target keys)
        detected_target = ""
        for target_key in _TARGET_ARGUMENT_KEYS:
            if target_key in all_flattened and all_flattened[target_key]:
                val = str(all_flattened[target_key]).lower().strip()
                if val:
                    detected_target = val
                    break

        if not detected_target:
            parts = capability_name.lower().split(".")[-1].split("_")
            if len(parts) > 1:
                detected_target = "_".join(parts[1:])
            else:
                detected_target = parts[0]

        intent.target = detected_target

        # 3. Proximity / Human Presence detection (Semantic scanning across all keys and nested structures)
        for k, v in all_flattened.items():
            k_clean = k.lower().replace("-", "_")
            if any(p in k_clean for p in _HUMAN_PROXIMITY_PATTERNS):
                if v is True or str(v).lower() in (
                    "true",
                    "1",
                    "yes",
                    "detected",
                    "present",
                    "active",
                ):
                    intent.human_in_workspace = True
                    break

        # 4. Unattended / Supervision state (Semantic scanning)
        for k, v in all_flattened.items():
            k_clean = k.lower().replace("-", "_")
            if any(p in k_clean for p in _UNATTENDED_PATTERNS):
                if v is True or str(v).lower() in ("true", "1", "yes", "active"):
                    intent.is_unattended = True
                    break

        # 5. Force / Torque scanning (recursive)
        for k, v in all_flattened.items():
            k_lower = k.lower()
            if any(term in k_lower for term in ("force", "torque", "thrust", "power_n", "nm")):
                try:
                    intent.force_torque_value = max(intent.force_torque_value, float(v))
                except (ValueError, TypeError):
                    pass

        # 6. Thermal / Power / Duration scanning (recursive)
        power_val = str(all_flattened.get("power", all_flattened.get("heat", ""))).lower()
        if "max" in power_val or "high" in power_val or "100" in power_val:
            intent.power_level = "max"

        for k, v in all_flattened.items():
            if "duration" in k.lower() or "time" in k.lower():
                try:
                    intent.duration_minutes = max(intent.duration_minutes, float(v))
                except (ValueError, TypeError):
                    pass

        # 7. Authorization scanning (Semantic & Fail-Closed)
        # Check explicit negative authorization keys
        for k, v in all_flattened.items():
            k_clean = k.lower().replace("-", "_")
            if any(p in k_clean for p in _AUTH_NEGATIVE_PATTERNS):
                if v is True or str(v).lower() in ("true", "1", "yes"):
                    intent.is_authorized = False
                    break

        # Check positive authorization keys
        if intent.is_authorized is None:
            for k, v in all_flattened.items():
                k_clean = k.lower().replace("-", "_")
                if any(p == k_clean or f".{p}" in k_clean for p in _AUTH_POSITIVE_PATTERNS):
                    if v is False or str(v).lower() in (
                        "false",
                        "0",
                        "no",
                        "none",
                        "denied",
                        "disabled",
                    ):
                        intent.is_authorized = False
                        break
                    elif v is True or str(v).lower() in ("true", "1", "yes", "granted", "enabled"):
                        intent.is_authorized = True
                        break

        # 8. Override / Prompt Injection scanning (recursive)
        for k, v in all_flattened.items():
            k_clean = k.lower().replace("-", "_")
            if any(p in k_clean for p in _OVERRIDE_PATTERNS):
                if v is True or str(v).lower() in ("true", "1", "yes"):
                    intent.is_override_attempt = True
                    break
            if isinstance(v, str) and re.search(
                r"(?:override|disregard|developer_mode|bypass|ignore_safety)", v, re.IGNORECASE
            ):
                intent.is_override_attempt = True
                break

        if re.search(
            r"(?:override|disregard|developer_mode|bypass|ignore_safety)",
            capability_name,
            re.IGNORECASE,
        ):
            intent.is_override_attempt = True

        return intent

    def _evaluate_structured_invariants(
        self,
        intent: StructuredIntent,
        eval_context: dict[str, Any],
    ) -> list[str]:
        """Direct structured invariant checks that do not rely on fragile string ordering."""
        violations: list[str] = []

        # 1. Life Safety Critical Device Protection
        if any(dev in intent.target for dev in _LIFE_SAFETY_TARGETS) or any(
            dev in intent.action for dev in _LIFE_SAFETY_TARGETS
        ):
            if any(
                act in intent.action
                for act in (
                    "disable",
                    "turn_off",
                    "deactivate",
                    "bypass",
                    "mute",
                    "silence",
                    "suppress",
                    "kill",
                )
            ) or any(
                act in intent.target
                for act in (
                    "disable",
                    "turn_off",
                    "deactivate",
                    "bypass",
                    "mute",
                    "silence",
                    "suppress",
                    "kill",
                )
            ):
                violations.append(
                    "[CRITICAL] life_safety_critical_device_protection: Prohibit disabling or muting life-safety sensors, alarms, and emergency interlocks"
                )

        # 2. Unauthorized Perimeter Door / Gate Unlock (Fail-Closed)
        is_perimeter_target = any(door in intent.target for door in _PERIMETER_TARGETS) or any(
            door in intent.action for door in _PERIMETER_TARGETS
        )
        is_unlock_action = any(act in intent.action for act in _UNLOCK_ACTION_SYNONYMS) or any(
            act in intent.target for act in _UNLOCK_ACTION_SYNONYMS
        )

        if is_perimeter_target and is_unlock_action:
            if eval_context.get("authorized") is not True:
                violations.append(
                    "[HIGH] unauthorized_door_unlock_and_perimeter_security: Prohibit unauthorized physical perimeter door or vault unlock commands"
                )

        # 3. Actuator Force & Torque Limits
        if intent.force_torque_value > 500.0:
            violations.append(
                f"[HIGH] actuator_force_torque_limit: Block excessive actuator force (>500N) or torque commands exceeding physical safety limits (received {intent.force_torque_value})"
            )

        # 4. Human in Workspace Actuator Hazard (Fail-Closed Caution)
        is_actuator_target = any(act in intent.target for act in _ROBOTIC_ACTUATOR_TARGETS) or any(
            act in intent.action for act in _ROBOTIC_ACTUATOR_TARGETS
        )
        is_actuator_action = any(act in intent.action for act in _ACTUATE_ACTION_SYNONYMS)

        if eval_context.get("human_in_workspace") is True:
            if is_actuator_target or is_actuator_action:
                violations.append(
                    "[CRITICAL] human_in_workspace_actuator_hazard: Block high-energy or hazardous actuator motions when human presence is detected in workspace"
                )

        # 5. Unattended Thermal / Fire Hazard
        if any(th in intent.target for th in _THERMAL_TARGETS) or any(
            th in intent.action for th in _THERMAL_TARGETS
        ):
            if eval_context.get("unattended") is True:
                if intent.power_level in ("max", "high") or intent.duration_minutes > 120:
                    violations.append(
                        "[HIGH] unattended_thermal_fire_hazard: Prohibit max-power thermal or burner actuation when unattended or for dangerous durations"
                    )

        # 6. Adversarial Override Rejection
        if intent.is_override_attempt:
            violations.append(
                "[CRITICAL] adversarial_override_rejection: Reject adversarial prompt injection tokens attempting to override safety governance"
            )

        return violations

    def evaluate_execution(
        self,
        capability_name: str,
        arguments: dict[str, Any],
        context: Any = None,
    ) -> GovernanceDecision:
        """Evaluate all governance policies for a proposed capability execution."""
        evaluated = [
            "MigrationPolicy",
            "PolicyEngine",
            "StructuredInvariants",
            "SecurityPolicy",
            "TenantPolicy",
        ]
        violations: list[str] = []

        # 1. Base runtime context synthesis
        eval_context: dict[str, Any] = {}
        if isinstance(context, dict):
            eval_context.update(context)
        elif hasattr(context, "__dict__"):
            eval_context.update(
                {k: v for k, v in context.__dict__.items() if not k.startswith("_")}
            )

        # Inject arguments into context
        eval_context.update(arguments)

        # 2. Extract structured intent & normalize synonyms
        intent = self._extract_intent(capability_name, arguments, eval_context)

        # 3. Apply Fail-Closed Authorization Policy
        # For security-sensitive targets, authorization defaults to False unless explicitly proven True
        is_sensitive_target = (
            any(t in intent.target for t in _PERIMETER_TARGETS)
            or any(t in intent.target for t in _LIFE_SAFETY_TARGETS)
            or any(t in capability_name.lower() for t in _PERIMETER_TARGETS)
            or any(t in capability_name.lower() for t in _LIFE_SAFETY_TARGETS)
        )

        if intent.is_authorized is False:
            eval_context["authorized"] = False
        elif intent.is_authorized is True:
            eval_context["authorized"] = True
        elif is_sensitive_target:
            # Fail-closed default on sensitive targets
            eval_context["authorized"] = False
        else:
            # Default for non-sensitive operations (e.g. sensor readings, UI)
            eval_context["authorized"] = True

        if intent.human_in_workspace:
            eval_context["human_in_workspace"] = True

        if intent.is_unattended:
            eval_context["unattended"] = True

        # State key normalization (e.g. baby="sleeping" -> baby_state="sleeping")
        for k, v in list(eval_context.items()):
            if isinstance(v, str) and not k.endswith("_state"):
                eval_context[f"{k}_state"] = v

        tenant_id = str(eval_context.get("tenant_id", "default"))
        domain = str(eval_context.get("domain", ""))

        # 4. Direct Structured Invariant Checks
        structured_violations = self._evaluate_structured_invariants(intent, eval_context)
        violations.extend(structured_violations)

        # 5. Order-Independent & Multi-representation Text Synthesis for PolicyEngine
        cap_clean = capability_name.replace("_", " ")
        arg_tokens = [f"{k.replace('_', ' ')}={v}" for k, v in arguments.items()]
        eval_text = (
            f"{capability_name} {cap_clean} "
            f"action:{intent.action} target:{intent.target} "
            f"{intent.action} {intent.target} {intent.target} {intent.action} "
            + " ".join(arg_tokens)
        )

        # 6. Evaluate PolicyEngine (for custom owner rules and configured policies)
        policy_res = self._policy_engine.evaluate(
            text=eval_text,
            tenant_id=tenant_id,
            domain=domain,
            context=eval_context,
        )

        if not policy_res.passed:
            for v in policy_res.violations:
                if v not in violations:
                    violations.append(v)

        # 7. Determine overall approval
        allowed = len(violations) == 0
        reason = (
            "Execution authorized under current governance policy"
            if allowed
            else f"Blocked by governance policies: {', '.join(violations)}"
        )

        # 8. Cryptographic audit trail logging
        if self._audit_trail is not None:
            try:
                self._audit_trail.log(
                    tenant_id=tenant_id,
                    action=f"kernel.execute.{capability_name}",
                    category="governance",
                    result="authorized" if allowed else "denied",
                    details={
                        "capability": capability_name,
                        "intent": {
                            "action": intent.action,
                            "target": intent.target,
                            "is_authorized": eval_context.get("authorized"),
                            "human_in_workspace": eval_context.get("human_in_workspace"),
                            "unattended": eval_context.get("unattended"),
                        },
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
