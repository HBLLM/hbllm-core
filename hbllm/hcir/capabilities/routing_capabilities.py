"""Routing & Execution Capabilities — HCIR §10.

Registers routing.resolve_route and execution.sandbox_run capabilities
and connects them to CapabilityResolver and CapabilitySandboxManager.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hbllm.hcir.kernel.capability_resolver import ICapabilityExecutor
from hbllm.hcir.kernel.cognitive_abi import CapabilityCall, CapabilityManifest, CapabilityResult

if TYPE_CHECKING:
    from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
    from hbllm.hcir.kernel.capability_sandboxing import CapabilitySandboxManager

logger = logging.getLogger(__name__)

ROUTING_MANIFESTS = [
    CapabilityManifest(name="routing.resolve_route", owner="HCIR.Router"),
    CapabilityManifest(name="execution.sandbox_run", owner="HCIR.Executor"),
]


def execute_routing_capability(
    call: CapabilityCall,
    resolver: CapabilityResolver | None = None,
    sandbox_manager: CapabilitySandboxManager | None = None,
) -> CapabilityResult:
    """Execute a routing or execution capability call."""
    cap_name = call.capability_name
    params = getattr(call, "arguments", None) or getattr(call, "parameters", None) or {}
    logger.debug("Executing routing capability: %s with params %s", cap_name, params)

    if cap_name == "routing.resolve_route":
        target_cap = (
            params.get("target_capability")
            or params.get("capability_name")
            or params.get("capability")
            or "world_prediction"
        )
        resolved_impl_id = "none"
        priority = 0
        is_sandboxed = False

        if resolver is not None:
            impl = resolver.resolve_implementation(target_cap)
            if impl is not None:
                resolved_impl_id = impl.implementation_id
                priority = impl.priority

        if sandbox_manager is not None and resolved_impl_id != "none":
            policy = sandbox_manager.get_policy(target_cap, resolved_impl_id)
            is_sandboxed = policy is not None

        output = {
            "capability": cap_name,
            "target_capability": target_cap,
            "resolved_route": resolved_impl_id,
            "priority": priority,
            "sandboxed": is_sandboxed,
        }
        status = "SUCCESS" if resolved_impl_id != "none" else "UNRESOLVED"
        return CapabilityResult(capability_name=cap_name, status=status, output=output)

    elif cap_name == "execution.sandbox_run":
        target_action = params.get("action") or params.get("command") or "noop"
        timeout_sec = float(params.get("timeout_seconds", 5.0))
        target_impl = params.get("implementation_id") or "sandbox_generic"

        is_allowed = True
        sandbox_profile = "default"

        if sandbox_manager is not None:
            policy = sandbox_manager.get_policy("execution.sandbox_run", target_impl)
            if policy is not None:
                sandbox_profile = str(policy.isolation_mode)
            is_allowed = sandbox_manager.check_permission(
                "execution.sandbox_run", target_impl, target_action
            )

        output = {
            "capability": cap_name,
            "action": target_action,
            "implementation_id": target_impl,
            "sandbox_profile": sandbox_profile,
            "sandboxed": True,
            "timeout_seconds": timeout_sec,
            "allowed": is_allowed,
        }
        return CapabilityResult(
            capability_name=cap_name,
            status="SUCCESS" if is_allowed else "SANDBOX_VIOLATION",
            output=output,
        )

    else:
        return CapabilityResult(
            capability_name=cap_name,
            status="UNKNOWN_CAPABILITY",
            output={"error": f"Routing capability '{cap_name}' is not recognized"},
        )


class RoutingCapabilityExecutor(ICapabilityExecutor):
    """Capability executor for routing.* and execution.sandbox_run capabilities."""

    def __init__(
        self,
        resolver: CapabilityResolver | None = None,
        sandbox_manager: CapabilitySandboxManager | None = None,
        default_capability: str = "routing.resolve_route",
    ) -> None:
        self._resolver = resolver
        self._sandbox_manager = sandbox_manager
        self._default_capability = default_capability

    @property
    def is_available(self) -> bool:
        return True

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Execute routing capability via CapabilityResolver."""
        cap_name = (
            params.get("capability") or params.get("capability_name") or self._default_capability
        )
        call = CapabilityCall(
            capability_name=cap_name,
            arguments=params.get("params") or params,
        )
        res = execute_routing_capability(
            call=call,
            resolver=self._resolver,
            sandbox_manager=self._sandbox_manager,
        )
        return {
            "status": res.status,
            **res.output,
        }
