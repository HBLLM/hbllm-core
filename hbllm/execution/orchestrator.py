"""
Execution Orchestrator — coordinates the full execution lifecycle.

Not just a planner — orchestrates:
    - Policy evaluation against system state
    - Runtime selection via RuntimeRegistry
    - Provider selection via ProviderRegistry
    - Capability negotiation via CapabilityResolver
    - Modifier selection (via policy + capabilities)
    - Frozen ExecutionPlan construction
    - Submission to ExecutionBus

The orchestrator is the bridge between cognition and execution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hbllm.execution.bus import ExecutionBus, ExecutionHandle
from hbllm.execution.capability import CapabilityResolver
from hbllm.execution.plan import ExecutionPlan, ExecutionRequest
from hbllm.execution.policy import GenerationPolicy
from hbllm.execution.registry import ProviderRegistry, RuntimeRegistry
from hbllm.execution.result import ExecutionResult

if TYPE_CHECKING:
    from hbllm.execution.manifest import ExecutionManifest

logger = logging.getLogger(__name__)


class ExecutionOrchestrator:
    """
    Coordinates the full execution lifecycle.

    Receives:  ExecutionRequest (from cognitive layer)
    Produces:  ExecutionPlan (frozen, immutable, with identity)
    Submits:   via ExecutionBus → RuntimeRegistry

    The orchestrator is the ONLY component that makes execution
    decisions. Runtimes simply execute the plan they receive.
    """

    def __init__(
        self,
        policy: GenerationPolicy,
        capability_resolver: CapabilityResolver,
        runtime_registry: RuntimeRegistry,
        provider_registry: ProviderRegistry,
        execution_bus: ExecutionBus,
    ) -> None:
        self._policy = policy
        self._capability_resolver = capability_resolver
        self._runtime_registry = runtime_registry
        self._provider_registry = provider_registry
        self._bus = execution_bus

        # Wire the bus to dispatch through the runtime registry
        self._bus.set_runtime_handler(self._dispatch_to_runtime)

    async def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """
        Full orchestration:
            1. Build ExecutionPlan from request + policy + capabilities
            2. Submit to ExecutionBus
            3. Await result

        This is the primary entry point for the cognitive layer.
        """
        plan = await self.plan(request)
        handle = await self._bus.submit(plan)
        return await self._bus.wait(handle)

    async def execute_async(self, request: ExecutionRequest) -> ExecutionHandle:
        """
        Submit for asynchronous execution. Returns a handle.

        Use ``bus.wait(handle)`` to await the result later.
        """
        plan = await self.plan(request)
        return await self._bus.submit(plan)

    async def plan(self, request: ExecutionRequest) -> ExecutionPlan:
        """
        Build a frozen ExecutionPlan from a request.

        Steps:
            1. Evaluate GenerationPolicy against system state
            2. Run CapabilityResolver for negotiation + fallbacks
            3. Determine provider and model
            4. Build frozen ExecutionPlan with identity
        """
        # 1. Resolve capabilities (runtime, provider, modifiers)
        resolved = await self._capability_resolver.resolve(
            request,
            self._policy,
            self._runtime_registry,
            self._provider_registry,
        )

        # 2. Extract payload messages for the plan
        payload_messages = tuple((m.role, m.content) for m in request.payload.messages)

        # 3. Build the frozen plan
        plan = ExecutionPlan(
            task_type=request.task_type,
            runtime=resolved.get("runtime", "text"),
            provider=resolved.get("provider", "local"),
            model_id=resolved.get("model_id"),
            payload_messages=payload_messages,
            temperature=0.7,
            max_tokens=min(request.constraints.max_tokens, 4096),
            streaming=resolved.get("streaming", False),
            capabilities_used=resolved.get("capabilities_used", ()),
        )

        logger.info(
            "ExecutionPlan created: plan_id=%s, runtime=%s, provider=%s, modifiers=%d",
            plan.plan_id,
            plan.runtime,
            plan.provider,
            len(plan.modifiers),
        )

        return plan

    async def plan_from_manifest(self, manifest: ExecutionManifest) -> ExecutionPlan:
        """
        Resolve an ExecutionManifest into an ExecutionPlan.

        The manifest provides declarative configuration;
        the orchestrator resolves it against live system state.
        """
        # Build an ExecutionRequest from the manifest
        from hbllm.execution.payload import ExecutionPayload

        request = ExecutionRequest(
            task_type=manifest.task,
            payload=ExecutionPayload(),
            constraints=manifest.constraints,
        )

        plan = await self.plan(request)

        # Override with manifest preferences if specified
        overrides: dict[str, Any] = {}
        if manifest.runtime is not None:
            overrides["runtime"] = manifest.runtime
        if manifest.provider is not None:
            overrides["provider"] = manifest.provider

        if overrides:
            plan = plan.with_fork(**overrides)

        return plan

    async def _dispatch_to_runtime(self, plan: ExecutionPlan) -> ExecutionResult:
        """
        Dispatch a plan to the appropriate runtime.

        This is the runtime handler wired into the ExecutionBus.
        """

        runtime = self._runtime_registry.resolve(plan.task_type)
        if runtime is None:
            raise RuntimeError(f"No runtime registered for task type: {plan.task_type}")

        logger.debug(
            "Dispatching plan %s to runtime '%s'",
            plan.plan_id,
            runtime.runtime_type,
        )

        return await runtime.execute(plan)
