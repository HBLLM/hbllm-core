"""
Runtime Capabilities & Capability Resolver.

Every runtime and provider advertises its capabilities.
The CapabilityResolver does set intersection:

    Needs ∩ Runtime ∩ Provider = Plan

This is much easier to maintain than procedural negotiation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hbllm.execution.plan import ExecutionRequest
    from hbllm.execution.policy import GenerationPolicy, SystemState
    from hbllm.execution.registry import ProviderRegistry, RuntimeRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeCapabilities:
    """
    Advertised by every runtime and provider.

    Runtimes and providers declare what they support.
    The resolver intersects needs with capabilities
    instead of procedural if/elif negotiation.
    """

    streaming: bool = False
    json_mode: bool = False
    vision: bool = False
    audio: bool = False
    grammar: bool = False
    tool_calls: bool = False
    embeddings: bool = False
    max_context: int = 4096
    max_output: int = 4096
    supports_lora: bool = False
    supported_modifiers: tuple[str, ...] = ()
    custom: dict[str, Any] = field(default_factory=dict)

    def satisfies(self, required: tuple[str, ...]) -> bool:
        """Check if all required capabilities are satisfied."""
        capability_map = {
            "streaming": self.streaming,
            "json_mode": self.json_mode,
            "json": self.json_mode,
            "vision": self.vision,
            "audio": self.audio,
            "grammar": self.grammar,
            "tool_calls": self.tool_calls,
            "embeddings": self.embeddings,
            "lora": self.supports_lora,
        }
        for cap in required:
            if cap in capability_map and not capability_map[cap]:
                return False
            if cap in self.custom and not self.custom[cap]:
                return False
        return True


class CapabilityResolver:
    """
    Resolves execution requirements against available capabilities.

    Flow:
        Needs ∩ Runtime ∩ Provider = Plan

    Fallback negotiation:
        Need JSON → Grammar modifier unavailable → Provider JSON mode → Use provider
        Need LoRA → LoRA unavailable → Prompt modifier → Fallback
        Need streaming → Provider supports → Enable
    """

    def __init__(self) -> None:
        self._fallback_chains: dict[str, list[str]] = {
            # If LoRA is unavailable, try prompt modifier, then none
            "lora": ["prompt", "none"],
            # If grammar is unavailable, try provider JSON mode
            "grammar": ["json_mode", "none"],
        }

    async def resolve(
        self,
        request: ExecutionRequest,
        policy: GenerationPolicy,
        runtime_registry: RuntimeRegistry,
        provider_registry: ProviderRegistry,
    ) -> dict[str, Any]:
        """
        Resolve the best execution configuration.

        Returns a dict of resolved values suitable for building
        an ExecutionPlan:
            {
                "runtime": "text",
                "provider": "local",
                "model_id": "gemma-3n",
                "modifiers": [...],
                "capabilities_used": [...],
                "streaming": True,
            }
        """

        # 1. Get system state
        system_state = await self._get_system_state(runtime_registry, provider_registry)

        # 2. Evaluate policy to get preferred modifiers
        preferred_modifiers = policy.resolve_modifiers(system_state)

        # 3. Find best runtime
        runtime = await self._resolve_runtime(request, runtime_registry)

        # 4. Find best provider
        provider = await self._resolve_provider(request, provider_registry, system_state)

        # 5. Check capabilities and apply fallbacks
        resolved_modifiers = await self._resolve_modifiers_with_fallback(
            preferred_modifiers, runtime, provider
        )

        # 6. Determine resolved capabilities
        capabilities_used = [
            cap
            for cap in request.constraints.required_capabilities
            if cap in self._get_all_capabilities(runtime, provider)
        ]

        return {
            "runtime": runtime,
            "provider": provider,
            "modifiers": resolved_modifiers,
            "capabilities_used": tuple(capabilities_used),
            "streaming": request.constraints.require_streaming,
        }

    async def _get_system_state(
        self,
        runtime_registry: RuntimeRegistry,
        provider_registry: ProviderRegistry,
    ) -> SystemState:
        """Build a snapshot of current system resources."""
        from hbllm.execution.policy import SystemState

        available_providers = provider_registry.list_available()
        return SystemState(
            active_provider=available_providers[0] if available_providers else "local",
            loaded_modifiers=[],
            network_available=True,
        )

    async def _resolve_runtime(
        self,
        request: ExecutionRequest,
        registry: RuntimeRegistry,
    ) -> str:
        """Find the best runtime for this task type."""
        runtime = registry.resolve(request.task_type)
        if runtime is not None:
            return runtime.runtime_type
        return "text"  # Default fallback

    async def _resolve_provider(
        self,
        request: ExecutionRequest,
        registry: ProviderRegistry,
        system_state: Any,
    ) -> str:
        """Find the best provider."""
        available = registry.list_available()
        if not available:
            return "local"
        return available[0]

    async def _resolve_modifiers_with_fallback(
        self,
        preferred: list[str],
        runtime: str,
        provider: str,
    ) -> list[str]:
        """Resolve modifiers with fallback chains."""
        resolved: list[str] = []
        for mod in preferred:
            if mod == "none":
                continue
            # Check if modifier is available, otherwise try fallbacks
            chain = self._fallback_chains.get(mod, [mod])
            for fallback in [mod, *chain]:
                if fallback == "none":
                    break
                # In a real implementation, check if the modifier is loaded
                resolved.append(fallback)
                break
        return resolved

    def _get_all_capabilities(self, runtime: str, provider: str) -> set[str]:
        """Get the union of all available capabilities."""
        # Placeholder — real implementation queries registries
        return {"streaming", "json_mode", "tool_calls"}
