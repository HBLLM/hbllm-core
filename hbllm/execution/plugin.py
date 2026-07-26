"""
Execution Plugin System — auto-registration for runtimes, providers, modifiers.

Installing a provider/runtime/modifier auto-registers it.
Plugins are discovered and loaded during Brain startup.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from hbllm.execution.registry import BaseRuntime, LLMProvider
    from hbllm.execution.text.modifiers.modifier import GenerationModifier

logger = logging.getLogger(__name__)


class PluginContext:
    """
    Context provided to plugins during registration.

    Plugins use this to register their runtimes, providers,
    and modifiers with the Execution OS.
    """

    def __init__(self) -> None:
        self._runtimes: list[Any] = []
        self._providers: list[Any] = []
        self._modifiers: list[Any] = []

    def register_runtime(self, runtime: BaseRuntime) -> None:
        """Register an execution runtime."""
        self._runtimes.append(runtime)
        logger.info("Plugin registered runtime: %s", runtime.runtime_type)

    def register_provider(self, provider: LLMProvider) -> None:
        """Register a model provider."""
        self._providers.append(provider)
        logger.info("Plugin registered provider: %s", provider.name)

    def register_modifier(self, modifier: GenerationModifier) -> None:
        """Register a generation modifier."""
        self._modifiers.append(modifier)
        logger.info("Plugin registered modifier: %s", modifier.name)

    @property
    def runtimes(self) -> list[Any]:
        return list(self._runtimes)

    @property
    def providers(self) -> list[Any]:
        return list(self._providers)

    @property
    def modifiers(self) -> list[Any]:
        return list(self._modifiers)


@runtime_checkable
class ExecutionPlugin(Protocol):
    """
    Plugin interface for extending the Execution OS.

    Implementations register their capabilities during
    the ``register`` call. The plugin system discovers
    and loads plugins during Brain startup.
    """

    @property
    def name(self) -> str:
        """Unique plugin name."""
        ...

    @property
    def version(self) -> str:
        """Plugin version string."""
        ...

    async def register(self, ctx: PluginContext) -> None:
        """
        Register capabilities with the Execution OS.

        Example:
            async def register(self, ctx):
                ctx.register_provider(MyProvider())
                ctx.register_modifier(MyModifier())
        """
        ...
