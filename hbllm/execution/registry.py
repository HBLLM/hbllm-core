"""
Runtime & Provider Registries — plugin-based, discoverable.

New modalities are just registrations — no if/elif chains.
Installing a provider automatically registers itself via the plugin system.

    runtime = registry.resolve(TaskType.TEXT_GENERATION)
    provider = provider_registry.resolve("openai")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from hbllm.execution.capability import RuntimeCapabilities
    from hbllm.execution.plan import ExecutionPlan, TaskType
    from hbllm.execution.result import ExecutionResult

logger = logging.getLogger(__name__)


# ── Provider Protocol ─────────────────────────────────────────────────────────


@runtime_checkable
class LLMProvider(Protocol):
    """
    Protocol for model providers.

    Abstracts: OpenAI, Anthropic, llama.cpp, Ollama, vLLM,
    TensorRT-LLM, ONNX, and any future provider.

    Runtimes and modifiers don't care who owns the model.
    """

    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> RuntimeCapabilities: ...

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs: Any,
    ) -> dict[str, Any]: ...

    async def is_available(self) -> bool: ...


# ── Runtime Protocol ──────────────────────────────────────────────────────────


@runtime_checkable
class BaseRuntime(Protocol):
    """
    Protocol for execution runtimes.

    Each runtime handles one family of task types.
    TextRuntime handles text generation.
    VisionRuntime handles image generation. Etc.
    """

    @property
    def runtime_type(self) -> str: ...

    @property
    def capabilities(self) -> RuntimeCapabilities: ...

    def supported_task_types(self) -> list[TaskType]: ...

    async def execute(self, plan: ExecutionPlan) -> ExecutionResult: ...

    async def is_available(self) -> bool: ...


# ── Runtime Registry ──────────────────────────────────────────────────────────


class RuntimeRegistry:
    """
    Pluggable registry of execution runtimes.

    Usage:
        registry.register(text_runtime)
        runtime = registry.resolve(TaskType.TEXT_GENERATION)

    New modalities are just registrations — no if/elif chains.
    """

    def __init__(self) -> None:
        self._runtimes: dict[str, BaseRuntime] = {}
        self._task_type_index: dict[str, str] = {}  # TaskType value → runtime_type

    def register(self, runtime: BaseRuntime) -> None:
        """Register a runtime and index its supported task types."""
        rt = runtime.runtime_type
        self._runtimes[rt] = runtime
        for task_type in runtime.supported_task_types():
            self._task_type_index[task_type.value] = rt
        logger.info(
            "Registered runtime '%s' for task types: %s",
            rt,
            [t.value for t in runtime.supported_task_types()],
        )

    def resolve(self, task_type: TaskType) -> BaseRuntime | None:
        """Find the best runtime for this task type."""
        rt = self._task_type_index.get(task_type.value)
        if rt is None:
            logger.warning("No runtime registered for task type: %s", task_type)
            return None
        return self._runtimes.get(rt)

    def get(self, runtime_type: str) -> BaseRuntime | None:
        """Get a runtime by type name."""
        return self._runtimes.get(runtime_type)

    def list_available(self) -> list[str]:
        """List all registered runtime type names."""
        return list(self._runtimes.keys())

    def __contains__(self, runtime_type: str) -> bool:
        return runtime_type in self._runtimes


# ── Provider Registry ─────────────────────────────────────────────────────────


class ProviderRegistry:
    """
    Pluggable registry of model providers.

    Abstracts: OpenAI, Anthropic, llama.cpp, Ollama, vLLM,
    TensorRT-LLM, ONNX, and any future provider.

    Usage:
        registry.register(openai_provider)
        provider = registry.resolve("openai")
    """

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}

    def register(self, provider: LLMProvider) -> None:
        """Register a provider."""
        self._providers[provider.name] = provider
        logger.info("Registered provider: %s", provider.name)

    def resolve(self, provider_name: str) -> LLMProvider | None:
        """Get a provider by name."""
        return self._providers.get(provider_name)

    def list_available(self) -> list[str]:
        """List all registered provider names."""
        return list(self._providers.keys())

    async def healthcheck(self) -> dict[str, bool]:
        """Check availability of all providers."""
        results: dict[str, bool] = {}
        for name, provider in self._providers.items():
            try:
                results[name] = await provider.is_available()
            except Exception:
                results[name] = False
        return results

    def __contains__(self, provider_name: str) -> bool:
        return provider_name in self._providers
