"""
LLM Provider Abstraction — unified interface for local and external models.

Supports:
  - LocalProvider: uses HBLLMForCausalLM + tokenizer
  - OpenAIProvider: calls OpenAI API (GPT-4o-mini, etc.)
  - AnthropicProvider: calls Anthropic API (Claude)

Use get_provider() to create the appropriate provider:
    provider = get_provider('openai')  # or 'anthropic' or 'local'
    response = await provider.generate(messages)
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    content: str
    model: str
    usage: dict[str, int]  # prompt_tokens, completion_tokens, total_tokens
    finish_reason: str = "stop"
    raw: Any = None  # Original provider response


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response tokens."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        ...


# ─── Local Provider ──────────────────────────────────────────────────────────

class LocalProvider(LLMProvider):
    """Uses the local HBLLM transformer for generation."""

    def __init__(self, model: Any = None, tokenizer: Any = None, device: str = "auto"):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    def _ensure_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "Local model not loaded. Provide a model at init or use an external provider."
            )

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> LLMResponse:
        self._ensure_loaded()
        import torch

        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        input_ids = self._tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        if hasattr(self._model, "device"):
            input_tensor = input_tensor.to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                input_ids=input_tensor,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self._tokenizer.eos_id,
            )

        # Decode only new tokens
        new_tokens = output[0][len(input_ids):].tolist()
        content = self._tokenizer.decode(new_tokens)

        return LLMResponse(
            content=content,
            model="hbllm-local",
            usage={
                "prompt_tokens": len(input_ids),
                "completion_tokens": len(new_tokens),
                "total_tokens": len(input_ids) + len(new_tokens),
            },
        )

    async def stream(self, messages, max_tokens=1024, temperature=0.7, **kwargs):
        # Local model doesn't support true streaming yet — yield full response
        response = await self.generate(messages, max_tokens, temperature, **kwargs)
        yield response.content

    @property
    def name(self) -> str:
        return "local"


# ─── OpenAI Provider ─────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    """Calls OpenAI-compatible APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._model = model
        self._base_url = base_url or "https://api.openai.com/v1"

        if not self._api_key:
            logger.warning("OPENAI_API_KEY not set. OpenAI provider will fail.")

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> LLMResponse:
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return LLMResponse(
            content=choice["message"]["content"],
            model=data.get("model", self._model),
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        import httpx

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        chunk = line[6:]
                        if chunk.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(chunk)
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError):
                            continue

    @property
    def name(self) -> str:
        return f"openai/{self._model}"


# ─── Anthropic Provider ──────────────────────────────────────────────────────

class AnthropicProvider(LLMProvider):
    """Calls Anthropic Claude API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._model = model

        if not self._api_key:
            logger.warning("ANTHROPIC_API_KEY not set. Anthropic provider will fail.")

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> LLMResponse:
        import httpx

        # Anthropic separates system message
        system = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_messages.append(m)

        headers = {
            "x-api-key": self._api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self._model,
            "messages": user_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["content"][0]["text"] if data.get("content") else ""
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=data.get("model", self._model),
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
            finish_reason=data.get("stop_reason", "stop"),
            raw=data,
        )

    async def stream(self, messages, max_tokens=1024, temperature=0.7, **kwargs):
        # Simplified — yield full response
        response = await self.generate(messages, max_tokens, temperature, **kwargs)
        yield response.content

    @property
    def name(self) -> str:
        return f"anthropic/{self._model}"


# ─── Registry ────────────────────────────────────────────────────────────────

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "local": LocalProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def get_provider(
    name: str = "openai",
    **kwargs: Any,
) -> LLMProvider:
    """
    Get an LLM provider by name.

    Args:
        name: Provider name ('local', 'openai', 'anthropic')
        **kwargs: Provider-specific arguments

    Returns:
        LLMProvider instance
    """
    # Support 'openai/gpt-4o' syntax
    if "/" in name:
        provider_name, model = name.split("/", 1)
        kwargs.setdefault("model", model)
    else:
        provider_name = name

    if provider_name not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available: {list(_PROVIDERS.keys())}"
        )

    return _PROVIDERS[provider_name](**kwargs)


def register_provider(name: str, cls: type[LLMProvider]) -> None:
    """Register a custom provider."""
    _PROVIDERS[name] = cls
    logger.info("Registered LLM provider: %s", name)
