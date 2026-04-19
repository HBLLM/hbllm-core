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

import asyncio
import json
import logging
import os
import random
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from hbllm.network.tracing import trace_span

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    content: str
    model: str
    usage: dict[str, int]  # prompt_tokens, completion_tokens, total_tokens
    finish_reason: str = "stop"
    raw: Any = None  # Original provider response


# ─── Retry Helper ─────────────────────────────────────────────────────────────

# HTTP status codes that indicate transient failures worth retrying
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


async def _retry_api_call(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    **kwargs: Any,
) -> Any:
    """
    Execute an async callable with exponential backoff on transient failures.

    Retries on:
      - httpx.HTTPStatusError with status in _RETRYABLE_STATUS_CODES
      - httpx.TransportError (connection reset, DNS failure, timeout)

    Backoff formula: delay = initial_delay * 2^attempt + jitter
    """
    import httpx

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            if e.response.status_code not in _RETRYABLE_STATUS_CODES:
                raise  # Non-retryable (e.g. 401, 403, 404)
            if attempt == max_retries:
                raise
            delay = min(initial_delay * (2**attempt), max_delay)
            delay += random.uniform(0, delay * 0.1)  # jitter
            logger.warning(
                "API call failed (HTTP %d), retrying in %.1fs (attempt %d/%d)",
                e.response.status_code,
                delay,
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(delay)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            if attempt == max_retries:
                raise
            delay = min(initial_delay * (2**attempt), max_delay)
            delay += random.uniform(0, delay * 0.1)
            logger.warning(
                "API transport error (%s), retrying in %.1fs (attempt %d/%d)",
                type(e).__name__,
                delay,
                attempt + 1,
                max_retries,
            )
            await asyncio.sleep(delay)


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
    def stream(
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
        self._draft_model: Any = None

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

        with trace_span(
            "llm.generate",
            {"provider": "local", "model": "hbllm-local", "max_tokens": str(max_tokens)},
        ):
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
            new_tokens = output[0][len(input_ids) :].tolist()
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

    async def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream tokens from the local model one at a time, using Speculative Decoding if configured."""
        self._ensure_loaded()
        import torch

        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        input_ids = self._tokenizer.encode(prompt, add_bos=True)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)

        if hasattr(self._model, "device"):
            input_tensor = input_tensor.to(self._model.device)

        top_p = kwargs.get("top_p", 0.9)
        eos_id = self._tokenizer.eos_id

        # Phase 4: Speculative Decoding Integration
        has_draft_model = hasattr(self, "_draft_model") and self._draft_model is not None

        if has_draft_model:
            from hbllm.model.speculative import speculate_step

            K = kwargs.get("speculative_k", 4)
            current_input = input_tensor

            tokens_generated = 0
            while tokens_generated < max_tokens:
                accepted_tokens, _, _ = speculate_step(
                    main_model=self._model,
                    draft_model=self._draft_model,
                    draft_input_ids=current_input,
                    main_input_ids=current_input,
                    K=K,
                    temperature=temperature,
                    top_p=top_p,
                )

                # Yield newly accepted tokens as chunks
                for token_id in accepted_tokens[0].tolist():
                    if token_id == eos_id:
                        return
                    yield self._tokenizer.decode([token_id])
                    tokens_generated += 1

                current_input = torch.cat(
                    [current_input, accepted_tokens.to(current_input.device)], dim=1
                )

        else:
            # Traditional Autoregressive Loop (with KV Cache)
            past_key_values = None
            if hasattr(self._model, "config"):
                from hbllm.serving.kv_cache import KVCache

                cfg = self._model.config
                budget = input_tensor.shape[1] + max_tokens
                # Initialize array of KVCaches (one per layer) if safe memory budget
                if budget <= getattr(cfg, "max_position_embeddings", 8192):
                    past_key_values = [
                        KVCache(
                            batch_size=1,
                            max_seq_len=budget,
                            num_kv_heads=cfg.num_kv_heads,
                            head_dim=cfg.hidden_size // cfg.num_attention_heads,
                            dtype=next(self._model.parameters()).dtype,
                            device=input_tensor.device,
                            sliding_window=getattr(cfg, "sliding_window", None),
                            attention_sinks=getattr(cfg, "attention_sinks", 4),
                        )
                        for _ in range(cfg.num_layers)
                    ]

            with torch.no_grad():
                next_token: Any = None
                for step in range(max_tokens):
                    # On step 0, process the full prompt; afterwards only the new token
                    if step == 0:
                        model_input = input_tensor
                    else:
                        model_input = next_token  # [1, 1] — single new token

                    output = self._model(
                        model_input,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )

                    if isinstance(output, dict):
                        logits = output["logits"]
                        past_key_values = output.get("past_key_values", past_key_values)
                    else:
                        logits = output

                    next_logits = logits[:, -1, :] / max(temperature, 1e-7)

                    # Top-p (nucleus) sampling
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[mask] = float("-inf")
                    probs = torch.softmax(sorted_logits, dim=-1)
                    next_index = torch.multinomial(probs, num_samples=1)
                    next_token = sorted_indices.gather(-1, next_index)

                    token_id = next_token.item()
                    if token_id == eos_id:
                        break

                    token_text = self._tokenizer.decode([token_id])
                    yield token_text

    @property
    def name(self) -> str:
        return "local"

    def load_lora_from_disk(
        self,
        lora_path: str,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ) -> None:
        """Dynamically load and activate a LoRA adapter from disk."""
        self._ensure_loaded()
        import os

        import torch

        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA adapter not found at {lora_path}")

        logger.info("Loading LoRA adapter from %s", lora_path)
        state_dict = torch.load(lora_path, map_location=self._device, weights_only=True)

        # Dispatch to the model
        if hasattr(self._model, "load_lora_adapter"):
            self._model.load_lora_adapter(
                state_dict, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
            # Ensure it's active
            self._model.set_lora_active(True)
        else:
            raise NotImplementedError("Model does not support load_lora_adapter")

    def set_lora_active(self, active: bool = True) -> None:
        """Toggle LoRA active state without unloading weights."""
        self._ensure_loaded()
        if hasattr(self._model, "set_lora_active"):
            self._model.set_lora_active(active)
            logger.info("LocalProvider LoRA active=%s", active)

    def load_draft_model(self, draft_model: Any) -> None:
        """
        Load a smaller draft model to enable Speculative Decoding.
        The draft model generates token proposals which the main model verifies in parallel.
        """
        self._ensure_loaded()
        if not hasattr(draft_model, "generate"):
            raise ValueError(
                "Draft model must be a valid HuggingFace/HBLLM model with a generate method."
            )
        self._draft_model = draft_model
        logger.info("LocalProvider: Speculative Decoding draft model loaded and enabled.")


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

        with trace_span(
            "llm.generate",
            {"provider": "openai", "model": self._model, "max_tokens": str(max_tokens)},
        ):
            headers: dict[str, str] = {
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

            async def _do_request() -> Any:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        f"{self._base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    resp.raise_for_status()
                    return resp.json()

            data = await _retry_api_call(_do_request)

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

        headers: dict[str, str] = {
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

        with trace_span(
            "llm.generate",
            {"provider": "anthropic", "model": self._model, "max_tokens": str(max_tokens)},
        ):
            # Anthropic separates system message
            system = ""
            user_messages = []
            for m in messages:
                if m["role"] == "system":
                    system = m["content"]
                else:
                    user_messages.append(m)

            headers: dict[str, str] = {
                "x-api-key": str(self._api_key),
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

            async def _do_request() -> Any:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers=headers,
                        json=payload,
                    )
                    resp.raise_for_status()
                    return resp.json()

            data = await _retry_api_call(_do_request)

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

    async def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream tokens from Anthropic's SSE API."""
        import httpx

        system = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_messages.append(m)

        headers: dict[str, str] = {
            "x-api-key": str(self._api_key),
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self._model,
            "messages": user_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=payload,
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:]
                    try:
                        data = json.loads(chunk)
                        event_type = data.get("type", "")
                        if event_type == "content_block_delta":
                            text = data.get("delta", {}).get("text", "")
                            if text:
                                yield text
                        elif event_type == "message_stop":
                            break
                    except (json.JSONDecodeError, KeyError):
                        continue

    @property
    def name(self) -> str:
        return f"anthropic/{self._model}"


# ─── Ollama Provider ─────────────────────────────────────────────────────────


class OllamaProvider(LLMProvider):
    """Calls local Ollama API."""

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434/api",
        **kwargs: Any,
    ):
        self._model = model
        self._base_url = base_url

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> LLMResponse:
        import httpx

        with trace_span(
            "llm.generate",
            {"provider": "ollama", "model": self._model, "max_tokens": str(max_tokens)},
        ):
            payload = {
                "model": self._model,
                "messages": messages,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                "stream": False,
            }

            async def _do_request() -> Any:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        f"{self._base_url}/chat",
                        json=payload,
                    )
                    resp.raise_for_status()
                    return resp.json()

            data = await _retry_api_call(_do_request)

        usage_info = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
        }

        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=data.get("model", self._model),
            usage=usage_info,
            finish_reason="stop",
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

        payload = {
            "model": self._model,
            "messages": messages,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/chat",
                json=payload,
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if data.get("done"):
                            break
                    except (json.JSONDecodeError, KeyError):
                        continue

    @property
    def name(self) -> str:
        return f"ollama/{self._model}"


# ─── Registry ────────────────────────────────────────────────────────────────

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "local": LocalProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
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
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(_PROVIDERS.keys())}")

    return _PROVIDERS[provider_name](**kwargs)


def register_provider(name: str, cls: type[LLMProvider]) -> None:
    """Register a custom provider."""
    _PROVIDERS[name] = cls
    logger.info("Registered LLM provider: %s", name)
