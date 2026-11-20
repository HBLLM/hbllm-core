"""
KV Prefix Cache — pre-compute and cache KV states for static prompt prefixes.

Caches the ``past_key_values`` from a forward pass over the full identity
composite (system prompt + identity + tools + persona) so that subsequent
inference requests skip re-evaluating the static prefix tokens.

This drops first-token latency from seconds to milliseconds for repeated
conversations with the same identity configuration.

Invalidation triggers:
    - LoRA adapter swap (KV states change with weights)
    - Tool schema change (tool descriptions are part of prefix)
    - Persona update (persona block changes)
    - Model reload (everything invalidated)

Per-tenant: each ``IdentityNode`` gets its own cached prefix keyed by
``PrefixIdentity.cache_key``.

Usage::

    from hbllm.model.kv_warmup import KVPrefixCache, PrefixIdentity

    identity = PrefixIdentity(
        system_prompt="You are HBLLM...",
        identity_prompt="Name: Jarvis...",
        tool_schemas="[{\"name\": \"web_search\", ...}]",
        persona_block="Formal, analytical...",
    )
    cache = KVPrefixCache()
    cached = cache.warmup(identity, model, tokenizer)

    # Later: use cached prefix for generation
    prefix = cache.get_cached(identity.cache_key)
    if prefix:
        # Feed prefix.past_key_values directly to model.forward()
        out = model.forward(
            input_ids=user_tokens,
            past_key_values=prefix.past_key_values,
            use_cache=True,
        )
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# PrefixIdentity — the complete static prefix that gets cached
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PrefixIdentity:
    """The full identity composite whose KV states are cached.

    Includes everything that forms the static prefix of a conversation:
    system prompt, identity configuration, tool schemas, and persona block.

    The ``cache_key`` is a SHA-256 hash of the full composite, so any
    change to any component automatically invalidates the cache.
    """

    system_prompt: str
    identity_prompt: str = ""
    tool_schemas: str = ""
    persona_block: str = ""

    @property
    def cache_key(self) -> str:
        """SHA-256 of the full composite for cache keying."""
        composite = (
            self.system_prompt + self.identity_prompt + self.tool_schemas + self.persona_block
        )
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()

    @property
    def full_text(self) -> str:
        """The concatenated prefix text for tokenization."""
        parts = [self.system_prompt]
        if self.identity_prompt:
            parts.append(self.identity_prompt)
        if self.tool_schemas:
            parts.append(self.tool_schemas)
        if self.persona_block:
            parts.append(self.persona_block)
        return "\n\n".join(parts)

    def __repr__(self) -> str:
        return f"PrefixIdentity(key={self.cache_key[:12]}..., len={len(self.full_text)} chars)"


# ═══════════════════════════════════════════════════════════════════════════
# CachedKV — the cached result of a warmup pass
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CachedKV:
    """Cached KV states from a prefix warmup pass.

    Attributes:
        cache_key: SHA-256 hash of the prefix that produced these KV states.
        past_key_values: The actual KV cache tensors from the forward pass.
            Format: list of (key, value) tuples per layer.
        prefix_length: Number of tokens in the cached prefix.
        created_at: Timestamp when this cache entry was created.
        device: Device the tensors reside on.
    """

    cache_key: str
    past_key_values: Any  # list[tuple[Tensor, Tensor]]
    prefix_length: int
    created_at: float = field(default_factory=time.time)
    device: str = "cpu"

    def clone(self) -> CachedKV:
        """Deep-clone the KV tensors for safe reuse across requests.

        Each inference request needs its own copy because the model
        appends to the KV cache during generation.
        """
        cloned_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for k, v in self.past_key_values:
            cloned_kv.append((k.clone(), v.clone()))

        return CachedKV(
            cache_key=self.cache_key,
            past_key_values=cloned_kv,
            prefix_length=self.prefix_length,
            created_at=self.created_at,
            device=self.device,
        )

    @property
    def age_seconds(self) -> float:
        """How old this cache entry is in seconds."""
        return time.time() - self.created_at

    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage of the cached KV tensors."""
        total = 0
        if self.past_key_values:
            for k, v in self.past_key_values:
                total += k.nelement() * k.element_size()
                total += v.nelement() * v.element_size()
        return total


# ═══════════════════════════════════════════════════════════════════════════
# KVPrefixCache — the cache manager
# ═══════════════════════════════════════════════════════════════════════════


class KVPrefixCache:
    """Manages cached KV states for static prompt prefixes.

    Supports multiple concurrent cache entries (one per tenant/identity).
    Each entry is keyed by the SHA-256 hash of the full prefix composite.

    Thread safety: this class is designed for single-threaded async usage.
    For multi-threaded scenarios, external locking is required.

    Args:
        max_entries: Maximum number of cached prefixes to keep.
            Oldest entries are evicted when the limit is reached.
        max_age_seconds: Maximum age of a cache entry before auto-eviction.
            Set to 0 to disable age-based eviction.
    """

    def __init__(
        self,
        max_entries: int = 16,
        max_age_seconds: float = 3600.0,
    ) -> None:
        self._cache: dict[str, CachedKV] = {}
        self._max_entries = max_entries
        self._max_age_seconds = max_age_seconds
        self._stats = _CacheStats()

    @torch.no_grad()
    def warmup(
        self,
        identity: PrefixIdentity,
        model: Any,
        tokenizer: Any,
    ) -> CachedKV:
        """Pre-compute KV states for a prefix identity.

        Tokenizes the full prefix text, runs a single forward pass with
        ``use_cache=True``, and stores the resulting ``past_key_values``.

        Args:
            identity: The prefix identity to warm up.
            model: An ``HBLLMForCausalLM`` or compatible model with a
                ``forward(input_ids, use_cache=True)`` method.
            tokenizer: A tokenizer with ``encode()`` method.

        Returns:
            The cached KV states.
        """
        cache_key = identity.cache_key

        # Check if already cached and still valid
        existing = self.get_cached(cache_key)
        if existing is not None:
            logger.debug(
                "KV prefix cache hit for key %s (age=%.1fs)",
                cache_key[:12],
                existing.age_seconds,
            )
            self._stats.hits += 1
            return existing

        # Tokenize the prefix
        start_time = time.time()
        prefix_text = identity.full_text

        if hasattr(tokenizer, "encode"):
            token_ids = tokenizer.encode(prefix_text, return_tensors="pt")
            if isinstance(token_ids, list):
                token_ids = torch.tensor([token_ids], dtype=torch.long)
        else:
            # Fallback for simple tokenizers
            token_ids = tokenizer(prefix_text, return_tensors="pt")["input_ids"]

        # Move to model device
        device = next(model.parameters()).device
        token_ids = token_ids.to(device)
        prefix_length = token_ids.shape[1]

        # Forward pass to generate KV cache
        model.eval()
        output = model.forward(
            input_ids=token_ids,
            use_cache=True,
        )

        past_key_values = output.get("past_key_values")
        if past_key_values is None:
            logger.warning("Model did not return past_key_values. KV warmup failed.")
            self._stats.failures += 1
            return CachedKV(
                cache_key=cache_key,
                past_key_values=None,
                prefix_length=prefix_length,
                device=str(device),
            )

        # Detach tensors from computation graph
        detached_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for k, v in past_key_values:
            detached_kv.append((k.detach(), v.detach()))

        cached = CachedKV(
            cache_key=cache_key,
            past_key_values=detached_kv,
            prefix_length=prefix_length,
            device=str(device),
        )

        # Store in cache (with eviction if needed)
        self._evict_if_needed()
        self._cache[cache_key] = cached
        self._stats.misses += 1

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            "KV prefix warmup: %d tokens cached in %.1fms (key=%s, mem=%.1fMB)",
            prefix_length,
            elapsed_ms,
            cache_key[:12],
            cached.memory_bytes / (1024 * 1024),
        )

        return cached

    def get_cached(self, cache_key: str) -> CachedKV | None:
        """Retrieve cached KV states for a given prefix hash.

        Returns a **clone** of the cached KV tensors so the caller
        can safely append to them during generation without corrupting
        the cached copy.

        Returns None on cache miss or if the entry has expired.
        """
        entry = self._cache.get(cache_key)
        if entry is None:
            return None

        # Check age-based expiry
        if self._max_age_seconds > 0 and entry.age_seconds > self._max_age_seconds:
            del self._cache[cache_key]
            self._stats.evictions += 1
            logger.debug(
                "KV cache entry expired: key=%s age=%.0fs",
                cache_key[:12],
                entry.age_seconds,
            )
            return None

        if entry.past_key_values is None:
            return None

        # Return a clone so caller can modify without corrupting cache
        self._stats.hits += 1
        return entry.clone()

    def invalidate(self, cache_key: str) -> bool:
        """Remove a specific cache entry.

        Call this when the identity composite changes (LoRA swap,
        tool change, persona update).

        Returns True if an entry was removed.
        """
        if cache_key in self._cache:
            del self._cache[cache_key]
            self._stats.invalidations += 1
            logger.info("KV cache invalidated: key=%s", cache_key[:12])
            return True
        return False

    def invalidate_all(self) -> int:
        """Clear all cache entries.

        Call this on model reload.

        Returns the number of entries cleared.
        """
        count = len(self._cache)
        self._cache.clear()
        self._stats.invalidations += count
        logger.info("KV cache fully cleared: %d entries removed", count)
        return count

    @property
    def size(self) -> int:
        """Number of entries currently in the cache."""
        return len(self._cache)

    @property
    def total_memory_bytes(self) -> int:
        """Total memory usage of all cached KV tensors."""
        return sum(e.memory_bytes for e in self._cache.values())

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "entries": self.size,
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "evictions": self._stats.evictions,
            "invalidations": self._stats.invalidations,
            "failures": self._stats.failures,
            "total_memory_mb": self.total_memory_bytes / (1024 * 1024),
            "hit_rate": (self._stats.hits / max(1, self._stats.hits + self._stats.misses)),
        }

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if we're at capacity."""
        while len(self._cache) >= self._max_entries:
            # Find oldest entry
            oldest_key = min(self._cache, key=lambda k: self._cache[k].created_at)
            del self._cache[oldest_key]
            self._stats.evictions += 1
            logger.debug("KV cache evicted oldest entry: key=%s", oldest_key[:12])


@dataclass
class _CacheStats:
    """Internal cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0
    failures: int = 0
