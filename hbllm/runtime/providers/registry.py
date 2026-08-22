"""Provider Registry — runtime discovery and lifecycle management.

Queries by CAPABILITY, not model name::

    registry.find_capability("transcribe_speech")  # → [WhisperProvider]
    registry.find_capability("visual_question_answering")  # → [VLMProvider]
    registry.find_by_modality("audio")  # → [WhisperProvider, YAMNetProvider]

The registry is the central index for all perception, cognition,
and action providers.  It enables the Cognitive Budget Engine to
select the best available cognitive capability without hardcoding
model names.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.runtime.providers.capability import ProviderCapability

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Runtime discovery and lifecycle management for all providers.

    Providers register with a ``ProviderCapability`` manifest.
    The registry indexes providers by capability, modality, and type,
    enabling capability-first discovery.

    Thread-safety note: This is designed for single-threaded async
    usage.  If concurrent registration is needed, add a lock.

    Usage::

        registry = ProviderRegistry()

        # Register providers
        registry.register(whisper_provider, whisper_capability)
        registry.register(yolo_provider, yolo_capability)

        # Find by capability
        providers = registry.find_capability("transcribe_speech")

        # Find by modality
        audio_providers = registry.find_by_modality("audio")

        # Check availability
        if registry.is_available("transcribe_speech"):
            provider = registry.find_capability("transcribe_speech")[0]
    """

    def __init__(self) -> None:
        self._providers: dict[str, Any] = {}
        self._capabilities: dict[str, ProviderCapability] = {}

        # Indexes for fast lookup
        self._by_capability: dict[str, list[str]] = {}  # capability → [provider_ids]
        self._by_modality: dict[str, list[str]] = {}  # modality → [provider_ids]
        self._by_type: dict[str, list[str]] = {}  # type → [provider_ids]

    # ── Registration ─────────────────────────────────────────────────────

    def register(
        self,
        provider: Any,
        capability: ProviderCapability,
    ) -> None:
        """Register a provider with its capability manifest.

        Args:
            provider: The provider instance.
            capability: Declarative capability manifest.

        Raises:
            ValueError: If provider_id is already registered.
        """
        pid = capability.provider_id
        if pid in self._providers:
            raise ValueError(
                f"Provider '{pid}' is already registered. Unregister it first to replace."
            )

        self._providers[pid] = provider
        self._capabilities[pid] = capability

        # Index by capability
        for cap in capability.capabilities:
            self._by_capability.setdefault(cap, []).append(pid)

        # Index by modality
        for mod in capability.modalities:
            self._by_modality.setdefault(mod, []).append(pid)

        # Index by type
        self._by_type.setdefault(capability.provider_type, []).append(pid)

        logger.info(
            "Registered provider '%s' (type=%s, capabilities=%s, modalities=%s)",
            pid,
            capability.provider_type,
            capability.capabilities,
            capability.modalities,
        )

    def unregister(self, provider_id: str) -> None:
        """Remove a provider from the registry.

        Args:
            provider_id: ID of the provider to remove.

        Raises:
            KeyError: If provider_id is not registered.
        """
        if provider_id not in self._providers:
            raise KeyError(f"Provider '{provider_id}' is not registered.")

        capability = self._capabilities[provider_id]

        # Remove from indexes
        for cap in capability.capabilities:
            if cap in self._by_capability:
                self._by_capability[cap] = [p for p in self._by_capability[cap] if p != provider_id]
                if not self._by_capability[cap]:
                    del self._by_capability[cap]

        for mod in capability.modalities:
            if mod in self._by_modality:
                self._by_modality[mod] = [p for p in self._by_modality[mod] if p != provider_id]
                if not self._by_modality[mod]:
                    del self._by_modality[mod]

        ptype = capability.provider_type
        if ptype in self._by_type:
            self._by_type[ptype] = [p for p in self._by_type[ptype] if p != provider_id]
            if not self._by_type[ptype]:
                del self._by_type[ptype]

        del self._providers[provider_id]
        del self._capabilities[provider_id]

        logger.info("Unregistered provider '%s'", provider_id)

    # ── Discovery ────────────────────────────────────────────────────────

    def find_capability(self, capability: str) -> list[Any]:
        """Find all providers that support a specific capability.

        Args:
            capability: The capability to search for
                (e.g., ``"transcribe_speech"``).

        Returns:
            List of provider instances (may be empty).
        """
        pids = self._by_capability.get(capability, [])
        return [self._providers[pid] for pid in pids]

    def find_by_modality(self, modality: str) -> list[Any]:
        """Find all providers that support a specific modality.

        Args:
            modality: The modality to search for (e.g., ``"audio"``).

        Returns:
            List of provider instances (may be empty).
        """
        pids = self._by_modality.get(modality, [])
        return [self._providers[pid] for pid in pids]

    def find_by_type(self, provider_type: str) -> list[Any]:
        """Find all providers of a specific type.

        Args:
            provider_type: Provider type (``"perception"``,
                ``"cognition"``, ``"action"``).

        Returns:
            List of provider instances (may be empty).
        """
        pids = self._by_type.get(provider_type, [])
        return [self._providers[pid] for pid in pids]

    def get_provider(self, provider_id: str) -> Any:
        """Get a specific provider by ID.

        Raises:
            KeyError: If provider_id is not registered.
        """
        if provider_id not in self._providers:
            raise KeyError(f"Provider '{provider_id}' is not registered.")
        return self._providers[provider_id]

    def get_capability(self, provider_id: str) -> ProviderCapability:
        """Get the capability manifest for a provider.

        Raises:
            KeyError: If provider_id is not registered.
        """
        if provider_id not in self._capabilities:
            raise KeyError(f"Provider '{provider_id}' is not registered.")
        return self._capabilities[provider_id]

    # ── Queries ──────────────────────────────────────────────────────────

    def is_available(self, capability: str) -> bool:
        """Check if any provider supports a capability."""
        return capability in self._by_capability and bool(self._by_capability[capability])

    def all_capabilities(self) -> list[ProviderCapability]:
        """Return all registered capability manifests."""
        return list(self._capabilities.values())

    def all_provider_ids(self) -> list[str]:
        """Return all registered provider IDs."""
        return list(self._providers.keys())

    @property
    def provider_count(self) -> int:
        """Number of registered providers."""
        return len(self._providers)

    def find_best_for(
        self,
        capability: str,
        *,
        prefer_low_latency: bool = False,
        prefer_high_quality: bool = False,
        require_local: bool = False,
    ) -> Any | None:
        """Find the best provider for a capability given preferences.

        Simple heuristic selection.  The ``CognitiveBudgetEngine``
        provides more sophisticated multi-factor selection.

        Args:
            capability: Required capability.
            prefer_low_latency: Prefer faster providers.
            prefer_high_quality: Prefer higher quality providers.
            require_local: Exclude providers that require network.

        Returns:
            Best provider instance, or ``None`` if no match.
        """
        pids = self._by_capability.get(capability, [])
        if not pids:
            return None

        candidates: list[tuple[str, ProviderCapability]] = [
            (pid, self._capabilities[pid]) for pid in pids
        ]

        # Filter: require local
        if require_local:
            candidates = [(pid, cap) for pid, cap in candidates if not cap.requires_network]

        if not candidates:
            return None

        # Sort by preference
        latency_order = {
            "very_low": 0,
            "low": 1,
            "medium": 2,
            "high": 3,
            "variable": 4,
        }
        quality_order = {
            "very_high": 0,
            "high": 1,
            "medium": 2,
            "low": 3,
        }

        def sort_key(item: tuple[str, ProviderCapability]) -> tuple[int, int]:
            _, cap = item
            lat = latency_order.get(cap.latency_profile, 2)
            qual = quality_order.get(cap.quality_profile, 2)
            if prefer_low_latency:
                return (lat, qual)
            if prefer_high_quality:
                return (qual, lat)
            return (lat + qual, 0)

        candidates.sort(key=sort_key)
        best_pid = candidates[0][0]
        return self._providers[best_pid]
