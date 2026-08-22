"""Provider Capability Declaration — HBLLM Cognitive Runtime.

Declarative capability manifests for perception, cognition, and action
providers.  The ``ProviderCapability`` is the first-class object that the
``ProviderRegistry`` indexes and the ``CognitiveBudgetEngine`` queries.

The key abstraction: the Cognitive Budget asks **"What capability do I need?"**
rather than **"Which model should I call?"**

Example::

    capability = ProviderCapability(
        provider_id="whisper_small",
        provider_type="perception",
        capabilities=["transcribe_speech", "detect_language"],
        modalities=["audio"],
        latency_profile="low",
        quality_profile="medium",
        memory_requirement_mb=500,
        hardware_requirements=["cpu"],
        requires_network=False,
    )
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProviderCapability(BaseModel):
    """Declarative capability manifest for any provider.

    First-class — not metadata.  The registry and budget engine
    query these to find the best available cognitive capability.

    Attributes:
        provider_id: Unique identifier (``"whisper_small"``,
            ``"yolo11n"``, ``"qwen3_8b"``).
        provider_type: Category (``"perception"``, ``"cognition"``,
            ``"action"``).
        capabilities: List of operations this provider can perform
            (``["transcribe_speech", "detect_language"]``).
        modalities: Sensory modalities supported
            (``["audio"]``, ``["vision", "depth"]``).
        latency_profile: Expected latency bucket
            (``"very_low"``, ``"low"``, ``"medium"``, ``"high"``,
            ``"variable"``).
        quality_profile: Expected output quality bucket
            (``"low"``, ``"medium"``, ``"high"``, ``"very_high"``).
        risk_profile: Risk level of using this provider
            (``"none"``, ``"low"``, ``"medium"``, ``"high"``).
        memory_requirement_mb: Approximate memory footprint.
        hardware_requirements: Required hardware
            (``["gpu"]``, ``["metal"]``, ``["npu"]``).
        requires_network: Whether internet is needed.
        precision: Model precision (``"int4"``, ``"fp16"``, ``"fp32"``).
        energy_cost: Relative energy cost per invocation
            (normalized ``[0.0, 1.0]``).
        monetary_cost: Monetary cost per invocation in USD.
        max_input_tokens: Maximum input size (for LLM-type providers).
        max_concurrent: Maximum concurrent invocations.
    """

    provider_id: str
    provider_type: str  # "perception", "cognition", "action"

    # Capabilities — what this provider can DO
    capabilities: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)

    # Performance profiles — for Cognitive Budget decisions
    latency_profile: str = "medium"  # very_low | low | medium | high | variable
    quality_profile: str = "medium"  # low | medium | high | very_high
    risk_profile: str = "none"  # none | low | medium | high

    # Resource requirements
    memory_requirement_mb: int = 0
    hardware_requirements: list[str] = Field(default_factory=list)

    # Operational constraints
    requires_network: bool = False
    precision: str = ""

    # Cost profiles — for budget engine
    energy_cost: float = 0.0  # Relative energy cost [0.0, 1.0]
    monetary_cost: float = 0.0  # USD per invocation

    # Capacity
    max_input_tokens: int | None = None
    max_concurrent: int = 1

    def supports_capability(self, capability: str) -> bool:
        """Check if this provider supports a specific capability."""
        return capability in self.capabilities

    def supports_modality(self, modality: str) -> bool:
        """Check if this provider supports a specific modality."""
        return modality in self.modalities
