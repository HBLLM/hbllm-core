"""
Brain Profiles — Declarative runtime capability configuration.

Instead of scattering ``if lite_mode`` checks throughout the codebase,
every subsystem queries ``profile.features`` to determine which
capabilities are available. This keeps components focused on
capabilities rather than deployment details.

Usage::

    from hbllm.config.brain_profile import BrainProfile, load_profile

    profile = load_profile("lite")
    if profile.features.local_inference:
        # Load PyTorch model
        ...
    if profile.features.snn_streams:
        # Boot SNN ComprehensionStream
        ...

Profiles::

    FullProfile       — All subsystems, local model, SNN, autonomy
    LiteProfile       — API-only inference, local memory, <250MB RAM
    EdgeProfile       — Minimal memory, edge hardware, no training
    ResearchProfile   — Full + experimental features, extra logging
    RobotProfile      — Full + embodiment, ROS2, sensor fusion
"""

from __future__ import annotations

import logging
import os
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Feature Flags
# ═══════════════════════════════════════════════════════════════════════════


class ProfileFeatures(BaseModel):
    """Capability feature flags queried by subsystems.

    Each boolean indicates whether a subsystem should be activated.
    Subsystems query this instead of checking deployment mode directly.
    """

    # ── Inference ────────────────────────────────────────────────────────
    local_inference: bool = True  # Load local PyTorch transformer model
    api_inference: bool = True  # Use external API providers (OpenAI, Anthropic, Ollama)
    rust_simd_acceleration: bool = True  # Enable Rust SIMD quantization kernels

    # ── Memory ───────────────────────────────────────────────────────────
    episodic_memory: bool = True
    semantic_memory: bool = True
    procedural_memory: bool = True
    value_memory: bool = True
    knowledge_graph: bool = True
    spatial_memory: bool = False  # Only for embodied profiles
    temporal_memory: bool = True
    importance_scoring: bool = True
    memory_consolidation: bool = True  # Background sleep cycles

    # ── Cognitive ────────────────────────────────────────────────────────
    snn_streams: bool = True  # ComprehensionStream, ExpressionStream, TrainedPRM
    autonomy_core: bool = True  # Background autonomous reasoning
    goal_decomposition: bool = True  # DAG task pursuit
    meta_cognition: bool = True  # Self-reflection and supervision
    curiosity_engine: bool = False  # Proactive exploration (research profile)
    spawner_neurogenesis: bool = True  # Dynamic LoRA adapter creation

    # ── Human Modeling ───────────────────────────────────────────────────
    user_model: bool = True
    project_graph: bool = True
    executive_cortex: bool = True
    relationship_memory: bool = True
    world_model: bool = True

    # ── Perception ───────────────────────────────────────────────────────
    vision: bool = False  # ViT-based image understanding
    audio_input: bool = False  # Whisper STT
    audio_output: bool = False  # TTS synthesis

    # ── Actions ──────────────────────────────────────────────────────────
    code_execution: bool = True
    browser_automation: bool = False
    iot_mqtt: bool = False
    ros2_robotics: bool = False
    mcp_client: bool = True
    shell_execution: bool = True

    # ── Training ─────────────────────────────────────────────────────────
    online_learning: bool = False  # DPO / continuous learning
    lora_training: bool = False  # Fine-tuning at runtime

    # ── Infrastructure ───────────────────────────────────────────────────
    persistent_daemon: bool = True  # Always-on background process
    session_persistence: bool = True  # Persist sessions across restarts
    multi_tenant: bool = True
    distributed_sync: bool = False  # SynapseGateway cross-device mesh

    # ── Experimental ─────────────────────────────────────────────────────
    experimental_features: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Profile Name Enum
# ═══════════════════════════════════════════════════════════════════════════


class ProfileName(StrEnum):
    """Known profile names."""

    FULL = "full"
    LITE = "lite"
    EDGE = "edge"
    RESEARCH = "research"
    ROBOT = "robot"


# ═══════════════════════════════════════════════════════════════════════════
# Brain Profile
# ═══════════════════════════════════════════════════════════════════════════


class BrainProfile(BaseModel):
    """Declarative runtime profile for the HBLLM cognitive engine.

    Defines which subsystems are active, resource budgets, and
    behavioral parameters. Every subsystem queries ``profile.features``
    instead of ``if lite_mode``.

    Attributes:
        name: Human-readable profile name.
        profile_type: Profile category (full, lite, edge, etc.).
        features: Granular feature flags.
        resource_limits: Resource budget constraints.
        description: Profile description.
    """

    name: str = "Full Profile"
    profile_type: ProfileName = ProfileName.FULL
    features: ProfileFeatures = Field(default_factory=ProfileFeatures)
    description: str = "All subsystems active — full cognitive capabilities."

    # Resource budgets (advisory, subsystems should respect these)
    max_ram_mb: int = 0  # 0 = unlimited
    max_concurrent_sessions: int = 100
    max_memory_entries_per_tenant: int = 100_000
    inference_timeout_s: float = 120.0

    def has(self, feature_name: str) -> bool:
        """Check if a feature is enabled.

        Args:
            feature_name: Name of the feature (must match a ProfileFeatures field).

        Returns:
            True if the feature is enabled, False otherwise.

        Raises:
            AttributeError: If the feature name doesn't exist.
        """
        return getattr(self.features, feature_name)


# ═══════════════════════════════════════════════════════════════════════════
# Pre-built Profiles
# ═══════════════════════════════════════════════════════════════════════════


def full_profile() -> BrainProfile:
    """All subsystems active — maximum cognitive capabilities."""
    return BrainProfile(
        name="Full Profile",
        profile_type=ProfileName.FULL,
        features=ProfileFeatures(
            local_inference=True,
            snn_streams=True,
            autonomy_core=True,
            spawner_neurogenesis=True,
            vision=True,
            audio_input=True,
            audio_output=True,
            online_learning=True,
            lora_training=True,
            distributed_sync=True,
        ),
        description="All subsystems active — full cognitive capabilities.",
    )


def lite_profile() -> BrainProfile:
    """API-only inference, local memory, <250MB RAM target."""
    return BrainProfile(
        name="Lite Profile",
        profile_type=ProfileName.LITE,
        features=ProfileFeatures(
            # Inference: API-only, no local model
            local_inference=False,
            api_inference=True,
            rust_simd_acceleration=False,
            # Memory: core systems only
            episodic_memory=True,
            semantic_memory=True,
            procedural_memory=True,
            value_memory=True,
            knowledge_graph=False,
            spatial_memory=False,
            temporal_memory=False,
            importance_scoring=True,
            memory_consolidation=False,
            # Cognitive: essential only
            snn_streams=False,
            autonomy_core=False,
            goal_decomposition=True,
            meta_cognition=False,
            curiosity_engine=False,
            spawner_neurogenesis=False,
            # Human modeling: core
            user_model=True,
            project_graph=True,
            executive_cortex=False,
            relationship_memory=False,
            world_model=False,
            # Perception: disabled
            vision=False,
            audio_input=False,
            audio_output=False,
            # Actions: lightweight
            code_execution=True,
            browser_automation=False,
            iot_mqtt=False,
            ros2_robotics=False,
            mcp_client=True,
            shell_execution=True,
            # Training: disabled
            online_learning=False,
            lora_training=False,
            # Infrastructure
            persistent_daemon=True,
            session_persistence=True,
            multi_tenant=True,
            distributed_sync=False,
        ),
        max_ram_mb=250,
        max_concurrent_sessions=20,
        description="API-only inference, local memory, <250MB RAM footprint.",
    )


def edge_profile() -> BrainProfile:
    """Minimal footprint for Raspberry Pi / IoT edge devices."""
    return BrainProfile(
        name="Edge Profile",
        profile_type=ProfileName.EDGE,
        features=ProfileFeatures(
            local_inference=True,
            api_inference=False,
            rust_simd_acceleration=True,
            episodic_memory=True,
            semantic_memory=False,
            procedural_memory=True,
            value_memory=False,
            knowledge_graph=False,
            spatial_memory=True,
            temporal_memory=False,
            importance_scoring=False,
            memory_consolidation=False,
            snn_streams=False,
            autonomy_core=True,
            goal_decomposition=False,
            meta_cognition=False,
            curiosity_engine=False,
            spawner_neurogenesis=False,
            user_model=False,
            project_graph=False,
            executive_cortex=False,
            relationship_memory=False,
            world_model=True,
            vision=False,
            audio_input=True,
            audio_output=True,
            iot_mqtt=True,
            ros2_robotics=False,
            code_execution=False,
            browser_automation=False,
            mcp_client=False,
            shell_execution=True,
            online_learning=False,
            lora_training=False,
            persistent_daemon=True,
            session_persistence=False,
            multi_tenant=False,
            distributed_sync=True,
        ),
        max_ram_mb=512,
        max_concurrent_sessions=5,
        description="Minimal footprint for Raspberry Pi / IoT edge devices.",
    )


def research_profile() -> BrainProfile:
    """Full profile + experimental features, extra logging."""
    return BrainProfile(
        name="Research Profile",
        profile_type=ProfileName.RESEARCH,
        features=ProfileFeatures(
            local_inference=True,
            snn_streams=True,
            autonomy_core=True,
            spawner_neurogenesis=True,
            vision=True,
            audio_input=True,
            audio_output=True,
            online_learning=True,
            lora_training=True,
            curiosity_engine=True,
            experimental_features=True,
            distributed_sync=True,
        ),
        description="Full capabilities plus experimental features and verbose logging.",
    )


def robot_profile() -> BrainProfile:
    """Full profile + embodiment, ROS2, and sensor fusion."""
    return BrainProfile(
        name="Robot Profile",
        profile_type=ProfileName.ROBOT,
        features=ProfileFeatures(
            local_inference=True,
            snn_streams=True,
            autonomy_core=True,
            spawner_neurogenesis=True,
            spatial_memory=True,
            vision=True,
            audio_input=True,
            audio_output=True,
            iot_mqtt=True,
            ros2_robotics=True,
            world_model=True,
            distributed_sync=True,
        ),
        description="Full cognitive capabilities plus embodied robotics (ROS2, IoT, sensors).",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Profile Loader
# ═══════════════════════════════════════════════════════════════════════════

_PROFILE_REGISTRY: dict[str, type[Any] | Any] = {
    "full": full_profile,
    "lite": lite_profile,
    "edge": edge_profile,
    "research": research_profile,
    "robot": robot_profile,
}


def load_profile(name: str | None = None) -> BrainProfile:
    """Load a brain profile by name.

    Resolution order:
        1. Explicit name argument.
        2. ``HBLLM_PROFILE`` environment variable.
        3. Defaults to ``full``.

    Args:
        name: Profile name (full, lite, edge, research, robot).

    Returns:
        The resolved BrainProfile.

    Raises:
        ValueError: If the profile name is not recognized.
    """
    if name is None:
        name = os.environ.get("HBLLM_PROFILE", "full").lower()

    factory = _PROFILE_REGISTRY.get(name)
    if factory is None:
        available = ", ".join(sorted(_PROFILE_REGISTRY.keys()))
        raise ValueError(f"Unknown profile '{name}'. Available profiles: {available}")

    profile = factory()
    logger.info("Loaded brain profile: %s (%s)", profile.name, profile.profile_type)
    return profile


def register_profile(name: str, factory: Any) -> None:
    """Register a custom profile factory.

    This allows third-party plugins to define new deployment profiles.

    Args:
        name: Profile name (lowercase).
        factory: Callable that returns a BrainProfile.
    """
    _PROFILE_REGISTRY[name.lower()] = factory
    logger.info("Registered custom brain profile: %s", name)
