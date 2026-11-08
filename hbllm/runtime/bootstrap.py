"""Canonical composition root for the HBLLM Cognitive OS (WP1).

Provides `build_cognitive_system()` as the single factory for instantiating
a fully-governed, verified cognitive architecture. All entry points
(Mind CLI, Serve, MyAgent, CognitiveBlackbox) must boot through this root.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hbllm.hcir.kernel.commit_authority import CommitAuthorityRegistry
from hbllm.hcir.kernel.governance.constitutional_verifier import ConstitutionalVerifier
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.verification import create_default_pipeline
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import TieredWorkspace

logger = logging.getLogger("hbllm.runtime.bootstrap")


@dataclass
class CognitiveSystemConfig:
    """Configuration parameters for the unified cognitive system."""

    world_id: str = "default"
    seed: int | None = None
    exploration_budget: int = 150
    enable_logging: bool = True
    instructions: Sequence[str] | str | None = None
    memory_dir: Path | str | None = None
    enable_grounding: bool = True
    perception_mode: str = "handcrafted"  # handcrafted | snn | learned
    enable_brain_llm: bool = False
    custom_verification_stages: list[Any] | None = None


@dataclass
class CognitiveSystem:
    """Container holding all components of a booted cognitive runtime."""

    services: KernelServices
    engine: Any  # AutonomousEpistemicEngine
    bridge: Any | None  # CorticalGroundingBridge
    workspace: HCIRWorkspaceState
    tiered_workspace: TieredWorkspace
    config: CognitiveSystemConfig
    brain: Any | None = None


def build_cognitive_system(config: CognitiveSystemConfig | None = None) -> CognitiveSystem:
    """Build and wire the single, canonical cognitive runtime.

    Guarantees:
    1. A single verified TransactionManager owning HCIR state mutations.
    2. ConstitutionalVerifier attached to all state modification pipelines.
    3. AutonomousEpistemicEngine and CorticalGroundingBridge sharing the same KernelServices.
    4. TieredWorkspace owning 4-tier memory state.
    """
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
    from hbllm.hcir.world.cortical_grounding_bridge import CorticalGroundingBridge

    cfg = config or CognitiveSystemConfig()

    # 1. State hierarchy & storage
    workspace = HCIRWorkspaceState()
    tiered_workspace = TieredWorkspace()

    # 2. Constitutional verification pipeline
    authority_registry = CommitAuthorityRegistry()
    verifier = ConstitutionalVerifier(authority_registry=authority_registry)
    pipeline = cfg.custom_verification_stages or create_default_pipeline(
        constitutional_verifier=verifier
    )

    # 3. Provision kernel services container (Single composition root)
    services = KernelServices.create_default(
        workspace=workspace,
        verification_pipeline=pipeline,
        tiered_workspace=tiered_workspace,
    )
    services.commit_authority_registry = authority_registry

    # 4. Instantiate Autonomous Epistemic Engine with shared KernelServices
    engine = AutonomousEpistemicEngine(
        exploration_budget=cfg.exploration_budget,
        enable_logging=cfg.enable_logging,
        instructions=cfg.instructions,
        services=services,
        world_id=cfg.world_id,
        seed=cfg.seed,
    )

    # 5. Wire Grounding Bridge with the exact same KernelServices
    bridge: CorticalGroundingBridge | None = None
    if cfg.enable_grounding:
        bridge = CorticalGroundingBridge(
            workspace=workspace,
            services=services,
            transaction_manager=services.transaction_manager,
        )
        engine.attach_grounding_bridge(bridge)

    logger.info(
        "Cognitive system booted successfully (world_id=%s, seed=%s, verifier=ConstitutionalVerifier)",
        cfg.world_id,
        cfg.seed,
    )

    return CognitiveSystem(
        services=services,
        engine=engine,
        bridge=bridge,
        workspace=workspace,
        tiered_workspace=tiered_workspace,
        config=cfg,
    )
