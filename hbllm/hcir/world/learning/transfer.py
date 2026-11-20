"""Cross-Context Knowledge Transfer & Persistent Invariant Promotion."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hbllm.hcir.world.causal.predicates import CausalPredicate

if TYPE_CHECKING:
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine

logger = logging.getLogger(__name__)


@dataclass
class TransferableInvariant:
    """A causal invariant abstracted for cross-game transfer.

    Stripped of position-specific data, retaining only the structural
    pattern that might generalize across environments.
    """

    invariant_id: str
    category: str  # e.g., "goal_feature", "barrier_feature", "action_dynamics", "hazard"
    description: str
    data: dict[str, Any] = field(default_factory=dict)
    source_game: str = ""
    confidence: float = 1.0
    transfer_successes: int = 0
    transfer_failures: int = 0

    def decay_confidence(self) -> None:
        """Decay confidence when transfer fails."""
        self.transfer_failures += 1
        self.confidence = max(0.1, self.confidence * 0.7)

    def boost_confidence(self) -> None:
        """Boost confidence when transfer succeeds."""
        self.transfer_successes += 1
        self.confidence = min(1.0, self.confidence * 1.1 + 0.05)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export as a formal Pearl do(a) causal predicate."""
        return CausalPredicate(
            variable=f"transfer_invariant:{self.invariant_id}",
            operator="==",
            value=f"cat:{self.category}:conf:{self.confidence:.2f}",
            relation=f"source:{self.source_game}",
            target_entity=self.invariant_id,
        )


class CrossContextKnowledgeTransfer:
    """Manages extraction, promotion, and retrieval of cross-context knowledge.

    On context completion:
        1. Extracts universal causal invariants from the symbolic theory
        2. Abstracts position-specific data into transferable patterns
        3. Stores in persistent knowledge base

    On new context start:
        1. Queries persistent knowledge for applicable priors
        2. Seeds the engine's symbolic theory with transferable invariants
        3. Tracks transfer success/failure for confidence updating
    """

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self.persistent_invariants: dict[str, TransferableInvariant] = {}
        self.active_transfers: dict[str, TransferableInvariant] = {}
        self.storage_dir: Path = Path(storage_dir or "./knowledge_store")
        self.load_from_disk()

    def save_to_disk(self, storage_dir: str | Path | None = None) -> bool:
        """Atomically persist learned invariants to disk."""
        target_dir = Path(storage_dir or self.storage_dir)
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            file_path = target_dir / "persistent_invariants.json"
            records = [
                {
                    "invariant_id": inv.invariant_id,
                    "category": inv.category,
                    "description": inv.description,
                    "data": inv.data,
                    "source_game": inv.source_game,
                    "confidence": inv.confidence,
                }
                for inv in self.persistent_invariants.values()
            ]
            temp_path = file_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            temp_path.replace(file_path)
            logger.info("Saved %d persistent invariants to %s", len(records), file_path)
            return True
        except Exception as e:
            logger.warning("Could not persist invariants to disk: %s", e)
            return False

    def load_from_disk(self, storage_dir: str | Path | None = None) -> int:
        """Load previously learned invariants from disk."""
        target_dir = Path(storage_dir or self.storage_dir)
        file_path = target_dir / "persistent_invariants.json"
        if not file_path.is_file():
            return 0
        try:
            with open(file_path, encoding="utf-8") as f:
                records = json.load(f)
            loaded = 0
            for r in records:
                inv = TransferableInvariant(
                    invariant_id=r["invariant_id"],
                    category=r["category"],
                    description=r["description"],
                    data=r.get("data", {}),
                    source_game=r.get("source_game", ""),
                    confidence=float(r.get("confidence", 1.0)),
                )
                self.persistent_invariants[inv.invariant_id] = inv
                loaded += 1
            logger.info("Loaded %d persistent invariants from %s", loaded, file_path)
            return loaded
        except Exception as e:
            logger.warning("Could not load invariants from disk: %s", e)
            return 0

    def extract_invariants(
        self,
        engine: AutonomousEpistemicEngine,
        context_id: str = "",
    ) -> list[TransferableInvariant]:
        """Extract universal causal invariants from current context's learned knowledge."""
        invariants: list[TransferableInvariant] = []
        theory = engine.symbolic_theory

        # Extract action dynamics (these often transfer across games with same action space)
        for act_id, dyn in engine.action_dynamics.items():
            if dyn.is_displacement_action() and dyn.probes_tested >= 3:
                inv_id = f"action_dynamics_{act_id}_{dyn.delta_r}_{dyn.delta_c}"
                invariants.append(
                    TransferableInvariant(
                        invariant_id=inv_id,
                        category="action_dynamics",
                        description=f"Action {act_id} displaces avatar by ({dyn.delta_r}, {dyn.delta_c})",
                        data={"action_id": act_id, "delta_r": dyn.delta_r, "delta_c": dyn.delta_c},
                        source_game=context_id,
                    )
                )

        # Extract lethal features (contact hazards)
        av_feats = engine.avatar_features or (
            {engine.avatar_feature} if engine.avatar_feature is not None else set()
        )
        for feat in engine.episodic_cortex.discovered_lethal_features:
            if feat in av_feats:
                continue
            inv_id = f"lethal_feature_{feat}"
            invariants.append(
                TransferableInvariant(
                    invariant_id=inv_id,
                    category="hazard",
                    description=f"Feature {feat} is lethal upon contact",
                    data={"feature": feat},
                    source_game=context_id,
                )
            )

        # Extract tool-barrier affinities (key-door relationships)
        for tool_feat, barrier_feats in theory.tool_barrier_affinities.items():
            for barrier_feat in barrier_feats:
                inv_id = f"tool_barrier_{tool_feat}_{barrier_feat}"
                invariants.append(
                    TransferableInvariant(
                        invariant_id=inv_id,
                        category="tool_barrier_affinity",
                        description=f"Feature {tool_feat} clears barrier feature {barrier_feat}",
                        data={"tool_feature": tool_feat, "barrier_feature": barrier_feat},
                        source_game=context_id,
                    )
                )

        return invariants

    def promote_to_persistent(self, invariants: list[TransferableInvariant]) -> int:
        """Promote extracted invariants to persistent cross-game knowledge store."""
        new_count = 0
        for inv in invariants:
            if inv.invariant_id in self.persistent_invariants:
                self.persistent_invariants[inv.invariant_id].boost_confidence()
            else:
                self.persistent_invariants[inv.invariant_id] = inv
                new_count += 1
        logger.info(
            "CrossGameTransfer: Promoted %d invariants (%d new, %d boosted)",
            len(invariants),
            new_count,
            len(invariants) - new_count,
        )
        return new_count

    def query_applicable_priors(
        self,
        available_actions: set[int],
        min_confidence: float = 0.3,
    ) -> list[TransferableInvariant]:
        """Query persistent knowledge for invariants applicable to a new game."""
        applicable: list[TransferableInvariant] = []
        for inv in self.persistent_invariants.values():
            if inv.confidence < min_confidence:
                continue
            if inv.category == "action_dynamics":
                act_id = inv.data.get("action_id")
                if act_id is not None and act_id not in available_actions:
                    continue
            applicable.append(inv)
        applicable.sort(key=lambda x: x.confidence, reverse=True)
        return applicable

    def seed_engine_with_priors(
        self,
        engine: AutonomousEpistemicEngine,
        available_actions: set[int],
    ) -> int:
        """Seed a new game's engine with applicable prior knowledge."""
        priors = self.query_applicable_priors(available_actions)
        applied = 0
        self.active_transfers.clear()

        for inv in priors:
            if inv.category == "action_dynamics":
                act_id = inv.data.get("action_id")
                if act_id is not None and act_id in available_actions:
                    if act_id not in engine.action_dynamics:
                        from hbllm.hcir.world.motor_calibration import ActionDynamicsModel

                        engine.action_dynamics[act_id] = ActionDynamicsModel(
                            action_id=act_id,
                            delta_r=int(inv.data.get("delta_r", 0)),
                            delta_c=int(inv.data.get("delta_c", 0)),
                            confidence=inv.confidence,
                            probes_tested=1,
                        )
                        self.active_transfers[inv.invariant_id] = inv
                        applied += 1
            elif inv.category == "hazard":
                feat = inv.data.get("feature")
                av_f = engine.avatar_features or (
                    {engine.avatar_feature} if engine.avatar_feature is not None else set()
                )
                if feat is not None and feat not in av_f:
                    engine.episodic_cortex.discovered_lethal_features.add(feat)
                    self.active_transfers[inv.invariant_id] = inv
                    applied += 1
            elif inv.category == "tool_barrier_affinity":
                tool_feat = inv.data.get("tool_feature")
                barrier_feat = inv.data.get("barrier_feature")
                if tool_feat is not None and barrier_feat is not None:
                    engine.symbolic_theory.record_tool_barrier_affinity(tool_feat, barrier_feat)
                    self.active_transfers[inv.invariant_id] = inv
                    applied += 1

        if applied > 0:
            logger.info("CrossGameTransfer: Seeded engine with %d prior invariants", applied)
        return applied

    def record_transfer_outcome(self, game_won_any_levels: bool) -> None:
        """Update confidence of actively transferred invariants based on game outcome."""
        for inv in self.active_transfers.values():
            if game_won_any_levels:
                inv.boost_confidence()
            else:
                inv.decay_confidence()
        self.active_transfers.clear()

    def on_game_complete(
        self,
        engine: AutonomousEpistemicEngine,
        context_id: str,
        levels_won: int,
    ) -> None:
        """Full lifecycle hook: extract, promote, record outcome, and save to disk."""
        invariants = self.extract_invariants(engine, context_id=context_id)
        self.promote_to_persistent(invariants)
        self.record_transfer_outcome(levels_won > 0)
        self.save_to_disk()

    def on_game_start(
        self,
        engine: AutonomousEpistemicEngine,
        available_actions: set[int],
    ) -> int:
        """Full lifecycle hook: query priors and seed engine."""
        return self.seed_engine_with_priors(engine, available_actions)
