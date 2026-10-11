"""Ventromedial Prefrontal Cortex (vmPFC) Remote Causal Attribution & Action-Effect Binding.

Modeled on mammalian vmPFC distal causal credit assignment (O'Doherty et al., 2017; Rushworth et al., 2011):
1. Spatially Displaced Action-Effect Binding: Correlates local motor interactions
   at (r_trigger, c_trigger) with distal environmental state transitions at (r_remote, c_remote)
   where Manhattan distance > 1.
2. Controllability Contingency Estimation: Tracks co-occurrence statistics:
   Delta P = P(remote_delta | trigger_action) - P(remote_delta | baseline).
3. Reverse Affordance Lookup: When a terminal or intermediate goal is obstructed by a barrier
   at (r_barrier, c_barrier), queries vmPFC to identify the distal switch/plate (r_trigger, c_trigger)
   governing that barrier.
4. Remote Actuation Subgoal Formulation: Synthesizes high-priority navigation subgoals directing
   the agent to activate the remote trigger before attempting to traverse the barrier corridor.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class RemoteCausalAffordance:
    """Discovered causal contingency linking a local trigger to a remote actuator effect."""

    trigger_pos: tuple[int, int]
    trigger_feature: int
    remote_pos: tuple[int, int]
    original_remote_feature: int
    mutated_remote_feature: int
    activation_count: int = 1
    confidence: float = 0.5
    is_reversible: bool = True

    def to_causal_predicate(self) -> CausalPredicate:
        """Export remote causal affordance as a declarative causal predicate."""
        return CausalPredicate(
            variable=f"remote_affordance:{self.trigger_feature}->{self.original_remote_feature}",
            operator="==",
            value=self.mutated_remote_feature,
            relation=f"conf:{self.confidence:.2f}",
            target_entity=f"trig:{self.trigger_pos}->rem:{self.remote_pos}",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording remote causal affordance."""
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=f"remote_aff_{self.trigger_pos[0]}_{self.trigger_pos[1]}_{self.remote_pos[0]}_{self.remote_pos[1]}",
                node_data={
                    "trigger_pos": self.trigger_pos,
                    "trigger_feature": self.trigger_feature,
                    "remote_pos": self.remote_pos,
                    "original_remote_feature": self.original_remote_feature,
                    "mutated_remote_feature": self.mutated_remote_feature,
                    "activation_count": self.activation_count,
                    "confidence": self.confidence,
                    "is_reversible": self.is_reversible,
                },
            )
        ]


class RemoteCausalAttributor:
    """Ventromedial Prefrontal Cortex engine for distal causal learning and remote mechanism control."""

    def __init__(self, min_remote_distance: int = 2, confidence_threshold: float = 0.70) -> None:
        self.min_remote_distance = min_remote_distance
        self.confidence_threshold = confidence_threshold

        # remote_pos -> list of RemoteCausalAffordance
        self.remote_actuators: defaultdict[tuple[int, int], list[RemoteCausalAffordance]] = (
            defaultdict(list)
        )
        # trigger_pos -> list of RemoteCausalAffordance
        self.trigger_registry: defaultdict[tuple[int, int], list[RemoteCausalAffordance]] = (
            defaultdict(list)
        )
        # Co-occurrence evidence: (trigger_pos, remote_pos) -> count
        self.co_occurrence_evidence: defaultdict[tuple[tuple[int, int], tuple[int, int]], int] = (
            defaultdict(int)
        )

    def reset_episode(self, retain_long_term: bool = True) -> None:
        """Reset transient episode state while optionally preserving confirmed causal laws."""
        if not retain_long_term:
            self.remote_actuators.clear()
            self.trigger_registry.clear()
            self.co_occurrence_evidence.clear()

    def reset(self) -> None:
        """Reset all causal affordances and evidence."""
        self.reset_episode(retain_long_term=False)

    def record_transition(
        self,
        prev_grid: np.ndarray,
        curr_grid: np.ndarray,
        action_pos: tuple[int, int] | None,
        background_feature: int = 0,
        avatar_features: set[int] | None = None,
    ) -> list[RemoteCausalAffordance]:
        """Correlate action at action_pos with distal changes across the environment.

        Args:
            prev_grid: Visual state before action.
            curr_grid: Visual state after action.
            action_pos: Position where avatar acted (stepped on or clicked).
            background_feature: Empty background index.
            avatar_features: Avatar feature set.

        Returns:
            List of newly discovered or strengthened RemoteCausalAffordances.
        """
        if (
            action_pos is None
            or not isinstance(prev_grid, np.ndarray)
            or not isinstance(curr_grid, np.ndarray)
        ):
            return []

        av_set = avatar_features or set()
        tr, tc = action_pos
        trigger_feat = int(prev_grid[tr, tc])

        # Compute all cell mutations between frames
        diff_mask = prev_grid != curr_grid
        if not np.any(diff_mask):
            return []

        diff_coords = np.argwhere(diff_mask)
        newly_confirmed: list[RemoteCausalAffordance] = []

        for dr, dc in diff_coords:
            rr, rc = int(dr), int(dc)
            dist = abs(rr - tr) + abs(rc - tc)

            # Must be distal (beyond immediate avatar body footprint)
            if dist < self.min_remote_distance:
                continue

            orig_feat = int(prev_grid[rr, rc])
            mut_feat = int(curr_grid[rr, rc])

            # Ignore changes caused solely by avatar movement
            if orig_feat in av_set or mut_feat in av_set:
                continue

            pair_key = (action_pos, (rr, rc))
            self.co_occurrence_evidence[pair_key] += 1
            count = self.co_occurrence_evidence[pair_key]

            # Compute causal confidence: rises with repeated observations
            # 1 obs -> 0.75, 2 obs -> 0.98
            conf = min(0.98, 0.50 + 0.25 * count)

            # Check if this affordance is already registered
            existing = [
                aff for aff in self.remote_actuators[(rr, rc)] if aff.trigger_pos == action_pos
            ]

            if existing:
                aff = existing[0]
                aff.activation_count = count
                aff.confidence = conf
                aff.mutated_remote_feature = mut_feat
            else:
                aff = RemoteCausalAffordance(
                    trigger_pos=action_pos,
                    trigger_feature=trigger_feat,
                    remote_pos=(rr, rc),
                    original_remote_feature=orig_feat,
                    mutated_remote_feature=mut_feat,
                    activation_count=count,
                    confidence=conf,
                )
                self.remote_actuators[(rr, rc)].append(aff)
                self.trigger_registry[action_pos].append(aff)
                logger.info(
                    "RemoteCausalAttributor: Discovered remote causal link! Trigger at %s (feat=%d) modifies distal cell %s (%d -> %d) [conf=%.2f]",
                    action_pos,
                    trigger_feat,
                    (rr, rc),
                    orig_feat,
                    mut_feat,
                    conf,
                )

            if conf >= self.confidence_threshold:
                newly_confirmed.append(aff)

        return newly_confirmed

    def get_trigger_for_barrier(
        self,
        barrier_pos: tuple[int, int],
        barrier_feature: int | None = None,
    ) -> RemoteCausalAffordance | None:
        """Find the remote switch or pressure plate that can open/clear the specified barrier coordinate."""
        candidates = self.remote_actuators.get(barrier_pos, [])
        if not candidates:
            # Check if any actuator with the same barrier feature exists
            if barrier_feature is not None:
                for affs in self.remote_actuators.values():
                    for a in affs:
                        if (
                            a.original_remote_feature == barrier_feature
                            and a.confidence >= self.confidence_threshold
                        ):
                            return a
            return None

        # Sort by highest causal confidence
        viable = [c for c in candidates if c.confidence >= self.confidence_threshold]
        if not viable and candidates:
            viable = [c for c in candidates if c.confidence >= 0.50]
        if viable:
            viable.sort(key=lambda a: a.confidence, reverse=True)
            return viable[0]
        return None

    # Alias for API compatibility across cortex subsystems
    get_remote_trigger_for_barrier = get_trigger_for_barrier

    def get_all_remote_triggers(self) -> list[tuple[int, int]]:
        """Return list of coordinates identified as functional remote triggers."""
        return [
            trig
            for trig, affs in self.trigger_registry.items()
            if any(a.confidence >= self.confidence_threshold for a in affs)
        ]

    def to_causal_predicate(self) -> CausalPredicate:
        """Export remote causal attributor state as a declarative causal predicate."""
        return CausalPredicate(
            variable="remote_causal_attributor",
            operator="==",
            value=len(self.co_occurrence_evidence),
            relation=f"actuators:{len(self.remote_actuators)}",
            target_entity="vmpfc_causal_attributor",
        )

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all discovered remote affordances as declarative causal predicates."""
        preds = [self.to_causal_predicate()]
        for aff_list in self.remote_actuators.values():
            for aff in aff_list:
                preds.append(aff.to_causal_predicate())
        return preds

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transactions persisting remote causal attribution state."""
        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="remote_causal_attributor_summary",
                node_data={
                    "evidence_count": len(self.co_occurrence_evidence),
                    "actuator_count": len(self.remote_actuators),
                    "trigger_count": len(self.trigger_registry),
                },
            )
        ]
        for aff_list in self.remote_actuators.values():
            for aff in aff_list:
                ops.extend(aff.to_transaction_operations())
        return ops
