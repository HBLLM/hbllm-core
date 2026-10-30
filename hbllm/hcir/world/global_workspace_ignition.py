"""Global Neuronal Workspace Theory (GNWT) Non-Linear Ignition & Conscious Broadcast.

Biological Foundations:
- Dehaene, Changeux (2005): "Ongoing spontaneous activity controls access to consciousness:
  a neuronal model for conscious vs. unconscious perception" (PLoS Biol 3(5): e141).
- Mashour, Roelfsema, Changeux, Dehaene (2020): "Conscious Processing and the Global Neuronal
  Workspace" (Neuron 105(5): 776-798).
- Baars, B. J. (2002): "The conscious access hypothesis: origins and recent evidence"
  (Trends Cogn Sci 6(1): 47-52).

Key Cognitive Capabilities:
1. Non-Linear "All-or-None" Ignition:
   - Unconscious/subliminal signals compete from sensory perception, ACC conflict,
     episodic memory, and causal theory. When candidate salience crosses dynamic threshold
     theta, recurrent amplification triggers sudden global ignition.
2. Global Workspace Broadcast Bus:
   - Once ignited, the winning mental packet is globally broadcast across all cortical
     subsystems (OFC planner, motor dispatch, episodic consolidation, deliberator).
3. Top-Down Reverberatory Sustained Access & Refractory Gating:
   - Sustains ignited contents across deliberation cycles while imposing an adaptive
     refractory gate to prevent strobe instability or runaway seizures.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hbllm.hcir.world.causal_discovery import CausalPredicate


class IgnitionPriority(str, Enum):
    """Priority level of candidate signal vying for conscious workspace ignition."""

    SUBLIMINAL = "subliminal"  # Below threshold: processed unconsciously
    PRECONSCIOUS = "preconscious"  # Near threshold: buffered for potential access
    IGNITED = "ignited"  # Above threshold: ignited and globally broadcast
    EMERGENCY = "emergency"  # Hard interrupt (e.g. fatal collision cone, pain, boundary breach)


@dataclass(frozen=True)
class CandidateIgnitionSignal:
    """A competing sensory, interoceptive, or deliberative signal seeking workspace access."""

    source_modality: str  # e.g., 'vision', 'conflict', 'motor_feedback', 'interoception'
    content_payload: str
    salience_score: float  # S in [0.0, 1.0]
    evidential_weight: float = 1.0
    payload_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConsciousIgnitionPacket:
    """The globally ignited conscious mental state broadcast across the cognitive OS."""

    packet_id: str
    source_modality: str
    content_payload: str
    salience_score: float
    reverberation_energy: float
    broadcast_cycle: int
    payload_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_causal_predicate(self) -> CausalPredicate:
        """Convert conscious workspace packet into an HCIR CausalPredicate."""
        return CausalPredicate(
            variable="global_workspace_broadcast",
            operator="==",
            value=f"{self.source_modality}:{self.content_payload}:energy:{self.reverberation_energy:.2f}:cycle:{self.broadcast_cycle}",
            relation="global_neuronal_workspace_ignition",
            target_entity="prefrontal_parietal_workspace",
        )


@dataclass(frozen=True)
class GlobalWorkspaceState:
    """Current state of the Global Neuronal Workspace."""

    active_ignition: ConsciousIgnitionPacket | None
    is_ignited: bool
    reverberation_decay: float
    competing_signals_count: int
    subliminal_suppressed_count: int


class GlobalWorkspaceIgnitionEngine:
    """Orchestrator for GNWT non-linear ignition, reverberation, and conscious broadcast."""

    def __init__(
        self,
        ignition_threshold: float = 0.65,
        reverberation_decay_rate: float = 0.25,
        refractory_period_cycles: int = 1,
    ) -> None:
        """Initialize GlobalWorkspaceIgnitionEngine.

        Args:
            ignition_threshold: Minimum combined salience theta required to trigger ignition.
            reverberation_decay_rate: Exponential decay rate of sustained conscious access.
            refractory_period_cycles: Cycles after ignition before another signal can ignite.
        """
        self.ignition_threshold: float = max(0.1, min(0.99, float(ignition_threshold)))
        self.decay_rate: float = max(0.05, min(0.95, float(reverberation_decay_rate)))
        self.refractory_cycles: int = max(0, int(refractory_period_cycles))

        self._active_packet: ConsciousIgnitionPacket | None = None
        self._current_cycle: int = 0
        self._last_ignition_cycle: int = -999

    @property
    def current_cycle(self) -> int:
        """Get current cognitive cycle index."""
        return self._current_cycle

    def compete_and_ignite(
        self,
        signals: tuple[CandidateIgnitionSignal, ...] | list[CandidateIgnitionSignal],
    ) -> GlobalWorkspaceState:
        """Run non-linear competition among candidate signals and ignite the global workspace.

        Args:
            signals: List of candidate sensory, cognitive, or interoceptive signals.

        Returns:
            GlobalWorkspaceState detailing the active conscious broadcast and competitor status.
        """
        self._current_cycle += 1

        # Step 1: Check existing reverberation decay
        if self._active_packet is not None:
            decayed_energy = self._active_packet.reverberation_energy * (1.0 - self.decay_rate)
            if decayed_energy < 0.15:
                # Conscious state collapsed back to baseline
                self._active_packet = None
            else:
                self._active_packet = ConsciousIgnitionPacket(
                    packet_id=self._active_packet.packet_id,
                    source_modality=self._active_packet.source_modality,
                    content_payload=self._active_packet.content_payload,
                    salience_score=self._active_packet.salience_score,
                    reverberation_energy=decayed_energy,
                    broadcast_cycle=self._current_cycle,
                    payload_metadata=self._active_packet.payload_metadata,
                )

        if not signals:
            return GlobalWorkspaceState(
                active_ignition=self._active_packet,
                is_ignited=self._active_packet is not None,
                reverberation_decay=self._active_packet.reverberation_energy
                if self._active_packet
                else 0.0,
                competing_signals_count=0,
                subliminal_suppressed_count=0,
            )

        # Step 2: Sort candidate signals by non-linear sigmoid amplified salience
        scored: list[tuple[float, CandidateIgnitionSignal]] = []
        for sig in signals:
            s = max(0.0, min(1.0, float(sig.salience_score)))
            w = max(0.1, float(sig.evidential_weight))
            # Non-linear sigmoidal amplification around threshold
            amplified = 1.0 / (1.0 + math.exp(-8.0 * (s - self.ignition_threshold))) * w
            scored.append((amplified, sig))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top_score, top_signal = scored[0]

        # Step 3: Evaluate all-or-none ignition criteria
        in_refractory = (self._current_cycle - self._last_ignition_cycle) <= self.refractory_cycles
        # Emergency signals can break through refractory period
        is_emergency = top_signal.salience_score >= 0.95

        can_ignite = (
            top_score >= 0.50 or top_signal.salience_score >= self.ignition_threshold
        ) and (not in_refractory or is_emergency)

        subliminal_count = sum(
            1 for s, sig in scored if sig.salience_score < self.ignition_threshold
        )

        if can_ignite:
            # Overwrite or ignite workspace
            reverberation = min(1.0, 0.70 + 0.30 * top_signal.salience_score)
            self._active_packet = ConsciousIgnitionPacket(
                packet_id=f"ignite_{self._current_cycle}_{top_signal.source_modality}",
                source_modality=top_signal.source_modality,
                content_payload=top_signal.content_payload,
                salience_score=top_signal.salience_score,
                reverberation_energy=reverberation,
                broadcast_cycle=self._current_cycle,
                payload_metadata=top_signal.payload_metadata,
            )
            self._last_ignition_cycle = self._current_cycle

        return GlobalWorkspaceState(
            active_ignition=self._active_packet,
            is_ignited=self._active_packet is not None,
            reverberation_decay=self._active_packet.reverberation_energy
            if self._active_packet
            else 0.0,
            competing_signals_count=len(signals),
            subliminal_suppressed_count=subliminal_count,
        )
