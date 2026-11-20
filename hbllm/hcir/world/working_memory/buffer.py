"""Prefrontal Cortex Basal Ganglia Working Memory (PBWM) Buffer."""

from __future__ import annotations

import logging
from typing import Any

from hbllm.hcir.world.working_memory.models import AttractorMemorySlot

logger = logging.getLogger(__name__)


class PBWMWorkingMemoryBuffer:
    """Prefrontal Cortex Basal Ganglia Working Memory (PBWM) bistable attractor buffer.

    Biologically grounded in O'Reilly & Frank (2006):
    - Gated Updating: Working memory representations only update when a striatal
      gating signal exceeds the gating threshold.
    - Bistable Maintenance: Once gated, active attractor states self-sustain
      across intervening steps, ignoring transient sensory distractors.
    - Transactional Eviction: Slots are systematically cleared upon goal satisfaction.
    """

    def __init__(
        self,
        n_slots: int = 8,
        gate_threshold: float = 0.5,
        decay_rate: float = 0.05,
    ) -> None:
        self.n_slots = n_slots
        self.gate_threshold = gate_threshold
        self.decay_rate = decay_rate
        self.slots: dict[str, AttractorMemorySlot] = {}

    def update_with_striatal_gate(
        self,
        slot_id: str,
        content: Any,
        striatal_gate_signal: float,
        tag: str = "",
    ) -> bool:
        """Attempt to update working memory slot through striatal gating.

        Returns True if gated and updated, False if distractor was filtered.
        """
        if striatal_gate_signal >= self.gate_threshold:
            slot = AttractorMemorySlot(
                slot_id=slot_id,
                content=content,
                activation_energy=1.0,
                maintenance_threshold=self.gate_threshold,
                is_bistable=True,
                stability_counter=0,
                tag=tag,
            )
            self.slots[slot_id] = slot
            return True
        return False

    def step_attractor_dynamics(self, distractor_perturbation: float = 0.0) -> None:
        """Step bistable recurrent dynamics and apply distractor filtering."""
        evicted: list[str] = []
        for slot_id, slot in list(self.slots.items()):
            if slot.is_bistable and slot.activation_energy >= slot.maintenance_threshold:
                # Bistable attractor dynamics: active slots self-reinforce against sub-threshold noise
                noise_impact = max(0.0, distractor_perturbation - 0.4)
                new_act = (
                    slot.activation_energy + 0.1 * (1.0 - slot.activation_energy) - noise_impact
                )
                slot.activation_energy = float(min(1.0, max(0.0, new_act)))
                if slot.activation_energy >= slot.maintenance_threshold:
                    slot.stability_counter += 1
                else:
                    slot.is_bistable = False
            else:
                # Sub-threshold decay
                slot.activation_energy = max(0.0, slot.activation_energy - self.decay_rate)
                if slot.activation_energy <= 0.1:
                    evicted.append(slot_id)

        for slot_id in evicted:
            self.slots.pop(slot_id, None)

    def get_slot(self, slot_id: str) -> AttractorMemorySlot | None:
        """Retrieve slot by ID if active and above maintenance threshold."""
        slot = self.slots.get(slot_id)
        if slot and slot.activation_energy >= slot.maintenance_threshold:
            return slot
        return None

    def clear_slot(self, slot_id: str) -> bool:
        """Explicitly clear a working memory slot upon goal completion."""
        return self.slots.pop(slot_id, None) is not None

    def active_slots(self) -> list[AttractorMemorySlot]:
        """Return all active bistable slots."""
        return [
            s
            for s in self.slots.values()
            if s.is_bistable and s.activation_energy >= s.maintenance_threshold
        ]
