"""Copy-On-Write State Projector for Future Simulations."""

from __future__ import annotations

import copy
from typing import Any

from hbllm.brain.simulation.models import FutureWorldState, PredictionOrigin
from hbllm.brain.world.world_state import EntityState, WorldStateEngine


class ProjectedState:
    """A copy-on-write wrapper around WorldStateEngine to prevent contamination."""

    def __init__(self, base_state: WorldStateEngine) -> None:
        self.base_state = base_state
        self.base_clock = getattr(base_state, "logical_clock", 0)
        # Only stores mutated EntityStates
        self.mutations: dict[str, EntityState] = {}

    def get_entity_state(self, entity_id: str) -> EntityState | None:
        """Get the entity state, preferring local mutations over the base state."""
        if entity_id in self.mutations:
            return self.mutations[entity_id]

        # Pull from base and deepcopy if we need to mutate it later
        base_entity = self.base_state.get_entity_state(entity_id)
        if base_entity:
            return copy.deepcopy(base_entity)

        return None

    def update_entity(
        self, entity_id: str, new_properties: dict[str, Any], confidence: float = 1.0
    ) -> None:
        """Apply a simulated mutation to an entity."""
        entity = self.get_entity_state(entity_id)
        if not entity:
            from hbllm.brain.world.world_state import EntityState

            entity = EntityState(entity_id=entity_id, confidence=confidence)

        entity.properties.update(new_properties)
        self.mutations[entity_id] = entity

    def finalize(self, origin: PredictionOrigin) -> FutureWorldState:
        """Finalize the projection into a static FutureWorldState."""
        serialized_mutations = {}
        for k, v in self.mutations.items():
            serialized_mutations[k] = {
                "entity_id": v.entity_id,
                "properties": v.properties,
                "confidence": v.confidence,
                "last_updated": v.last_updated,
                "source_set": list(v.source_set),
            }

        return FutureWorldState(
            base_clock=self.base_clock,
            mutations=serialized_mutations,
            predicted_confidence=1.0,
            prediction_origin=origin,
        )
