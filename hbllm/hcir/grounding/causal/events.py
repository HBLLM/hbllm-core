from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hbllm.hcir.grounding.contracts.references import (
    EntityCandidateRef,
    GroundedEntityRef,
    GroundedRef,
)
from hbllm.hcir.grounding.relations.dyads import SpatialDyad


class EventType(str, Enum):
    """Categorical discrete event transitions detected in sensory-motor streams."""

    STATE_CHANGE = "state_change"  # Discrete property change (color, open/closed, mode)
    CONTACT_INITIATED = "contact_initiated"  # Mechanical impact or touch begun
    CONTACT_BROKEN = "contact_broken"  # Contact ceased
    MOTION_STARTED = "motion_started"  # Entity transitioned from stationary to moving
    MOTION_STOPPED = "motion_stopped"  # Entity came to rest
    BOUNDARY_ENTERED = "boundary_entered"  # Entity crossed into region / containment
    BOUNDARY_EXITED = "boundary_exited"  # Entity departed region / containment
    ACTION_PERFORMED = "action_performed"  # Emitted by an active motor actuator


@dataclass(frozen=True)
class EventCandidate:
    """A discrete temporal transition detected across observation frames.

    Provides the empirical substrate for Pearl's causal induction and schema learning.
    """

    event_id: str
    event_type: EventType
    primary_entity_ref: GroundedRef
    secondary_entity_ref: GroundedRef | None = None
    pre_state: dict[str, Any] = field(default_factory=dict)
    post_state: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    confidence: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id must be non-empty")
        if not isinstance(self.primary_entity_ref, GroundedRef):
            raise TypeError(
                f"primary_entity_ref must be GroundedRef, got {type(self.primary_entity_ref)}"
            )


class EventDetector:
    """Detects discrete kinematic, topological, and state transitions between frames.

    Pure deterministic processing (zero external LLM calls).
    """

    def __init__(self, speed_threshold: float = 0.05) -> None:
        self._speed_threshold = speed_threshold
        self._prior_states: dict[str, dict[str, Any]] = {}
        self._prior_dyads: dict[tuple[str, str], SpatialDyad] = {}

    def detect_events(
        self,
        current_states: dict[str, dict[str, Any]],
        current_dyads: list[SpatialDyad] = (),
        timestamp: float = 0.0,
    ) -> list[EventCandidate]:
        """Compares current entity states and dyad relations against prior snapshot."""
        now = timestamp if timestamp > 0.0 else time.time()
        events: list[EventCandidate] = []

        # 1. Detect individual entity state and motion transitions
        for ent_id, curr_st in current_states.items():
            ent_ref = self._to_ref(ent_id)
            prior_st = self._prior_states.get(ent_id)

            if prior_st is not None:
                # Motion transitions
                curr_spd = curr_st.get("speed", 0.0)
                prior_spd = prior_st.get("speed", 0.0)

                if prior_spd < self._speed_threshold and curr_spd >= self._speed_threshold:
                    events.append(
                        EventCandidate(
                            event_id=f"evt_motion_start_{uuid.uuid4().hex[:8]}",
                            event_type=EventType.MOTION_STARTED,
                            primary_entity_ref=ent_ref,
                            pre_state={"speed": prior_spd},
                            post_state={"speed": curr_spd},
                            timestamp=now,
                            description=f"{ent_id} started moving (speed {curr_spd:.2f})",
                        )
                    )
                elif prior_spd >= self._speed_threshold and curr_spd < self._speed_threshold:
                    events.append(
                        EventCandidate(
                            event_id=f"evt_motion_stop_{uuid.uuid4().hex[:8]}",
                            event_type=EventType.MOTION_STOPPED,
                            primary_entity_ref=ent_ref,
                            pre_state={"speed": prior_spd},
                            post_state={"speed": curr_spd},
                            timestamp=now,
                            description=f"{ent_id} came to rest",
                        )
                    )

                # Arbitrary discrete property changes
                for prop_k, prop_v in curr_st.items():
                    if prop_k in ("speed", "position", "velocity", "bbox"):
                        continue
                    prior_v = prior_st.get(prop_k)
                    if prior_v is not None and prior_v != prop_v:
                        events.append(
                            EventCandidate(
                                event_id=f"evt_state_chg_{uuid.uuid4().hex[:8]}",
                                event_type=EventType.STATE_CHANGE,
                                primary_entity_ref=ent_ref,
                                pre_state={prop_k: prior_v},
                                post_state={prop_k: prop_v},
                                timestamp=now,
                                description=f"{ent_id} changed {prop_k}: {prior_v} -> {prop_v}",
                            )
                        )

            # Update prior
            self._prior_states[ent_id] = dict(curr_st)

        # 2. Detect dyadic contact and containment transitions
        for dyad in current_dyads:
            pair_key = (dyad.subject_id, dyad.object_id)
            sub_ref = self._to_ref(dyad.subject_id)
            obj_ref = self._to_ref(dyad.object_id)
            prior_dyad = self._prior_dyads.get(pair_key)

            if prior_dyad is not None:
                # Contact transitions
                if not prior_dyad.contact and dyad.contact:
                    events.append(
                        EventCandidate(
                            event_id=f"evt_contact_init_{uuid.uuid4().hex[:8]}",
                            event_type=EventType.CONTACT_INITIATED,
                            primary_entity_ref=sub_ref,
                            secondary_entity_ref=obj_ref,
                            pre_state={"contact": False, "distance": prior_dyad.distance},
                            post_state={"contact": True, "distance": dyad.distance},
                            timestamp=now,
                            description=f"{dyad.subject_id} made contact with {dyad.object_id}",
                        )
                    )
                elif prior_dyad.contact and not dyad.contact:
                    events.append(
                        EventCandidate(
                            event_id=f"evt_contact_brk_{uuid.uuid4().hex[:8]}",
                            event_type=EventType.CONTACT_BROKEN,
                            primary_entity_ref=sub_ref,
                            secondary_entity_ref=obj_ref,
                            pre_state={"contact": True, "distance": prior_dyad.distance},
                            post_state={"contact": False, "distance": dyad.distance},
                            timestamp=now,
                            description=f"{dyad.subject_id} broke contact with {dyad.object_id}",
                        )
                    )

                # Boundary / containment transitions
                if prior_dyad.containment_ratio < 0.8 and dyad.containment_ratio >= 0.8:
                    events.append(
                        EventCandidate(
                            event_id=f"evt_bnd_enter_{uuid.uuid4().hex[:8]}",
                            event_type=EventType.BOUNDARY_ENTERED,
                            primary_entity_ref=sub_ref,
                            secondary_entity_ref=obj_ref,
                            pre_state={"containment": prior_dyad.containment_ratio},
                            post_state={"containment": dyad.containment_ratio},
                            timestamp=now,
                            description=f"{dyad.subject_id} entered containment of {dyad.object_id}",
                        )
                    )
                elif prior_dyad.containment_ratio >= 0.8 and dyad.containment_ratio < 0.8:
                    events.append(
                        EventCandidate(
                            event_id=f"evt_bnd_exit_{uuid.uuid4().hex[:8]}",
                            event_type=EventType.BOUNDARY_EXITED,
                            primary_entity_ref=sub_ref,
                            secondary_entity_ref=obj_ref,
                            pre_state={"containment": prior_dyad.containment_ratio},
                            post_state={"containment": dyad.containment_ratio},
                            timestamp=now,
                            description=f"{dyad.subject_id} exited containment of {dyad.object_id}",
                        )
                    )

            self._prior_dyads[pair_key] = dyad

        return events

    def reset(self) -> None:
        """Clears state cache on episode boundaries."""
        self._prior_states.clear()
        self._prior_dyads.clear()

    @staticmethod
    def _to_ref(entity_id: str) -> GroundedRef:
        if entity_id.startswith("e_"):
            return GroundedEntityRef(entity_id)
        elif entity_id.startswith("ecand_"):
            return EntityCandidateRef(entity_id)
        return GroundedEntityRef(f"e_{entity_id}")
