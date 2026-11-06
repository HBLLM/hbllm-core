"""Pulvinar Thalamic Transthalamic Routing & Attentional Saliency Gating.

Biologically grounded in mammalian pulvinar nucleus transthalamic coordination
(Saalmann & Kastner, 2012, 2015; Halassa & Kastner, 2017):
1. Transthalamic Cortico-Cortical Relays:
   - The pulvinar coordinates information transfer between sensory, prefrontal,
     and motor cortices by dynamically modulating transmission bandwidth across
     functional channels.
2. Cognitive Load Throttling:
   - When prefrontal cognitive load is elevated (deep forward planning, MCTS),
     the pulvinar throttles peripheral perceptual streams to prevent working memory
     distractor overflow.
3. Emergency Defensive Gating:
   - Under acute threat urgency, the pulvinar elevates motor and avoidance pathways
     to maximum priority while suppressing non-urgent long-horizon episodic retrieval.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class TransthalamicRoute:
    """A functional communication pathway coordinated by the pulvinar nucleus."""

    route_id: str
    source_faculty: str
    target_faculty: str
    bandwidth: float  # [0.0, 1.0] transmission capacity
    priority: int  # 1 (lowest) to 5 (highest emergency)
    is_active: bool = True


@dataclass
class PulvinarRoutingState:
    """Snapshot of active pulvinar transthalamic communication channels."""

    cognitive_load: float
    threat_urgency: float
    routes: dict[str, TransthalamicRoute]
    throttled_routes: list[str]
    dominant_pathway: str

    def to_causal_predicate(self) -> CausalPredicate:
        """Export pulvinar routing state as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="pulvinar_routing_state",
            operator="==",
            value=f"dominant:{self.dominant_pathway}:throttled:{len(self.throttled_routes)}",
            relation="pulvinar_transthalamic_gating",
            target_entity="thalamus_pulvinar",
        )


class PulvinarAttentionRouter:
    """Thalamic pulvinar executive router managing cortico-cortical transmission bandwidth."""

    def __init__(self) -> None:
        """Initialize pulvinar attention router with canonical pathways."""
        self.routes: dict[str, TransthalamicRoute] = {}
        self._init_canonical_routes()
        self.last_state: PulvinarRoutingState | None = None
        self.total_transmissions: int = 0

    def _init_canonical_routes(self) -> None:
        """Register the 4 principal transthalamic information highways."""
        self.routes = {
            "perception_to_deliberation": TransthalamicRoute(
                route_id="perception_to_deliberation",
                source_faculty="parietal_perception",
                target_faculty="prefrontal_deliberation",
                bandwidth=1.0,
                priority=3,
            ),
            "threat_to_motor_escape": TransthalamicRoute(
                route_id="threat_to_motor_escape",
                source_faculty="superior_colliculus",
                target_faculty="basal_ganglia_motor",
                bandwidth=1.0,
                priority=5,
            ),
            "episodic_to_working_memory": TransthalamicRoute(
                route_id="episodic_to_working_memory",
                source_faculty="hippocampal_replay",
                target_faculty="prefrontal_working_memory",
                bandwidth=0.8,
                priority=2,
            ),
            "conflict_to_effort_control": TransthalamicRoute(
                route_id="conflict_to_effort_control",
                source_faculty="anterior_cingulate",
                target_faculty="prefrontal_cortex",
                bandwidth=1.0,
                priority=4,
            ),
        }

    def update_routing(
        self,
        cognitive_load: float,
        threat_urgency: float = 0.0,
    ) -> PulvinarRoutingState:
        """Dynamically calibrate transmission bandwidth across all transthalamic routes.

        Args:
            cognitive_load: Current prefrontal cognitive load in [0, 1].
            threat_urgency: Imminent hazard proximity / threat level in [0, 1].

        Returns:
            PulvinarRoutingState indicating active and throttled channels.
        """
        clamped_load = max(0.0, min(1.0, float(cognitive_load)))
        clamped_urgency = max(0.0, min(1.0, float(threat_urgency)))

        throttled: list[str] = []

        for rid, route in self.routes.items():
            if clamped_urgency > 0.6:
                # Acute threat: open emergency motor escape, throttle long-term memory
                if rid == "threat_to_motor_escape":
                    route.bandwidth = 1.0
                elif rid == "episodic_to_working_memory":
                    route.bandwidth = 0.1
                    throttled.append(rid)
                else:
                    route.bandwidth = 0.5
            elif clamped_load > 0.7:
                # High cognitive load: throttle peripheral sensory stream to prevent distraction
                if rid == "perception_to_deliberation":
                    route.bandwidth = max(0.2, 1.0 - clamped_load * 0.8)
                    throttled.append(rid)
                elif rid == "episodic_to_working_memory":
                    route.bandwidth = 0.4
                    throttled.append(rid)
                else:
                    route.bandwidth = 1.0
            else:
                # Normal cognitive regime: full bandwidth
                route.bandwidth = 1.0

        dominant = (
            "threat_to_motor_escape" if clamped_urgency > 0.5 else "perception_to_deliberation"
        )

        state = PulvinarRoutingState(
            cognitive_load=clamped_load,
            threat_urgency=clamped_urgency,
            routes=self.routes,
            throttled_routes=throttled,
            dominant_pathway=dominant,
        )
        self.last_state = state
        return state

    def can_transmit(
        self,
        route_id: str,
        message_cost: float = 0.2,
    ) -> bool:
        """Check whether a message can pass through the pulvinar route given its current bandwidth."""
        if route_id not in self.routes:
            return True  # Unregulated pathway defaults to passing

        route = self.routes[route_id]
        if not route.is_active:
            return False

        allowed = route.bandwidth >= max(0.05, float(message_cost))
        if allowed:
            self.total_transmissions += 1
        return allowed

    def reset(self) -> None:
        """Reset routing state and counters."""
        self._init_canonical_routes()
        self.last_state = None
        self.total_transmissions = 0
