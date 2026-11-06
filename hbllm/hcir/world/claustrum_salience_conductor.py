from __future__ import annotations

"""Claustrum Multimodal Salience Conductor & Perceptual Binding.

Biologically grounded in:
1. F. Crick & C. Koch (2005) - "What is the function of the claustrum?"
   (Philos Trans R Soc Lond B Biol Sci). The claustrum acts as the "conductor
   of the cortical orchestra", orchestrating synchrony across visual,
   somatosensory, and motor cortices into a coherent conscious gestalt.
2. J.B. Smith et al. (2019) - Anatomical and physiological properties of
   claustro-cortical circuits regulating cognitive salience and attention.

In HBLLM, the Claustrum Salience Conductor fuses heterogeneous perceptual
streams (visual contours, Gibsonian affordances, SNN hazard fields, and
entorhinal odometry) into an integrated spatial salience landscape and
synchronous perceptual gestalt.
"""

import math
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate


@dataclass
class UnifiedPerceptualGestalt:
    """Integrated multimodal perceptual state bound by claustrocortical synchrony."""

    focal_salience_peak: tuple[int, int] | None
    cross_modal_synchrony: float
    salience_map: dict[tuple[int, int], float] = field(default_factory=dict)
    dominant_modality: str = "balanced"
    bound_features: dict[str, Any] = field(default_factory=dict)
    grid_shape: tuple[int, int] = (10, 10)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export unified gestalt as a formal HCIR causal predicate."""
        peak_str = (
            f"({self.focal_salience_peak[0]},{self.focal_salience_peak[1]})"
            if self.focal_salience_peak
            else "none"
        )
        return CausalPredicate(
            variable="claustrum:gestalt_bound",
            operator="==",
            value=f"synchrony:{self.cross_modal_synchrony:.2f}:peak:{peak_str}",
            relation=f"dominant_modality:{self.dominant_modality}",
            target_entity="claustrum_conductor",
        )

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic HCIR transaction recording bound perceptual gestalt."""
        peak_list = (
            [self.focal_salience_peak[0], self.focal_salience_peak[1]]
            if self.focal_salience_peak
            else None
        )
        return [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="unified_perceptual_gestalt",
                node_data={
                    "focal_salience_peak": peak_list,
                    "cross_modal_synchrony": float(self.cross_modal_synchrony),
                    "dominant_modality": self.dominant_modality,
                    "bound_feature_keys": list(self.bound_features.keys()),
                    "grid_shape": [self.grid_shape[0], self.grid_shape[1]],
                },
            )
        ]


class ClaustrumSalienceConductor:
    """Multimodal binding conductor synthesizing spatial, threat, and goal streams."""

    def __init__(
        self,
        visual_weight: float = 0.25,
        hazard_weight: float = 0.45,
        goal_weight: float = 0.30,
        synchrony_threshold: float = 0.60,
    ) -> None:
        self.visual_weight = visual_weight
        self.hazard_weight = hazard_weight
        self.goal_weight = goal_weight
        self.synchrony_threshold = synchrony_threshold

    def bind_modalities(
        self,
        grid_shape: tuple[int, int],
        visual_entities: list[SpatialEntity] | None = None,
        hazard_map: dict[tuple[int, int], float] | None = None,
        goal_coords: list[tuple[int, int]] | None = None,
        affordance_fields: dict[str, Any] | None = None,
        agent_coord: tuple[int, int] | None = None,
    ) -> UnifiedPerceptualGestalt:
        """Bind heterogeneous sensory modalities into a unified spatial salience gestalt."""
        rows, cols = grid_shape
        salience_map: dict[tuple[int, int], float] = {
            (r, c): 0.0 for r in range(rows) for c in range(cols)
        }

        # 1. Visual Stream Salience
        visual_layer: dict[tuple[int, int], float] = {
            (r, c): 0.0 for r in range(rows) for c in range(cols)
        }
        if visual_entities:
            for entity in visual_entities:
                pos = getattr(entity, "grid_pos", None)
                if pos is None:
                    pos = getattr(entity, "position", None)
                if pos is None and hasattr(entity, "centroid"):
                    pos = (int(round(entity.centroid[0])), int(round(entity.centroid[1])))
                if pos and 0 <= pos[0] < rows and 0 <= pos[1] < cols:
                    # Interactive or non-wall entities evoke higher visual curiosity
                    is_salient_role = entity.role in {
                        EntityRole.MANIPULABLE,
                        EntityRole.RESOURCE,
                        EntityRole.AGENT,
                        EntityRole.ACTUATOR,
                    } or getattr(entity.role, "name", "").upper() in {
                        "CARGO",
                        "TOOL",
                        "AGENT",
                        "MANIPULABLE",
                        "RESOURCE",
                    }
                    weight = 0.8 if is_salient_role else 0.4
                    visual_layer[pos] = max(visual_layer[pos], weight)

        # 2. Hazard / Threat Stream Salience
        hazard_layer: dict[tuple[int, int], float] = {
            (r, c): 0.0 for r in range(rows) for c in range(cols)
        }
        if hazard_map:
            for (hr, hc), intensity in hazard_map.items():
                if 0 <= hr < rows and 0 <= hc < cols:
                    hazard_layer[(hr, hc)] = max(0.0, min(1.0, intensity))

        # 3. Goal / Attractor Stream Salience
        goal_layer: dict[tuple[int, int], float] = {
            (r, c): 0.0 for r in range(rows) for c in range(cols)
        }
        if goal_coords:
            for gr, gc in goal_coords:
                if 0 <= gr < rows and 0 <= gc < cols:
                    goal_layer[(gr, gc)] = 1.0
                    # Diffuse slight proximity attractor around goals
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = gr + dr, gc + dc
                            if 0 <= nr < rows and 0 <= nc < cols:
                                goal_layer[(nr, nc)] = max(goal_layer[(nr, nc)], 0.5)

        # 4. Integrate weighted multimodal salience
        max_salience = 0.0
        focal_peak: tuple[int, int] | None = None
        for coord in salience_map:
            s_val = (
                (visual_layer[coord] * self.visual_weight)
                + (hazard_layer[coord] * self.hazard_weight)
                + (goal_layer[coord] * self.goal_weight)
            )
            salience_map[coord] = s_val
            if s_val > max_salience:
                max_salience = s_val
                focal_peak = coord

        # 5. Measure Cross-Modal Synchrony / Alignment
        # Calculate alignment between peaks of active modalities
        active_peaks: list[tuple[int, int]] = []
        for layer in (visual_layer, hazard_layer, goal_layer):
            peak = max(layer, key=lambda k: layer[k]) if layer else None
            if peak and layer.get(peak, 0.0) > 0.1:
                active_peaks.append(peak)

        synchrony = 1.0
        if len(active_peaks) >= 2:
            distances: list[float] = []
            for i in range(len(active_peaks)):
                for j in range(i + 1, len(active_peaks)):
                    p1, p2 = active_peaks[i], active_peaks[j]
                    dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                    distances.append(dist)
            avg_dist = sum(distances) / len(distances)
            max_dist = math.hypot(rows, cols)
            synchrony = max(0.0, 1.0 - (avg_dist / max(1.0, max_dist)))
        elif len(active_peaks) == 1:
            synchrony = 0.8
        else:
            synchrony = 0.5

        # Determine dominant modality at focal salience peak
        if focal_peak is not None:
            v_val = visual_layer[focal_peak] * self.visual_weight
            h_val = hazard_layer[focal_peak] * self.hazard_weight
            g_val = goal_layer[focal_peak] * self.goal_weight
            if h_val > v_val and h_val > g_val and h_val > 0.05:
                dominant = "threat_avoidance"
            elif g_val > v_val and g_val > h_val and g_val > 0.05:
                dominant = "goal_directed"
            elif v_val > 0.05:
                dominant = "visual_curiosity"
            else:
                dominant = "balanced"
        else:
            dominant = "balanced"

        bound_features: dict[str, Any] = {
            "entity_count": len(visual_entities) if visual_entities else 0,
            "hazard_count": sum(1 for v in hazard_layer.values() if v > 0.1),
            "goal_count": len(goal_coords) if goal_coords else 0,
            "has_affordance": affordance_fields is not None,
            "agent_pos": agent_coord,
        }

        return UnifiedPerceptualGestalt(
            focal_salience_peak=focal_peak,
            cross_modal_synchrony=synchrony,
            salience_map=salience_map,
            dominant_modality=dominant,
            bound_features=bound_features,
            grid_shape=grid_shape,
        )
