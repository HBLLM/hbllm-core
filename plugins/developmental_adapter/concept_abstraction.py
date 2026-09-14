"""Concept Abstraction Engine (Stage D9).

Induces invariant categorical concept nodes into the HCIR cognitive graph
by clustering perceptual property bundles from unlabelled sensorimotor experience.
"""

from __future__ import annotations

import logging
from typing import Any

from .blank_brain import BlankBrainSubstrate
from .environment import BabyWorldEnvironment
from .types import (
    BabyObjectState,
    BeliefTransitionEvent,
    BeliefTransitionType,
    ConceptCluster,
)

logger = logging.getLogger(__name__)


class ConceptAbstractionEngine:
    """Unsupervised concept induction and categorization from perceptual bundles."""

    def __init__(self, substrate: BlankBrainSubstrate, env: BabyWorldEnvironment) -> None:
        self.substrate = substrate
        self.env = env
        self.discovered_concepts: dict[str, ConceptCluster] = {}

    def extract_entity_feature_bundle(self, obj: BabyObjectState) -> dict[str, Any]:
        """Extract invariant property signature from perceptual state."""
        return {
            "is_spherical": obj.object_type.value == "ball" or obj.rollable,
            "is_container": obj.is_container or obj.object_type.value in ("container", "box"),
            "is_elongated_tool": obj.is_tool or obj.tool_length > 0.4,
            "is_heavy": obj.mass >= 5.0,
            "is_light": obj.mass <= 2.5,
            "rollable": obj.rollable,
        }

    def induce_concepts_from_experience(
        self, objects: list[BabyObjectState] | None = None
    ) -> dict[str, ConceptCluster]:
        """Cluster perceived objects into stable concept categories without human labels."""
        objs = objects if objects is not None else list(self.env.objects.values())

        clusters: dict[str, list[str]] = {
            "SPHERICAL_BALL": [],
            "MANIPULABLE_BLOCK": [],
            "RECEPTACLE_CONTAINER": [],
            "REACH_TOOL": [],
            "HEAVY_OBSTACLE": [],
        }

        for obj in objs:
            features = self.extract_entity_feature_bundle(obj)
            if features["is_heavy"]:
                clusters["HEAVY_OBSTACLE"].append(obj.id)
            elif features["is_elongated_tool"] and features["is_light"]:
                clusters["REACH_TOOL"].append(obj.id)
            elif features["is_container"]:
                clusters["RECEPTACLE_CONTAINER"].append(obj.id)
            elif features["is_spherical"]:
                clusters["SPHERICAL_BALL"].append(obj.id)
            else:
                clusters["MANIPULABLE_BLOCK"].append(obj.id)

        # Build induced concept clusters
        for c_name, members in clusters.items():
            if members:
                exemplar = self.env.objects.get(members[0])
                archetype = self.extract_entity_feature_bundle(exemplar) if exemplar else {}
                cluster = ConceptCluster(
                    concept_name=c_name,
                    archetype_features=archetype,
                    exemplar_ids=members,
                    confidence=min(1.0, 0.5 + 0.1 * len(members)),
                )
                self.discovered_concepts[c_name] = cluster
                # Update blank brain learned stores
                self.substrate.semantic_concepts[c_name] = cluster
                self.substrate.object_categories[c_name] = members

                # Log belief transition
                if hasattr(self.substrate, "profile") and hasattr(
                    self.substrate.profile, "belief_transitions"
                ):
                    self.substrate.profile.belief_transitions.append(
                        BeliefTransitionEvent(
                            event_type=BeliefTransitionType.CONCEPT_INDUCED,
                            variable="concept",
                            condition=f"{c_name} (n={len(members)})",
                            posterior_confidence=cluster.confidence,
                            evidence={"exemplars": members},
                        )
                    )

        return self.discovered_concepts

    def categorize_novel_entity(self, novel_obj: BabyObjectState) -> str:
        """Classify a novel unseen entity into an acquired abstract concept."""
        if not self.discovered_concepts:
            self.induce_concepts_from_experience()

        features = self.extract_entity_feature_bundle(novel_obj)
        if features["is_heavy"]:
            return "HEAVY_OBSTACLE"
        if features["is_elongated_tool"] and features["is_light"]:
            return "REACH_TOOL"
        if features["is_container"]:
            return "RECEPTACLE_CONTAINER"
        if features["is_spherical"]:
            return "SPHERICAL_BALL"
        return "MANIPULABLE_BLOCK"
