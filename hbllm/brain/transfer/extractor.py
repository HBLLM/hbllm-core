"""Relational Schema Extractor for A20.

Induces generalized structural graph templates from grounded episodes in HCIR.
Generalizes concrete node instances into variable roles while preserving
relational topology, physical preconditions, and predictive consequences.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.transfer.schema import (
    ActionTemplate,
    ConsequenceTemplate,
    RelationalSchema,
    SchemaConstraint,
    SchemaLifecycleStatus,
    SchemaRelation,
    SchemaRole,
)
from hbllm.hcir.graph import CognitiveGraph

logger = logging.getLogger(__name__)


@dataclass
class GroundedEpisode:
    """A recorded historical interaction episode in canonical HCIR."""

    episode_id: str = field(default_factory=lambda: f"ep_{uuid.uuid4().hex[:8]}")
    graph: CognitiveGraph = field(default_factory=CognitiveGraph)
    action_sequence: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    observed_consequences: list[str] = field(default_factory=list)
    is_success: bool = True


class RelationalSchemaExtractor:
    """Induces reusable relational schemas from grounded interaction episodes."""

    def __init__(self) -> None:
        self.schema_library: dict[str, RelationalSchema] = {}

    def register_schema(self, schema: RelationalSchema) -> None:
        self.schema_library[schema.schema_id] = schema

    def get_schema(self, schema_id: str) -> RelationalSchema | None:
        return self.schema_library.get(schema_id)

    def all_schemas(self) -> list[RelationalSchema]:
        return list(self.schema_library.values())

    def extract_support_schema(self, episode: GroundedEpisode) -> RelationalSchema:
        """Extract a generalized 2-tier or multi-tier support schema from a stacking episode."""
        roles = [
            SchemaRole(role_id="Base", type_requirement="physical_entity"),
            SchemaRole(role_id="Payload", type_requirement="physical_entity"),
        ]
        relations = [
            SchemaRelation(source_role="Payload", edge_type="LOCATED_ON", target_role="Base"),
            SchemaRelation(source_role="Base", edge_type="SUPPORTS", target_role="Payload"),
        ]
        constraints = [
            SchemaConstraint(role_id="Base", property_key="geometry", expected_value="flat"),
        ]
        actions = [
            ActionTemplate(
                operator_name="STACK",
                role_parameters={"item_id": "Payload", "base_id": "Base"},
            )
        ]
        consequences = [
            ConsequenceTemplate(
                consequence_type="stable_support",
                predicted_edge_type="LOCATED_ON",
                source_role="Payload",
                target_role="Base",
            )
        ]

        schema = RelationalSchema(
            schema_id=f"schema_support_{uuid.uuid4().hex[:6]}",
            name="Support-Chain",
            roles=roles,
            relations=relations,
            constraints=constraints,
            action_templates=actions,
            predicted_consequences=consequences,
            source_episode_ids=[episode.episode_id],
            status=SchemaLifecycleStatus.TRANSFERABLE,
            alpha_success=4.0,
            beta_failure=1.0,
        )
        self.register_schema(schema)
        return schema

    def extract_containment_schema(self, episode: GroundedEpisode) -> RelationalSchema:
        """Extract a generalized containment schema from an episode."""
        roles = [
            SchemaRole(role_id="Container", type_requirement="physical_entity"),
            SchemaRole(role_id="Item", type_requirement="physical_entity"),
        ]
        relations = [
            SchemaRelation(source_role="Item", edge_type="LOCATED_IN", target_role="Container"),
        ]
        constraints = [
            SchemaConstraint(role_id="Container", property_key="is_closed", expected_value=False),
        ]
        actions = [
            ActionTemplate(
                operator_name="PUT_IN",
                role_parameters={"item_id": "Item", "container_id": "Container"},
            )
        ]
        consequences = [
            ConsequenceTemplate(
                consequence_type="contained",
                predicted_edge_type="LOCATED_IN",
                source_role="Item",
                target_role="Container",
            )
        ]

        schema = RelationalSchema(
            schema_id=f"schema_containment_{uuid.uuid4().hex[:6]}",
            name="Container-Payload",
            roles=roles,
            relations=relations,
            constraints=constraints,
            action_templates=actions,
            predicted_consequences=consequences,
            source_episode_ids=[episode.episode_id],
            status=SchemaLifecycleStatus.TRANSFERABLE,
            alpha_success=4.0,
            beta_failure=1.0,
        )
        self.register_schema(schema)
        return schema

    def extract_tool_use_schema(self, episode: GroundedEpisode) -> RelationalSchema:
        """Extract a generalized tool-mediated displacement schema."""
        roles = [
            SchemaRole(role_id="Agent", type_requirement="agent"),
            SchemaRole(role_id="Tool", type_requirement="physical_entity"),
            SchemaRole(role_id="Target", type_requirement="physical_entity"),
        ]
        relations = [
            SchemaRelation(source_role="Agent", edge_type="HOLDS", target_role="Tool"),
            SchemaRelation(source_role="Tool", edge_type="APPLIES_TO", target_role="Target"),
        ]
        constraints = [
            SchemaConstraint(role_id="Tool", property_key="is_rigid", expected_value=True),
        ]
        actions = [
            ActionTemplate(
                operator_name="PUSH",
                role_parameters={"target_id": "Target", "dx": 2.0, "dy": 0.0},
            )
        ]
        consequences = [
            ConsequenceTemplate(
                consequence_type="displacement",
                predicted_edge_type="DISPLACED",
                source_role="Tool",
                target_role="Target",
            )
        ]

        schema = RelationalSchema(
            schema_id=f"schema_tool_{uuid.uuid4().hex[:6]}",
            name="Tool-Intermediary",
            roles=roles,
            relations=relations,
            constraints=constraints,
            action_templates=actions,
            predicted_consequences=consequences,
            source_episode_ids=[episode.episode_id],
            status=SchemaLifecycleStatus.TRANSFERABLE,
            alpha_success=3.0,
            beta_failure=1.0,
        )
        self.register_schema(schema)
        return schema

    def extract_schema_from_graph(
        self, graph: CognitiveGraph, name: str = "induced_schema", episode_id: str = ""
    ) -> RelationalSchema:
        """Induce a generalized relational schema from an observed CognitiveGraph."""
        nodes = graph.all_nodes()
        roles: list[SchemaRole] = []
        constraints: list[SchemaConstraint] = []
        role_map: dict[str, str] = {}

        functional_invariants = {
            "rigidity",
            "is_rigid",
            "open",
            "is_open",
            "has_cavity",
            "is_pivot",
            "is_level",
            "is_mobile",
            "blocks_path",
            "is_graspable",
        }

        for idx, node in enumerate(nodes):
            role_id = f"Role_{idx}_{node.id.split('_')[-1]}"
            role_map[node.id] = role_id
            props = dict(getattr(node, "properties", {}) or {})
            roles.append(
                SchemaRole(
                    role_id=role_id,
                    type_requirement=getattr(node, "entity_type", "physical_entity"),
                    required_properties={},
                )
            )
            for k, v in props.items():
                if k in functional_invariants:
                    constraints.append(
                        SchemaConstraint(role_id=role_id, property_key=k, expected_value=v)
                    )

        relations: list[SchemaRelation] = []
        for edge in graph.all_edges():
            for src in edge.sources:
                for tgt in edge.targets:
                    src_role = role_map.get(src)
                    tgt_role = role_map.get(tgt)
                    if src_role and tgt_role:
                        etype_str = str(
                            edge.edge_type.value
                            if hasattr(edge.edge_type, "value")
                            else edge.edge_type
                        )
                        relations.append(
                            SchemaRelation(
                                source_role=src_role,
                                edge_type=etype_str,
                                target_role=tgt_role,
                            )
                        )

        schema = RelationalSchema(
            schema_id=f"schema_induced_{uuid.uuid4().hex[:6]}",
            name=name,
            roles=roles,
            relations=relations,
            constraints=constraints,
            source_episode_ids=[episode_id] if episode_id else [],
            status=SchemaLifecycleStatus.TRANSFERABLE,
            alpha_success=4.0,
            beta_failure=1.0,
        )
        self.register_schema(schema)
        return schema

    def extract_schema_from_observation(
        self, observation: Any, name: str = "induced_schema", episode_id: str = ""
    ) -> RelationalSchema:
        """Induce a generalized relational schema directly from an EnvironmentObservation."""
        entities = getattr(observation, "visible_entities", []) or []
        spatial_rels = getattr(observation, "spatial_relations", []) or []

        roles: list[SchemaRole] = []
        constraints: list[SchemaConstraint] = []
        role_map: dict[str, str] = {}

        functional_invariants = {
            "rigidity",
            "is_rigid",
            "open",
            "is_open",
            "has_cavity",
            "is_pivot",
            "is_level",
            "is_mobile",
            "blocks_path",
            "is_graspable",
        }

        for idx, ent in enumerate(entities):
            ent_id = ent.get("id", f"ent_{idx}")
            role_id = f"Role_{idx}_{ent_id.split('_')[-1]}"
            role_map[ent_id] = role_id
            props = dict(ent.get("properties", {}) or {})
            roles.append(
                SchemaRole(
                    role_id=role_id,
                    type_requirement=ent.get("type", "physical_entity"),
                    required_properties={},
                )
            )
            for k, v in props.items():
                if k in functional_invariants:
                    constraints.append(
                        SchemaConstraint(role_id=role_id, property_key=k, expected_value=v)
                    )

        relations: list[SchemaRelation] = []
        for rel in spatial_rels:
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            rtype_str = rel.get("relation", "SUPPORTS")
            src_role = role_map.get(src)
            tgt_role = role_map.get(tgt)
            if src_role and tgt_role:
                relations.append(
                    SchemaRelation(
                        source_role=src_role,
                        edge_type=str(rtype_str),
                        target_role=tgt_role,
                    )
                )

        schema = RelationalSchema(
            schema_id=f"schema_induced_{uuid.uuid4().hex[:6]}",
            name=name,
            roles=roles,
            relations=relations,
            constraints=constraints,
            source_episode_ids=[episode_id] if episode_id else [],
            status=SchemaLifecycleStatus.TRANSFERABLE,
            alpha_success=4.0,
            beta_failure=1.0,
        )
        self.register_schema(schema)
        return schema
