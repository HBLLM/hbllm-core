"""Unit and Scientific Tests for Curriculum Relational Schemas & Dynamic Structural Transfer (A20 / E7).

Tests:
1. Pairwise structural mapping scores across the 5 canonical curriculum schemas.
2. Discovered transfer ordering without hardcoded heuristics: T1↔T2 > T1↔T5 > T1↔T4.
3. Physical and geometric constraint validation and failure rejection.
4. Adversarial tests: relational structure vs surface/entity naming independence.
5. End-to-end E7 lifelong curriculum continual learning matrix evaluation.
"""

from __future__ import annotations

from hbllm.brain.transfer.extractor import RelationalSchemaExtractor
from hbllm.brain.transfer.mapper import MappingStatus, StructureMappingEngine
from hbllm.brain.transfer.schema import RelationalSchema
from hbllm.experiment.cohorts import HBLLMCoreCohort
from hbllm.experiment.tasks import E7_LifelongCurriculumTask, _build_curriculum_observation
from hbllm.hcir.graph import (
    CognitiveGraph,
    HCIREdge,
    HCIREdgeType,
    PhysicalEntityNode,
)


def _observation_to_graph(obs) -> CognitiveGraph:
    """Helper to convert EnvironmentObservation into CognitiveGraph."""
    graph = CognitiveGraph()
    for ent in obs.visible_entities:
        node = PhysicalEntityNode(
            id=ent["id"],
            entity_type=ent.get("type", "physical_entity"),
            properties=dict(ent.get("properties", {})),
        )
        graph.add_node(node)

    for rel in obs.spatial_relations:
        src = rel["source"]
        tgt = rel["target"]
        rtype_str = rel.get("relation", "SUPPORTS")
        try:
            rtype = HCIREdgeType[rtype_str]
        except KeyError:
            rtype = HCIREdgeType.SUPPORTS
        edge_id = f"edge_{src}_{rtype_str}_{tgt}"
        if graph.has_node(src) and graph.has_node(tgt):
            graph.add_edge(HCIREdge(id=edge_id, edge_type=rtype, sources=[src], targets=[tgt]))

    return graph


def _extract_task_schema(task_name: str) -> RelationalSchema:
    """Helper to induce canonical schema directly from task exemplar graph."""
    obs = _build_curriculum_observation(task_name, prefix="src_exemplar")
    graph = _observation_to_graph(obs)
    return RelationalSchemaExtractor().extract_schema_from_graph(graph, name=task_name)


class TestCurriculumStructuralMapping:
    """Evaluates dynamic structure mapping between canonical schemas and grounded task graphs."""

    def test_all_schemas_map_perfectly_to_their_own_grounded_graph(self) -> None:
        """Every canonical schema must achieve high alignment and APPLICABLE status on its own domain."""
        mapper = StructureMappingEngine()
        task_names = [
            "T1_SpatialStacking",
            "T2_ContainerPacking",
            "T3_BalanceBeam",
            "T4_ObstacleNav",
            "T5_ToolAffordance",
        ]

        for task_name in task_names:
            schema = _extract_task_schema(task_name)
            obs = _build_curriculum_observation(task_name, prefix="self_test")
            graph = _observation_to_graph(obs)

            result = mapper.map_schema_to_target(schema, graph)
            assert result.status == MappingStatus.APPLICABLE, (
                f"Failed for {task_name}: {result.violated_constraints}"
            )
            assert result.relational_alignment_score >= 0.60, (
                f"Low alignment score for {task_name}: {result.relational_alignment_score}"
            )
            assert len(result.violated_constraints) == 0

    def test_discovered_cross_domain_structural_transfer_hierarchy(self) -> None:
        """Dynamic mapping from T1 (Spatial Stacking) must discover higher alignment to T2 (Container) and T3 (Balance) than to T4 (Navigation)."""
        mapper = StructureMappingEngine()
        t1_schema = _extract_task_schema("T1_SpatialStacking")

        # Target 1: T2 Container Packing (shares SUPPORTS, containment stability)
        t2_obs = _build_curriculum_observation("T2_ContainerPacking", prefix="t2_target")
        t2_graph = _observation_to_graph(t2_obs)
        res_t1_t2 = mapper.map_schema_to_target(t1_schema, t2_graph)

        # Target 2: T3 Balance Beam (shares SUPPORTS, resting equilibrium)
        t3_obs = _build_curriculum_observation("T3_BalanceBeam", prefix="t3_target")
        t3_graph = _observation_to_graph(t3_obs)
        res_t1_t3 = mapper.map_schema_to_target(t1_schema, t3_graph)

        # Target 3: T4 Obstacle Navigation (orthogonal path/avoid topology)
        t4_obs = _build_curriculum_observation("T4_ObstacleNav", prefix="t4_target")
        t4_graph = _observation_to_graph(t4_obs)
        res_t1_t4 = mapper.map_schema_to_target(t1_schema, t4_graph)

        # Discovered transfer score hierarchy: T1->T2 >= 0.50, T1->T3 >= 0.50, T1->T4 < 0.35
        assert res_t1_t2.relational_alignment_score > res_t1_t4.relational_alignment_score
        assert res_t1_t3.relational_alignment_score > res_t1_t4.relational_alignment_score
        assert res_t1_t2.relational_alignment_score >= 0.50
        assert res_t1_t3.relational_alignment_score >= 0.50
        assert res_t1_t4.relational_alignment_score < 0.35


class TestPhysicalConstraintValidation:
    """Verifies that physical/geometric constraint violations cause mapping penalties or rejection."""

    def test_non_rigid_base_violates_stacking_constraint(self) -> None:
        """Mapping T1 onto an observation with a flexible/deformable base must flag constraint violations."""
        mapper = StructureMappingEngine()
        t1_schema = _extract_task_schema("T1_SpatialStacking")

        # Observation with flexible base
        graph = CognitiveGraph()
        graph.add_node(
            PhysicalEntityNode(
                id="foam_sponge",
                properties={"rigidity": "flexible", "surface": "flat", "stable": False},
            )
        )
        graph.add_node(
            PhysicalEntityNode(
                id="heavy_brick",
                properties={"stable": True, "mass": 5.0},
            )
        )
        graph.add_edge(
            HCIREdge(
                id="e_sponge_brick",
                edge_type=HCIREdgeType.SUPPORTS,
                sources=["foam_sponge"],
                targets=["heavy_brick"],
            )
        )

        result = mapper.map_schema_to_target(t1_schema, graph)
        assert len(result.violated_constraints) > 0
        assert any("rigidity" in v or "rigid" in v for v in result.violated_constraints)
        assert result.status in (MappingStatus.REJECTED, MappingStatus.PARTIALLY_APPLICABLE)

    def test_closed_container_violates_packing_constraint(self) -> None:
        """Mapping T2 onto a sealed/closed container must flag constraint violations."""
        mapper = StructureMappingEngine()
        t2_schema = _extract_task_schema("T2_ContainerPacking")

        graph = CognitiveGraph()
        graph.add_node(
            PhysicalEntityNode(
                id="sealed_jar",
                properties={"open": False, "has_cavity": True},
            )
        )
        graph.add_node(
            PhysicalEntityNode(
                id="marble",
                properties={"fits_inside": True},
            )
        )
        graph.add_node(
            PhysicalEntityNode(
                id="interior_space",
                properties={"is_space": True},
            )
        )

        result = mapper.map_schema_to_target(t2_schema, graph)
        assert len(result.violated_constraints) > 0
        assert any("open" in v for v in result.violated_constraints)


class TestAdversarialStructuralIndependence:
    """Verifies that A20 transfers relational structure rather than superficial entity names."""

    def test_arbitrary_entity_names_with_matching_structure_yields_high_transfer(self) -> None:
        """Graphs with completely arbitrary entity names but isomorphic relations must achieve high structural alignment."""
        mapper = StructureMappingEngine()
        t1_schema = _extract_task_schema("T1_SpatialStacking")

        # Arbitrary names: 'quantum_substrate' and 'plasma_core'
        graph = CognitiveGraph()
        graph.add_node(
            PhysicalEntityNode(
                id="quantum_substrate",
                properties={"rigidity": "rigid", "surface": "flat", "stable": True},
            )
        )
        graph.add_node(
            PhysicalEntityNode(
                id="plasma_core",
                properties={"stable": True, "mass": 1.0},
            )
        )
        graph.add_edge(
            HCIREdge(
                id="e1",
                edge_type=HCIREdgeType.SUPPORTS,
                sources=["quantum_substrate"],
                targets=["plasma_core"],
            )
        )
        graph.add_edge(
            HCIREdge(
                id="e2",
                edge_type=HCIREdgeType.STABLE_FOR,
                sources=["quantum_substrate"],
                targets=["plasma_core"],
            )
        )
        graph.add_edge(
            HCIREdge(
                id="e3",
                edge_type=HCIREdgeType.ABOVE,
                sources=["plasma_core"],
                targets=["quantum_substrate"],
            )
        )

        result = mapper.map_schema_to_target(t1_schema, graph)
        assert result.status == MappingStatus.APPLICABLE
        assert result.relational_alignment_score >= 0.70
        assert len(result.role_bindings) == 2

    def test_identical_entity_names_with_adversarial_relations_yields_low_transfer(self) -> None:
        """Graphs retaining entity names but replacing functional edges with conflicting relations must receive low scores."""
        mapper = StructureMappingEngine()
        t1_schema = _extract_task_schema("T1_SpatialStacking")

        graph = CognitiveGraph()
        graph.add_node(
            PhysicalEntityNode(
                id="Base",
                properties={"rigidity": "rigid", "surface": "flat", "stable": True},
            )
        )
        graph.add_node(
            PhysicalEntityNode(
                id="Payload",
                properties={"stable": True, "mass": 1.0},
            )
        )
        # Adversarial contradictory edge
        graph.add_edge(
            HCIREdge(
                id="e_conflict",
                edge_type=HCIREdgeType.CONTRADICTS,
                sources=["Base"],
                targets=["Payload"],
            )
        )

        result = mapper.map_schema_to_target(t1_schema, graph)
        assert result.relational_alignment_score < 0.25


class TestE7LifelongCurriculumContinualTransfer:
    """Evaluates the full E7 lifelong curriculum sequential training and R_{i,j} matrix."""

    def test_e7_continual_learning_forward_transfer(self) -> None:
        """HBLLMCoreCohort in E7 must achieve genuine positive forward transfer from dynamic structural mapping."""
        cohort = HBLLMCoreCohort()
        task = E7_LifelongCurriculumTask()

        result = task.evaluate(cohort, seed=42)

        assert result.task_id == "E7_LifelongCurriculum"
        assert result.continual_matrix_r is not None
        assert len(result.continual_matrix_r) == 5

        # Diagonal elements (performance after training on task) must be near 1.0
        for i in range(5):
            assert result.continual_matrix_r[i][i] >= 0.90

        # R_{0, 1} (T1 trained -> zero-shot T2 evaluated) must reflect discovered structural transfer (> 0.40)
        assert result.continual_matrix_r[0][1] >= 0.40

        # Overall Forward Transfer (FWT) must be positive and non-trivial
        assert result.fwt > 0.0
        # Backward Transfer (BWT) must exhibit minimal catastrophic forgetting
        assert result.bwt >= -0.05
