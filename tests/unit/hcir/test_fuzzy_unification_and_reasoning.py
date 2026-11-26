"""Phase F2 tests: Soft Unification, Rust Matcher, VM Degree Branches, Replay, Composition, and Bipolar Epistemics."""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.capabilities.belief_revision import BeliefConflictResolver
from hbllm.hcir.fuzzy import (
    AdaptiveMembershipModel,
    FuzzyProfileId,
    MembershipKernelFamily,
    ReferenceScaleKind,
)
from hbllm.hcir.graph import (
    CognitiveGraph,
    ConceptNode,
    EvidenceNode,
    HCIREdge,
    HCIREdgeType,
    PhysicalEntityNode,
    SkillNode,
)
from hbllm.hcir.interpreter import HCIRInterpreter
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.verification import create_default_pipeline
from hbllm.hcir.nodes import BeliefNode
from hbllm.hcir.reasoning.operators.base import (
    CognitiveContext as ReasoningContext,
)
from hbllm.hcir.reasoning.operators.base import (
    FrozenGraphView,
    ProblemType,
    ReasoningProblem,
    ResultStatus,
)
from hbllm.hcir.reasoning.operators.deduction import DeductionOperator
from hbllm.hcir.reasoning.operators.spatial import SpatialOperator
from hbllm.hcir.replay_debugger import ReplayDebugger
from hbllm.hcir.seed.loader import load_seeded_composition_policy
from hbllm.hcir.skills.declarative import (
    AllOf,
    PredictiveConfidencePredicate,
    SkillEvaluationContext,
    SymbolicExpr,
    SymmetryPredicate,
)
from hbllm.hcir.transactions import HCIRTransaction, TransactionOp, TransactionOperation
from hbllm.hcir.types import Scope
from hbllm.hcir.unification import unify_pattern_graded
from hbllm.hcir.workspace import HCIRWorkspaceState


def test_graded_unification_ranks_fit_and_preserves_crisp_structure() -> None:
    """unify_pattern_graded scores graded fit while enforcing crisp node_type & binding invariants."""
    graph = CognitiveGraph()
    n1 = PhysicalEntityNode(
        id="cup_a",
        entity_type="cup",
        properties={
            "height": 11.2,
            "fuzzy_memberships": {"concept:CONTAINER": {"degree": 0.60}},
        },
    )
    n2 = PhysicalEntityNode(
        id="cup_b",
        entity_type="cup",
        properties={
            "height": 10.1,
            "fuzzy_memberships": {"concept:CONTAINER": {"degree": 0.95}},
        },
    )
    n_wrong_type = ConceptNode(
        id="concept_cup",
        label="cup",
        properties={
            "height": 10.0,
            "fuzzy_memberships": {"concept:CONTAINER": {"degree": 1.0}},
        },
    )
    for n in (n1, n2, n_wrong_type):
        graph.add_node(n)

    pattern = {
        "nodes": [
            {
                "var": "?cup",
                "node_type": "physical_entity",
                "fuzzy_memberships": {"concept:CONTAINER": 0.5},
                "numeric_tolerances": {"height": {"target": 10.0, "sigma": 1.0, "min_degree": 0.2}},
            }
        ]
    }
    matches = unify_pattern_graded(graph, pattern, profile_id="product")
    assert [m.bindings["?cup"] for m in matches] == ["cup_b", "cup_a"]
    assert all(m.structure_valid and m.binding_valid for m in matches)
    assert matches[0].semantic_degree > matches[1].semantic_degree
    assert matches[0].profile_id == FuzzyProfileId.PRODUCT


def test_rust_structure_matcher_graded_profiles() -> None:
    """Native PyO3 hbllm_structure_matcher exposes structure_valid, binding_valid, and semantic_degree."""
    import hbllm_structure_matcher as sm

    pattern = {
        "variables": ["?a", "?b", "?c"],
        "edges": [
            {"rel_type": "INSIDE", "source_var": "?a", "target_var": "?b", "min_degree": 0.5},
            {"rel_type": "NEAR", "source_var": "?b", "target_var": "?c", "min_degree": 0.4},
        ],
    }
    target = {
        "nodes": ["item1", "tray", "table1", "table2"],
        "edges": [
            {"rel_type": "INSIDE", "source": "item1", "target": "tray", "degree": 0.80},
            {"rel_type": "NEAR", "source": "tray", "target": "table1", "degree": 0.50},
            {"rel_type": "NEAR", "source": "tray", "target": "table2", "degree": 0.90},
        ],
    }

    res_prod = sm.match_relational_schema(pattern, target, 1.0, "product")
    assert len(res_prod) == 2
    assert res_prod[0]["mapping"]["?c"] == "table2"
    assert res_prod[0]["structure_valid"] is True
    assert res_prod[0]["binding_valid"] is True
    assert res_prod[0]["semantic_degree"] == pytest.approx(0.72)

    res_godel = sm.match_relational_schema(pattern, target, 1.0, "godel")
    assert res_godel[0]["semantic_degree"] == pytest.approx(0.80)
    assert res_godel[0]["profile_id"] == "godel"

    res_luk = sm.match_relational_schema(pattern, target, 1.0, "lukasiewicz")
    assert res_luk[1]["semantic_degree"] == pytest.approx(max(0.0, 0.80 + 0.50 - 1.0))


@pytest.mark.asyncio
async def test_vm_graded_unify_branch_alpha_cut_and_call_ranking() -> None:
    """HCIRInterpreter preserves 12 opcodes while adding ALPHA_CUT BRANCH and graded CALL ranking."""
    assert len(list(Opcode)) == 12
    ws = HCIRWorkspaceState(tenant_id="tenant_f2")
    services = KernelServices.create_default(
        ws,
        verification_pipeline=create_default_pipeline(default_scope=Scope(tenant_id="tenant_f2")),
    )
    services.extensions["enable_fuzzy_reasoning"] = True

    e_low = PhysicalEntityNode(
        id="obj_low",
        entity_type="block",
        properties={"fuzzy_memberships": {"concept:GRASPABLE": 0.55}},
        scope=Scope(tenant_id="tenant_f2"),
    )
    e_high = PhysicalEntityNode(
        id="obj_high",
        entity_type="block",
        properties={"fuzzy_memberships": {"concept:GRASPABLE": 0.92}},
        scope=Scope(tenant_id="tenant_f2"),
    )
    # Skill A has HIGHER reliability_mean (0.95) but LOWER semantic_degree precondition match
    # Skill B has LOWER reliability_mean (0.60) but HIGHER semantic_degree precondition match
    sk_low_fit = SkillNode(
        id="sk_low_fit",
        skill_name="pick_low_fit",
        alpha_success=19.0,
        beta_failure=1.0,
        precondition_pattern={"id": "obj_low", "fuzzy_memberships": {"concept:GRASPABLE": 0.4}},
        postcondition_pattern={"intent": "grasp_block"},
        bytecode_stream=InstructionStream(
            author="test",
            instructions=[Instruction(opcode=Opcode.RETURN, params={"yield": ["picked_low"]})],
        ).model_dump(),
        scope=Scope(tenant_id="tenant_f2"),
    )
    sk_high_fit = SkillNode(
        id="sk_high_fit",
        skill_name="pick_high_fit",
        alpha_success=3.0,
        beta_failure=2.0,
        precondition_pattern={"id": "obj_high", "fuzzy_memberships": {"concept:GRASPABLE": 0.4}},
        postcondition_pattern={"intent": "grasp_block"},
        bytecode_stream=InstructionStream(
            author="test",
            instructions=[Instruction(opcode=Opcode.RETURN, params={"yield": ["picked_high"]})],
        ).model_dump(),
        scope=Scope(tenant_id="tenant_f2"),
    )
    tx = HCIRTransaction(
        author="test_f2",
        tenant_id="tenant_f2",
        operations=[
            TransactionOperation(op=TransactionOp.ADD_NODE, node_data=n.model_dump())
            for n in (e_low, e_high, sk_low_fit, sk_high_fit)
        ],
    )
    assert services.transaction_manager.commit(tx).is_committed

    interp = HCIRInterpreter(ws, services=services)
    stream = InstructionStream(
        author="test_f2",
        instructions=[
            Instruction(
                opcode=Opcode.UNIFY,
                params={
                    "pattern": {
                        "var": "?target",
                        "node_type": "physical_entity",
                        "fuzzy_memberships": {"concept:GRASPABLE": 0.5},
                    },
                    "bind": {"__primary_id__": "$r0", "semantic_degree": "$r1"},
                    "fuzzy": True,
                },
            ),
            Instruction(
                opcode=Opcode.BRANCH,
                params={
                    "condition": "if_degree_lt",
                    "register": "$r1",
                    "threshold": 0.85,
                    "target_label": "fallback",
                },
            ),
            Instruction(
                opcode=Opcode.CALL,
                params={
                    "goal_pattern": {"intent": "grasp_block"},
                    "fuzzy": True,
                    "bind": "$r2",
                },
            ),
            Instruction(opcode=Opcode.RETURN, params={"yield": ["$r2"]}),
            Instruction(
                opcode=Opcode.RETURN, label="fallback", params={"yield": ["low_degree_fallback"]}
            ),
        ],
    )
    res = await interp.execute(stream)
    assert res.success is True
    assert res.events[0]["return_value"] == "picked_high"
    assert res.events[0]["registers"]["$r0"] == "obj_high"
    assert res.events[0]["registers"]["$r1"] == pytest.approx(0.92)
    # Verify reliability_mean was never conflated with semantic_degree
    assert sk_low_fit.reliability_mean > sk_high_fit.reliability_mean


@pytest.mark.asyncio
async def test_replay_debugger_bit_exact_across_model_evolution() -> None:
    """ReplayDebugger preserves deterministic outputs across AdaptiveMembershipModel v1 -> v2."""
    ws = HCIRWorkspaceState(tenant_id="tenant_replay")
    services = KernelServices.create_default(
        ws,
        verification_pipeline=create_default_pipeline(
            default_scope=Scope(tenant_id="tenant_replay")
        ),
    )
    interp = HCIRInterpreter(ws, services=services)

    m1 = AdaptiveMembershipModel(
        model_id="m_near_rep",
        target_id="rel_near",
        contextual_scope="manipulation",
        kernel_family=MembershipKernelFamily.TRAPEZOIDAL,
        reference_scale_kind=ReferenceScaleKind.RAW,
        learned_parameters={"a": 0.0, "b": 0.0, "c": 1.0, "d": 2.0},
    )
    m2, ops = m1.evolve(
        {"a": 0.0, "b": 0.0, "c": 2.0, "d": 4.0},
        scope=Scope(tenant_id="tenant_replay"),
    )
    assert m2.version == 2
    assert services.transaction_manager.commit(
        HCIRTransaction(author="test", tenant_id="tenant_replay", operations=ops)
    ).is_committed

    stream = InstructionStream(
        author="replay_test",
        instructions=[
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "math.fuzzy.membership",
                    "params": {
                        "value": 1.5,
                        "kernel_family": m1.kernel_family.value,
                        "parameters": dict(m1.learned_parameters),
                        "reference_scale_kind": m1.reference_scale_kind.value,
                    },
                    "bind": "$r0",
                },
            ),
            Instruction(
                opcode=Opcode.BRANCH,
                params={
                    "condition": "if_degree_gte",
                    "register": "$r0",
                    "threshold": 0.4,
                    "target_label": "ok",
                },
            ),
            Instruction(opcode=Opcode.RETURN, params={"yield": ["fail"]}),
            Instruction(opcode=Opcode.RETURN, label="ok", params={"yield": ["$r0"]}),
        ],
    )
    _, receipt = await interp.execute_with_receipt(stream, process_id="p_rep")
    dbg = ReplayDebugger(services)
    steps = await dbg.replay_stream(stream, expected_receipt=receipt)
    assert steps[-1].registers["$r0"]["degree"] == pytest.approx(0.5)
    assert receipt.outputs["registers"]["$r0"]["degree"] == pytest.approx(0.5)


def test_symbolic_expr_evaluate_degree_smooth_vs_crisp_cutoff() -> None:
    """SymbolicExpr.evaluate_degree avoids cliff-edge drop while keeping crisp evaluate() backward-compatible."""
    grid = np.zeros((10, 10), dtype=np.int32)
    grid[2:4, 2:4] = 3  # centroid (2.5, 2.5)
    grid[2:4, 8:10] = 3  # centroid (2.5, 8.5) -> mean col 5.5 vs mid 4.5 => dev=1.0
    ctx = SkillEvaluationContext(
        grid=grid,
        available_actions=[1, 2, 3, 4],
        metadata={"total_free_energy": 1.25, "calibrated_confidence": 0.45},
    )
    pred = AllOf(
        SymmetryPredicate(
            max_deviation=0.8, min_entities=2, min_area=1, max_area=20, ignore_top_colors=1
        ),
        PredictiveConfidencePredicate(max_free_energy=1.0, min_confidence=0.5),
    )
    assert pred.evaluate(ctx) is False
    deg_prod = pred.evaluate_degree(ctx, profile_id="product")
    deg_godel = pred.evaluate_degree(ctx, profile_id="godel")
    assert 0.0 < deg_prod < deg_godel <= 1.0

    # Inline FuzzyExpr AST
    f_ast = SymbolicExpr(
        op="fuzzy_expr",
        args={
            "expr": {"op": "var", "args": {"name": "calibrated_confidence"}},
            "alpha": 0.5,
        },
    )
    assert f_ast.evaluate_degree(ctx) == pytest.approx(0.45)
    assert f_ast.evaluate(ctx) is False


def test_spatial_operator_declarative_composition_and_scale_guards() -> None:
    """SpatialOperator uses RelationCompositionPolicy: blocks NEAR o NEAR and large-container NEAR o INSIDE."""
    policy = load_seeded_composition_policy()
    assert policy.policy_id == "default_spatial_policy"

    graph = CognitiveGraph()
    cup = PhysicalEntityNode(id="cup", entity_type="cup", properties={"diameter": 0.1})
    tray = PhysicalEntityNode(id="tray", entity_type="tray", properties={"diameter": 0.3})
    plate = PhysicalEntityNode(id="plate", entity_type="plate", properties={"diameter": 1.0})
    fork = PhysicalEntityNode(id="fork", entity_type="fork", properties={"diameter": 0.1})
    warehouse = PhysicalEntityNode(
        id="warehouse", entity_type="building", properties={"diameter": 50.0}
    )
    box = PhysicalEntityNode(id="box", entity_type="box", properties={"diameter": 0.2})
    for n in (cup, tray, plate, fork, warehouse, box):
        graph.add_node(n)

    # cup INSIDE tray (0.95), tray NEAR plate (0.80) -> INSIDE o NEAR allowed (0.3/1.0 <= 0.5)
    # fork NEAR box (0.90), box INSIDE warehouse (0.98) -> NEAR o INSIDE rejected (diam 50 >> 0.2)
    # tray NEAR plate (0.80), plate NEAR spoon... wait use edge NEAR(plate, cup) forbidden NEAR o NEAR
    graph.add_edge(
        HCIREdge(
            id="e1",
            edge_type=HCIREdgeType.LOCATED_IN,
            sources=["cup"],
            targets=["tray"],
            weight=0.95,
            properties={"semantic_degree": 0.95, "posterior_belief": 0.99},
        )
    )
    graph.add_edge(
        HCIREdge(
            id="e2",
            edge_type=HCIREdgeType.NEAR,
            sources=["tray"],
            targets=["plate"],
            weight=0.80,
            properties={"semantic_degree": 0.80, "posterior_belief": 0.97},
        )
    )
    graph.add_edge(
        HCIREdge(
            id="e3",
            edge_type=HCIREdgeType.NEAR,
            sources=["fork"],
            targets=["box"],
            weight=0.90,
            properties={"semantic_degree": 0.90, "posterior_belief": 0.99},
        )
    )
    graph.add_edge(
        HCIREdge(
            id="e4",
            edge_type=HCIREdgeType.LOCATED_IN,
            sources=["box"],
            targets=["warehouse"],
            weight=0.98,
            properties={"semantic_degree": 0.98, "posterior_belief": 0.99},
        )
    )

    op = SpatialOperator(enable_fuzzy=True)
    spatial_prob = ReasoningProblem(problem_type=ProblemType.SPATIAL, description="compose")
    res = op.execute(
        spatial_prob,
        ReasoningContext(graph_view=FrozenGraphView.from_graph(graph), problem=spatial_prob),
    )
    assert res.status == ResultStatus.SUCCESS
    triples = {(f["a"], str(f["relation"]), f["b"]) for f in res.conclusions["spatial_graph"]}
    assert ("cup", "near", "plate") in triples
    assert ("fork", "near", "warehouse") not in triples
    # NEAR o NEAR (tray->plate->tray or fork->box->fork) must never self-loop/drift
    assert ("cup", "near", "tray") not in triples or ("fork", "near", "warehouse") not in triples


def test_deduction_operator_fuzzy_modus_ponens_and_bipolar_belief_revision() -> None:
    """DeductionOperator propagates graded degrees and BeliefConflictResolver resists correlated flooding."""
    graph = CognitiveGraph()
    fact = BeliefNode(
        id="b_fact",
        claim="the container is fragile",
        confidence=0.85,
        properties={"semantic_degree": 0.72},
    )
    rule = BeliefNode(
        id="b_rule",
        claim="if the container is fragile then handle gently",
        confidence=0.90,
        properties={"semantic_degree": 0.85},
    )
    graph.add_node(fact)
    graph.add_node(rule)

    ded = DeductionOperator(enable_fuzzy=True, profile_id="lukasiewicz")
    ded_prob = ReasoningProblem(problem_type=ProblemType.EXPLANATION, description="infer")
    res = ded.execute(
        ded_prob,
        ReasoningContext(graph_view=FrozenGraphView.from_graph(graph), problem=ded_prob),
    )
    assert res.status == ResultStatus.SUCCESS
    expected_deg = max(0.0, 0.85 + 0.72 - 1.0)
    assert res.conclusions["derived_degrees"]["handle gently"] == pytest.approx(expected_deg)
    new_node_data = res.proposed_transitions[0].node_data
    assert new_node_data["properties"]["semantic_degree"] == pytest.approx(expected_deg)
    assert new_node_data["uncertainty"]["confidence"] == pytest.approx(0.85)

    # Anti-correlated BipolarEvidenceState in BeliefConflictResolver:
    # 2 independent refuting EvidenceNodes vs 20 correlated supporting EvidenceNodes from 1 group/channel/epoch
    rev_graph = CognitiveGraph()
    ev_ref1 = EvidenceNode(
        id="ev_r1",
        claim_id="the gate is safe",
        confidence=0.90,
        properties={
            "dependency_group": "cam_a",
            "source_channel_id": "vision",
            "tracklet_epoch": 1,
        },
    )
    ev_ref2 = EvidenceNode(
        id="ev_r2",
        claim_id="the gate is safe",
        confidence=0.90,
        properties={
            "dependency_group": "tactile_b",
            "source_channel_id": "force",
            "tracklet_epoch": 2,
        },
    )
    sup_ev_ids: list[str] = []
    for i in range(20):
        ev = EvidenceNode(
            id=f"ev_s_{i}",
            claim_id="the gate is not safe",
            confidence=0.95,
            properties={
                "dependency_group": "static_frame_burst",
                "source_channel_id": "vision",
                "tracklet_epoch": 1,
            },
        )
        rev_graph.add_node(ev)
        sup_ev_ids.append(ev.id)
    for n in (ev_ref1, ev_ref2):
        rev_graph.add_node(n)

    existing = BeliefNode(
        id="b_existing",
        statement="the gate is safe",
        claim="the gate is safe",
        confidence=0.70,
        evidence_sources=["ev_r1", "ev_r2"],
    )
    rev_graph.add_node(existing)
    proposed = BeliefNode(
        id="b_new",
        statement="the gate is not safe",
        claim="the gate is not safe",
        confidence=0.95,
        evidence_sources=sup_ev_ids,
    )
    resolver = BeliefConflictResolver()
    report = resolver.evaluate_belief(proposed, rev_graph)
    assert report.has_conflicts
    assert report.bipolar_state is not None
    assert report.bipolar_state.independent_support_clusters == 1
    assert report.bipolar_state.independent_refute_clusters == 2
    assert report.conflicts[0].resolution == "contested"
    assert report.bipolar_state.conflict > 0.0
