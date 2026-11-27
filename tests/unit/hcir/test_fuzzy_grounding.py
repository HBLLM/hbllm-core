"""Phase F1 tests: Fuzzy SGE Concepts, Adaptive Relations, Committer, and Experiment B vs A."""

from __future__ import annotations

import numpy as np
import pytest

from hbllm.hcir.grounding.concepts.concept_learner import ConceptLearner
from hbllm.hcir.grounding.concepts.model import GroundedConcept
from hbllm.hcir.grounding.concepts.store import WorkingConceptStore
from hbllm.hcir.grounding.contracts.enums import ConceptLifecycle, ModalitySource
from hbllm.hcir.grounding.contracts.quality import GroundingQuality
from hbllm.hcir.grounding.contracts.references import GroundedConceptRef
from hbllm.hcir.grounding.features.feature_extractor import PerceptualFeatureVector
from hbllm.hcir.grounding.relations.dyads import SpatialDyad
from hbllm.hcir.grounding.relations.relation_learner import RelationLearner
from hbllm.hcir.nodes import AdaptiveMembershipNode, GroundedConceptNode, HCIRNodeType
from hbllm.hcir.types import Scope
from hbllm.hcir.world.percept import EpisodeBoundary
from hbllm.hcir.world.planning.goal_scheduler import EpistemicGoalScheduler
from hbllm.hcir.world.state.context import CognitiveContext
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


def _make_fv(
    fid: str,
    embedding: tuple[float, ...],
    *,
    entity_id: str | None = None,
    raw_color: int = 2,
    labels: dict[str, float] | None = None,
) -> PerceptualFeatureVector:
    meta: dict[str, object] = {"raw_color": raw_color}
    if entity_id is not None:
        meta["entity_id"] = entity_id
    return PerceptualFeatureVector(
        feature_id=fid,
        observation_id=f"obs_{fid}",
        modality=ModalitySource.VISION,
        channel_id="cortical_perception",
        embedding=embedding,
        semantic_labels=labels or {},
        feature_metadata=meta,
    )


def test_fuzzy_concept_overlapping_membership_and_stability() -> None:
    """Entities hold graded membership across multiple overlapping concepts without erasure."""
    store = WorkingConceptStore()
    learner = ConceptLearner(store, enable_fuzzy=True)

    c_container = GroundedConcept(
        concept_ref=GroundedConceptRef(id="C-CONTAINER"),
        lifecycle_state=ConceptLifecycle.PROVISIONAL,
        grounding_quality=GroundingQuality.balanced(0.7),
    )
    c_cup = GroundedConcept(
        concept_ref=GroundedConceptRef(id="C-CUP"),
        lifecycle_state=ConceptLifecycle.PROVISIONAL,
        grounding_quality=GroundingQuality.balanced(0.7),
    )
    c_bottle = GroundedConcept(
        concept_ref=GroundedConceptRef(id="C-BOTTLE"),
        lifecycle_state=ConceptLifecycle.PROVISIONAL,
        grounding_quality=GroundingQuality.balanced(0.7),
    )

    for _ in range(4):
        c_container.add_prototype({"dim_0": 1.0, "dim_1": 1.0, "dim_2": 0.5})
        c_cup.add_prototype({"dim_0": 0.9, "dim_1": 0.95, "dim_2": 0.4})
        c_bottle.add_prototype({"dim_0": 0.3, "dim_1": 0.9, "dim_2": 1.8})

    store.register_concept(c_container)
    store.register_concept(c_cup)
    store.register_concept(c_bottle)

    fv_mug = _make_fv("mug_1", (0.95, 0.97, 0.45), entity_id="ent_mug")
    multi = learner.learn_fuzzy(fv_mug, exemplar_entity_id="ent_mug", spawn_alpha=0.45)

    assert multi.primary_assignment.is_new_concept is False
    assert multi.memberships["C-CONTAINER"] > 0.85
    assert multi.memberships["C-CUP"] > 0.85
    assert multi.memberships.get("C-BOTTLE", 0.0) < 0.25
    # FuzzyAssertions emitted without confidence field
    for fa in multi.assertions:
        assert "confidence" not in fa.model_dump()


def test_scale_invariant_fuzzy_relations_and_versioned_adaptation() -> None:
    """1x vs 10x scaled scenes preserve NEAR/TOUCHING degrees; adaptation creates v2 non-retroactively."""
    rl = RelationLearner(proximity_threshold=3.0, enable_fuzzy=True)

    dyad_1x = SpatialDyad(subject_id="e_a", object_id="e_b", distance=1.5, dx=1.5, dy=0.0)
    dyad_10x = SpatialDyad(subject_id="e_a", object_id="e_b", distance=15.0, dx=15.0, dy=0.0)

    rels_1x = {
        fa.semantics: fa
        for fa in rl.evaluate_fuzzy_relations(
            dyad_1x, subject_extent=0.5, object_extent=0.5, contextual_scope="manipulation"
        )
    }
    rels_10x = {
        fa.semantics: fa
        for fa in rl.evaluate_fuzzy_relations(
            dyad_10x, subject_extent=5.0, object_extent=5.0, contextual_scope="manipulation"
        )
    }

    assert rels_1x["relation:NEAR"].degree == pytest.approx(rels_10x["relation:NEAR"].degree)
    assert 0.5 < rels_1x["relation:NEAR"].degree < 1.0

    # Adapt NEAR manipulation model and verify v1 -> v2 ops
    next_m, ops = rl.adapt_relation_model(
        "rel_near",
        "manipulation",
        [2.5, 3.0, 3.2],
        learning_rate=0.5,
        scope=Scope(tenant_id="tenant_f1"),
    )
    assert next_m.version == 2
    assert len(ops) == 3
    rels_after = {
        fa.semantics: fa
        for fa in rl.evaluate_fuzzy_relations(
            SpatialDyad(subject_id="e_a", object_id="e_b", distance=2.5, dx=2.5, dy=0.0),
            subject_extent=0.5,
            object_extent=0.5,
            contextual_scope="manipulation",
        )
    }
    assert rels_after["relation:NEAR"].model_version == 2
    assert rels_after["relation:NEAR"].degree > 0.7


def test_committer_separates_semantic_degree_and_posterior_belief() -> None:
    """HCIRCommitter persists semantic_degree on HCIREdge distinct from posterior_belief."""
    sys = build_cognitive_system(
        CognitiveSystemConfig(
            world_id="f1_commit_test",
            memory_dir=None,
            tenant_id="tenant_f1",
            enable_grounding=True,
            enable_fuzzy_grounding=True,
        )
    )
    assert sys.bridge is not None
    rl = sys.bridge.relation_learner
    dyad = SpatialDyad(subject_id="e_cup", object_id="e_plate", distance=1.8, dx=1.8, dy=0.0)
    assertions = rl.evaluate_fuzzy_relations(dyad, subject_extent=0.5, object_extent=0.5)
    near_fa = next(fa for fa in assertions if fa.semantics == "relation:NEAR")

    txs = sys.bridge.committer.commit_fuzzy_assertions(
        [near_fa],
        posterior_by_assertion={near_fa.assertion_id: 0.99},
        scope=Scope(tenant_id="tenant_f1"),
    )
    assert txs and txs[0].is_committed

    edges = [e for e in sys.workspace.graph.edges.values() if e.id.startswith("edge_frel_")]
    assert len(edges) == 1
    props = edges[0].properties
    assert props["semantic_degree"] == pytest.approx(near_fa.degree)
    assert props["posterior_belief"] == pytest.approx(0.99)
    assert props["semantic_degree"] != props["posterior_belief"]


def test_experiment_b_vs_a_scale_and_appearance_shifted_grounding() -> None:
    """Controlled Experiment B vs A: Fuzzy SGE improves downstream hazard recognition under frozen alpha-cuts."""
    cfg_a = CognitiveSystemConfig(
        world_id="exp_a",
        memory_dir=None,
        tenant_id="tenant_exp",
        enable_grounding=True,
        enable_fuzzy_grounding=False,
        enable_fuzzy_reasoning=False,
        perception_mode="handcrafted",
    )
    cfg_b = CognitiveSystemConfig(
        world_id="exp_b",
        memory_dir=None,
        tenant_id="tenant_exp",
        enable_grounding=True,
        enable_fuzzy_grounding=True,
        enable_fuzzy_reasoning=False,
        perception_mode="handcrafted",
    )
    sys_a = build_cognitive_system(cfg_a)
    sys_b = build_cognitive_system(cfg_b)
    assert sys_a.bridge is not None and sys_b.bridge is not None

    # Train 3x3 compact hazard (color=5) and 3x9 elongated goal (color=3) in Episode 1
    grid_ep1 = np.zeros((16, 16), dtype=np.int32)
    grid_ep1[1, 1] = 2  # avatar
    grid_ep1[3:6, 3:6] = 5  # 3x3 compact hazard (center (4, 4))
    grid_ep1[10:13, 3:12] = 3  # 3x9 elongated goal (center (11, 7))
    ctx_ep1 = CognitiveContext(curr_grid=grid_ep1, bg_feature=0)
    ents_ep1 = [
        {"id": "e_haz1", "position": (4.0, 4.0), "color": 5, "size": 9, "bbox": (3, 3, 3, 3)},
        {"id": "e_goal1", "position": (11.0, 7.0), "color": 3, "size": 27, "bbox": (10, 3, 3, 9)},
    ]

    for sys in (sys_a, sys_b):
        sys.bridge.process_perception(ctx=ctx_ep1, extracted_entities=ents_ep1)
        sys.engine.symbolic_theory.induce_goal(3, evidence="terminal_reward")
        sys.bridge.sync_theory_roles(sys.engine.symbolic_theory, hazard_features={5})
        for c in sys.bridge.concept_store.get_all_concepts():
            if "role:hazard" in c.affordances or "role:goal" in c.affordances:
                c.update_grounding_quality(GroundingQuality.balanced(0.90))
        sys.bridge.process_perception(ctx=ctx_ep1, extracted_entities=ents_ep1)
        sys.bridge.on_boundary(EpisodeBoundary(same_level=False, same_context=True))

    # In Episode 2, present a scaled & recolored 32x32 scene (5x5 hazard color=8, 5x15 goal color=7)
    grid_ep2 = np.zeros((32, 32), dtype=np.int32)
    grid_ep2[1, 1] = 2
    grid_ep2[6:11, 6:11] = 8
    grid_ep2[15:20, 6:21] = 7
    ctx_ep2 = CognitiveContext(curr_grid=grid_ep2, bg_feature=0)
    ents_ep2 = [
        {"id": "e_haz2", "position": (8.0, 8.0), "color": 8, "size": 25, "bbox": (6, 6, 5, 5)},
        {"id": "e_goal2", "position": (17.0, 13.0), "color": 7, "size": 75, "bbox": (15, 6, 5, 15)},
    ]

    sig_a = sys_a.bridge.process_perception(ctx=ctx_ep2, extracted_entities=ents_ep2)
    sig_b = sys_b.bridge.process_perception(ctx=ctx_ep2, extracted_entities=ents_ep2)

    priors_a = sig_a["sge_role_priors"]
    priors_b = sig_b["sge_role_priors"]

    # Crisp Experiment A gets zero priors in handcrafted mode (and hard distance cutoff misses scaled features)
    assert priors_a == {}

    # Fuzzy Experiment B recovers graded hazard & goal priors across scale/color shift
    assert priors_b.get(8, {}).get("hazard", 0.0) >= 0.65
    assert priors_b.get(7, {}).get("goal", 0.0) >= 0.65

    # Verify frozen downstream crisp GoalScheduler alpha-cut (>= 0.65) detects hazard in B, not A
    sys_a.engine.working_memory.concept_role_priors = dict(priors_a)
    sys_b.engine.working_memory.concept_role_priors = dict(priors_b)
    haz_a = EpistemicGoalScheduler._high_confidence_prior_hazards(sys_a.engine, bg=0, av_feats={2})
    haz_b = EpistemicGoalScheduler._high_confidence_prior_hazards(sys_b.engine, bg=0, av_feats={2})
    assert 8 not in haz_a
    assert 8 in haz_b

    # Verify AdaptiveMembershipNode + GroundedConceptNode persisted in B's graph
    assert len(sys_b.workspace.graph.nodes_by_type(HCIRNodeType.GROUNDED_CONCEPT)) >= 1
    assert any(
        isinstance(n, AdaptiveMembershipNode)
        for n in sys_b.workspace.graph.nodes_by_type(HCIRNodeType.ADAPTIVE_MEMBERSHIP)
    )
    assert any(
        isinstance(n, GroundedConceptNode)
        for n in sys_b.workspace.graph.nodes_by_type(HCIRNodeType.GROUNDED_CONCEPT)
    )
    # Verify FuzzyAssertions emitted for relations in B
    assert len(sig_b["sge_fuzzy_relations"]) >= 1

    # Verify canonical CognitiveGraph read-path consumption when working store is cleared
    sys_b.bridge.concept_store._concepts.clear()
    sys_b.bridge.state.cached_entity_priors.clear()
    haz_fv_ep2 = next(
        fv
        for fv in sys_b.bridge.state.last_feature_vectors
        if fv.feature_metadata.get("raw_color") == 8
    )
    graph_priors = sys_b.bridge.concept_learner.infer_role_priors_fuzzy(
        haz_fv_ep2, graph=sys_b.workspace.graph
    )
    assert graph_priors.get("hazard", 0.0) >= 0.65
