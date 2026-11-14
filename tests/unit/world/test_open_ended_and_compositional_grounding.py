"""Unit and integration tests for G7-G15 live wiring, compositional concepts,

open-ended PCFG hypothesis synthesis, and independent external benchmarks.
"""

from __future__ import annotations

import numpy as np

from benchmarks.envs.external_benchmarks import (
    GymFrozenLakeBenchmarkEnv,
    MiniGridDoorKeyBenchmarkEnv,
    MiniGridEmptyBenchmarkEnv,
    MiniGridLavaGapBenchmarkEnv,
)
from benchmarks.envs.protocol import BenchmarkEnvironment
from benchmarks.suites.runner import run_closed_loop_episode
from hbllm.hcir.grounding.concepts.concept_learner import ConceptLearner
from hbllm.hcir.grounding.concepts.store import WorkingConceptStore
from hbllm.hcir.grounding.features.feature_extractor import (
    ModalitySource,
    PerceptualFeatureVector,
)
from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory
from hbllm.hcir.world.cortex_hypothesis import (
    HypothesisStatus,
    HypothesisType,
    InductiveHypothesisEngine,
    PopperianFalsifier,
    SynthesizedRuleAST,
    TransitionObservationContext,
)
from hbllm.hcir.world.cortical_grounding_bridge import CorticalGroundingBridge
from hbllm.hcir.world.library_learning import LibraryLearner
from hbllm.hcir.world.percept import EmbodiedObservation
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


def test_live_pipeline_executes_g8_g10_g11_g15_and_dynamic_bridge() -> None:
    """Verify G8 hints, G10 social cognition, G11 options, G15 provenance, and dynamic bridge run live."""
    system = build_cognitive_system(
        CognitiveSystemConfig(world_id="test_g7_g15_live", enable_logging=False)
    )
    engine = system.engine

    # Wrap engine.grounding_bridge *after* construction to prove CorticalPipeline resolves it dynamically.
    original_bridge = engine.grounding_bridge
    bridge_calls: list[str] = []

    class BridgeSpy:
        def __getattr__(self, name: str):
            attr = getattr(original_bridge, name)
            if callable(attr) and name.startswith("process_"):

                def _wrapped(*args, **kwargs):
                    bridge_calls.append(name)
                    return attr(*args, **kwargs)

                return _wrapped
            return attr

    engine.grounding_bridge = BridgeSpy()

    # 2-room bottleneck grid (7x7): walls=5, doorway at (3, 3), avatar=1, moving agent=9
    grid1 = np.zeros((7, 7), dtype=int)
    grid1[0, :] = 5
    grid1[-1, :] = 5
    grid1[:, 0] = 5
    grid1[:, -1] = 5
    grid1[:, 3] = 5
    grid1[3, 3] = 0  # single doorway bottleneck connecting room 1 and room 2
    grid1[3, 1] = 1  # avatar
    grid1[1, 5] = 9  # independent moving agent
    grid1[5, 5] = 8  # candidate goal

    engine.avatar_features = {1}
    engine.avatar_pos = (3, 1)
    engine.goals = [(5, 5)]
    engine.mobile_threat_features = {9}

    obs1 = EmbodiedObservation(
        lattice=grid1,
        available_actions=(1, 2, 3, 4),
        context_id="g7_g15_test",
        episode_id="ep1",
        metadata={"rule_hint": "Key 5 unlocks door 4"},
    )
    engine.step(obs1)

    grid2 = grid1.copy()
    grid2[1, 5] = 0
    grid2[2, 5] = 9  # moving agent steps downward
    obs2 = EmbodiedObservation(
        lattice=grid2,
        available_actions=(1, 2, 3, 4),
        context_id="g7_g15_test",
        episode_id="ep1",
        metadata={"rule_hint": "Key 5 unlocks door 4"},
    )
    engine.step(obs2)

    # G8: utterance rule hint compiled into symbolic_theory causal_predicates and cached in working memory
    assert "Key 5 unlocks door 4" in engine.working_memory.compiled_linguistic_hints
    assert len(engine.symbolic_theory.causal_predicates) > 0

    # G11: topological bottleneck & option induced live in pipeline
    assert len(engine.option_discovery.discovered_options) > 0
    assert (3, 3) in {
        opt.bottleneck_state for opt in engine.option_discovery.discovered_options.values()
    }

    # G10: social cognition tracked dynamic agent
    assert len(engine.social_cognition.tracked_agents) > 0

    # G15: goal provenance registered active goal
    assert len(engine.goal_provenance.nodes) > 0

    # Dynamic grounding_bridge lookup worked post-construction
    assert "process_perception" in bridge_calls


def test_compositional_concept_induction_and_zero_shot_transfer() -> None:
    """Verify ConceptLearner induces composite concepts and transfers relational priors zero-shot."""
    store = WorkingConceptStore()
    learner = ConceptLearner(concept_store=store)

    fv_cargo = PerceptualFeatureVector(
        feature_id="fv_20",
        observation_id="obs1",
        modality=ModalitySource.VISION,
        channel_id="grid",
        embedding=(0.15, 0.25, 0.05, 0.85),
        feature_metadata={"raw_color": 20, "entity_id": "ent_20"},
    )
    fv_receptacle = PerceptualFeatureVector(
        feature_id="fv_30",
        observation_id="obs1",
        modality=ModalitySource.VISION,
        channel_id="grid",
        embedding=(0.85, 0.75, 0.90, 0.15),
        feature_metadata={"raw_color": 30, "entity_id": "ent_30"},
    )

    c_cargo = learner.bind_role_pure(
        None,
        fv_cargo,
        "cargo",
        exemplar_entity_id="ent_20",
    )
    c_receptacle = learner.bind_role_pure(
        None,
        fv_receptacle,
        "receptacle",
        exemplar_entity_id="ent_30",
    )
    assert c_cargo is not None and c_receptacle is not None

    composite = learner.induce_composite_concept(
        parts={
            "cargo": c_cargo.concept_ref.id,
            "receptacle": c_receptacle.concept_ref.id,
        },
        relation_name="docks_into",
        composite_role="push_docking_system",
    )
    assert composite is not None
    assert composite.is_composite is True
    assert len(composite.parts) == 2
    assert len(composite.higher_order_relations) == 1
    assert composite.relational_graph is not None

    # Single-entity nearest/role inference must still resolve pure concept cleanly
    single_priors = learner.infer_role_priors(fv_cargo)
    assert "cargo" in single_priors

    # Zero-shot transfer to novel unseen feature IDs 71 (cargo-like) and 82 (receptacle-like)
    fv_novel_cargo = PerceptualFeatureVector(
        feature_id="fv_71",
        observation_id="obs2",
        modality=ModalitySource.VISION,
        channel_id="grid",
        embedding=(0.16, 0.24, 0.06, 0.84),
        feature_metadata={"raw_color": 71, "entity_id": "ent_71"},
    )
    fv_novel_receptacle = PerceptualFeatureVector(
        feature_id="fv_82",
        observation_id="obs2",
        modality=ModalitySource.VISION,
        channel_id="grid",
        embedding=(0.84, 0.76, 0.89, 0.16),
        feature_metadata={"raw_color": 82, "entity_id": "ent_82"},
    )
    comp_matches = learner.infer_composite_role_priors(
        {71: fv_novel_cargo, 82: fv_novel_receptacle}
    )
    assert len(comp_matches) >= 1
    bindings = comp_matches[0]["part_feature_bindings"]
    assert bindings.get("cargo") == 71
    assert bindings.get("receptacle") == 82

    # CorticalGroundingBridge.sync_theory_roles induces composite concepts for paired roles
    bridge = CorticalGroundingBridge()
    bridge.state.last_feature_vectors = [
        PerceptualFeatureVector(
            feature_id="fv_41",
            observation_id="obs3",
            modality=ModalitySource.VISION,
            channel_id="grid",
            embedding=(0.10, 0.20, 0.30, 0.40),
            feature_metadata={"raw_color": 41, "entity_id": "e41"},
        ),
        PerceptualFeatureVector(
            feature_id="fv_42",
            observation_id="obs3",
            modality=ModalitySource.VISION,
            channel_id="grid",
            embedding=(0.50, 0.60, 0.70, 0.80),
            feature_metadata={"raw_color": 42, "entity_id": "e42"},
        ),
        PerceptualFeatureVector(
            feature_id="fv_43",
            observation_id="obs3",
            modality=ModalitySource.VISION,
            channel_id="grid",
            embedding=(0.20, 0.80, 0.10, 0.90),
            feature_metadata={"raw_color": 43, "entity_id": "e43"},
        ),
        PerceptualFeatureVector(
            feature_id="fv_44",
            observation_id="obs3",
            modality=ModalitySource.VISION,
            channel_id="grid",
            embedding=(0.90, 0.10, 0.80, 0.20),
            feature_metadata={"raw_color": 44, "entity_id": "e44"},
        ),
    ]
    theory = HCIRSymbolicWorldTheory()
    theory.induce_cargo(41, evidence="test_cargo")
    theory.induce_receptacle(42, evidence="test_receptacle")
    theory.induce_actuator(43, evidence="test_actuator")
    theory.induce_barrier(44, evidence="test_barrier")
    bound_count = bridge.sync_theory_roles(theory)
    assert bound_count == 4
    composites = [c for c in bridge.concept_store.get_all_concepts() if c.is_composite]
    assert len(composites) >= 2


def test_open_ended_pcfg_hypothesis_synthesis_falsification_and_proposals() -> None:
    """Verify open-ended AST synthesis, PCFG weight updates, Popperian falsification, and proposals."""
    engine = InductiveHypothesisEngine(min_support_to_confirm=2)

    prev = np.zeros((5, 5), dtype=int)
    prev[2, 2] = 1  # avatar
    prev[2, 3] = 4  # trigger cell
    prev[4, 4] = 7  # remote cell before transform

    curr = prev.copy()
    curr[2, 2] = 0
    curr[2, 3] = 1
    curr[4, 4] = 8  # remote cell transformed 7 -> 8

    emitted1 = engine.observe_transition(
        prev_grid=prev,
        action=1,
        curr_grid=curr,
        prev_avatar_pos=(2, 2),
        curr_avatar_pos=(2, 3),
        is_dead=False,
        is_won=False,
    )
    assert emitted1

    prog_hyps = [
        h for h in engine.hypotheses.values() if h.rule_type == HypothesisType.COMPOSITIONAL_PROGRAM
    ]
    assert len(prog_hyps) >= 1
    target_hyp = prog_hyps[0]
    ast_obj = target_hyp.metadata.get("ast")
    assert isinstance(ast_obj, SynthesizedRuleAST)

    weight_before = engine.pcfg_prior.production_counts.get(ast_obj.productions_used[0], 1.0)

    # Second supporting observation confirms and increases PCFG weight
    engine.observe_transition(
        prev_grid=prev,
        action=1,
        curr_grid=curr,
        prev_avatar_pos=(2, 2),
        curr_avatar_pos=(2, 3),
        is_dead=False,
        is_won=False,
    )
    assert target_hyp.status == HypothesisStatus.CONFIRMED
    weight_after_confirm = engine.pcfg_prior.production_counts.get(ast_obj.productions_used[0], 1.0)
    assert weight_after_confirm > weight_before

    # Export confirmed compositional AST to world theory
    theory = HCIRSymbolicWorldTheory()
    engine.export_to_world_theory(theory)
    assert len(theory.causal_predicates) > 0

    # Counterexample observation: precondition holds, expected transformation does NOT occur
    curr_fail = prev.copy()
    curr_fail[2, 2] = 0
    curr_fail[2, 3] = 1
    ctx_fail = TransitionObservationContext(
        prev_grid=prev,
        curr_grid=curr_fail,
        action=1,
        prev_avatar_pos=(2, 2),
        curr_avatar_pos=(2, 3),
        is_dead=False,
        is_won=False,
    )
    falsified = PopperianFalsifier.falsify(
        ctx_fail,
        engine.hypotheses,
        pcfg_prior=engine.pcfg_prior,
    )
    assert len(falsified) >= 1
    weight_after_falsify = engine.pcfg_prior.production_counts.get(ast_obj.productions_used[0], 1.0)
    assert weight_after_falsify < weight_after_confirm

    # LearnedProposalDistribution proposes ranked open-ended intents combining macros + hypotheses
    learner = LibraryLearner()
    proposals = learner.sample_open_ended_proposals(
        context_key=(1, 2),
        hypotheses=engine.hypotheses,
        top_k=5,
    )
    assert len(proposals) > 0
    assert all(isinstance(name, str) and isinstance(score, float) for name, score in proposals)


def test_independent_external_benchmarks_frozenlake_and_minigrid() -> None:
    """Verify third-party Gymnasium FrozenLake & MiniGrid environments work end-to-end with zero leakage."""
    for env_cls, seed_base, budget in (
        (GymFrozenLakeBenchmarkEnv, 500, 60),
        (MiniGridEmptyBenchmarkEnv, 510, 60),
        (MiniGridLavaGapBenchmarkEnv, 520, 60),
        (MiniGridDoorKeyBenchmarkEnv, 530, 80),
    ):
        # Default domain-consistent scrambled ontology keeps non-standard IDs (>= 10)
        env = env_cls(max_steps=budget)
        assert isinstance(env, BenchmarkEnvironment)

        obs_default = env.reset(seed=seed_base)
        assert obs_default.lattice is not None and obs_default.lattice.ndim == 2
        assert len(obs_default.entities) == 0
        non_zero_feats = {int(v) for v in np.unique(obs_default.lattice) if int(v) != 0}
        assert non_zero_feats and all(f >= 10 for f in non_zero_feats)

        # Opt-in per-seed ontology scrambling varies feature IDs across seeds
        env_scrambled = env_cls(max_steps=budget, scramble_per_seed=True)
        obs_a = env_scrambled.reset(seed=seed_base)
        obs_b = env_scrambled.reset(seed=seed_base + 1)
        assert obs_a.lattice is not None and obs_a.lattice.ndim == 2
        assert obs_b.lattice is not None and obs_b.lattice.ndim == 2
        assert len(obs_a.entities) == 0
        assert len(obs_b.entities) == 0
        assert set(np.unique(obs_a.lattice).tolist()) != set(np.unique(obs_b.lattice).tolist())

        # Oracle solves 100% across 3 independent seeds
        for s in range(seed_base, seed_base + 3):
            succ_o, steps_o, _, errs_o = run_closed_loop_episode(
                env=env,
                seed=s,
                max_budget=budget,
                baseline="oracle",
            )
            assert errs_o == []
            assert succ_o is True, f"{env_cls.__name__} oracle failed on seed={s} (steps={steps_o})"

        # HBLLM cognitive system executes closed-loop steps without errors
        succ_s, steps_s, _, errs_s = run_closed_loop_episode(
            env=env,
            seed=seed_base,
            max_budget=25,
            baseline="system",
        )
        assert errs_s == []
        assert steps_s >= 1
        assert isinstance(succ_s, bool)
