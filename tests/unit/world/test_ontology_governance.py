from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from benchmarks.envs.grid_nav import GridNavEnvironment
from hbllm.hcir.graph import ConceptNode
from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory, OntologyEvidenceRank
from hbllm.hcir.world.percept import TerminalSignal
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


def test_sets_are_readonly_frozenset() -> None:
    """Verify ontology sets are exposed as read-only frozenset preventing direct mutation."""
    theory = HCIRSymbolicWorldTheory()
    assert isinstance(theory.goal_features, frozenset)
    assert isinstance(theory.candidate_goal_features, frozenset)
    assert isinstance(theory.barrier_features, frozenset)
    assert isinstance(theory.walkable_features, frozenset)
    assert isinstance(theory.cargo_features, frozenset)
    assert isinstance(theory.receptacle_features, frozenset)
    assert isinstance(theory.actuator_features, frozenset)
    assert isinstance(theory.avatar_features, frozenset)

    with pytest.raises(AttributeError):
        theory.goal_features.add(1)  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        theory.barrier_features.add(2)  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        theory.walkable_features.add(0)  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        theory.actuator_features.add(9)  # type: ignore[attr-defined]


def test_induce_goal_requires_terminal_reward() -> None:
    """Verify induce_goal requires terminal reward evidence and registers weak evidence as candidate only."""
    theory = HCIRSymbolicWorldTheory()

    # Weak evidence registers as candidate goal only, not confirmed goal
    res_weak = theory.induce_goal(2, evidence="heuristic")
    assert res_weak is False
    assert 2 not in theory.goal_features
    assert 2 in theory.candidate_goal_features

    # Depletion hypothesis without reward evidence also registers as candidate
    theory.induce_depletion_goal(3, evidence="depletion_guess")
    assert 3 not in theory.goal_features
    assert 3 in theory.candidate_goal_features

    # Genuine terminal reward evidence confirms goal
    res_term = theory.induce_goal(8, evidence=TerminalSignal.SUCCESS)
    assert res_term is True
    assert 8 in theory.goal_features
    assert 8 in theory.walkable_features


def test_induce_goal_cannot_delete_barrier_without_outranking_evidence() -> None:
    """Verify induce_goal must not delete an existing barrier unless evidence strictly outranks it."""
    theory = HCIRSymbolicWorldTheory()

    # Collision barrier has rank 80
    theory.induce_barrier(5, evidence="collision")
    assert 5 in theory.barrier_features

    # Weak/heuristic evidence (rank 10-20) cannot delete collision barrier
    res_fail = theory.induce_goal(5, evidence="heuristic")
    assert res_fail is False
    assert 5 in theory.barrier_features
    assert 5 not in theory.goal_features

    # Terminal reward (rank 100) outranks collision barrier (rank 80) and deletes it
    res_success = theory.induce_goal(5, evidence=TerminalSignal.SUCCESS)
    assert res_success is True
    assert 5 not in theory.barrier_features
    assert 5 in theory.goal_features


def test_induce_barrier_cannot_overwrite_confirmed_goal() -> None:
    """Verify induce_barrier cannot overwrite a confirmed terminal goal."""
    theory = HCIRSymbolicWorldTheory()
    theory.induce_goal(8, evidence="terminal_reward")
    assert 8 in theory.goal_features

    res_bar = theory.induce_barrier(8, evidence="collision")
    assert res_bar is False
    assert 8 in theory.goal_features
    assert 8 not in theory.barrier_features


def test_retract_methods() -> None:
    """Verify retract_* methods cleanly remove ontology features and associated predicates."""
    theory = HCIRSymbolicWorldTheory()
    theory.induce_goal(8, evidence="terminal_reward")
    theory.induce_barrier(2, evidence="collision")
    theory.induce_walkable(0, evidence="traversal")

    theory.retract_goal(8)
    assert 8 not in theory.goal_features
    assert 8 not in theory.candidate_goal_features

    theory.retract_barrier(2)
    assert 2 not in theory.barrier_features

    theory.retract_walkable(0)
    assert 0 not in theory.walkable_features


def test_hydration_requires_terminal_reward_for_goals() -> None:
    """Verify hydrate_from_nodes only loads goal roles whose concept node carries terminal reward."""
    theory = HCIRSymbolicWorldTheory()
    nodes = [
        ConceptNode(
            id="ontology_goal_2",
            label="goal_2",
            definition="",
            domain="spatial_ontology",
            properties={"role": "goal", "feature_id": 2, "provenance": "heuristic"},
        ),
        ConceptNode(
            id="ontology_goal_8",
            label="goal_8",
            definition="",
            domain="spatial_ontology",
            properties={
                "role": "goal",
                "feature_id": 8,
                "has_terminal_reward": True,
                "evidence_rank": OntologyEvidenceRank.TERMINAL_REWARD,
            },
        ),
        ConceptNode(
            id="ontology_barrier_5",
            label="barrier_5",
            definition="",
            domain="spatial_ontology",
            properties={"role": "barrier", "feature_id": 5},
        ),
    ]

    theory.hydrate_from_nodes(nodes)
    assert 2 not in theory.goal_features
    assert 2 in theory.candidate_goal_features
    assert 8 in theory.goal_features
    assert 5 in theory.barrier_features


def test_scenario_b_no_false_goal_induction() -> None:
    """Verify Scenario B does not falsely induce door feature 2 as a goal."""
    with tempfile.TemporaryDirectory() as d:
        system = build_cognitive_system(
            CognitiveSystemConfig(world_id="diag_gov", memory_dir=d, enable_logging=False)
        )
        engine = system.engine
        env = GridNavEnvironment()

        def _run_episode(budget: int = 100) -> None:
            obs = env.reset(seed=100010)
            for _ in range(budget):
                a, _ = engine.step(obs)
                obs, r, done, info = env.step(a)
                if done:
                    engine.step(obs)
                    break

        # Episode A: self fresh
        _run_episode()
        assert engine.symbolic_theory.goal_features == frozenset()

        # Episode B: self same layout (previously falsely induced door 2 as a goal)
        _run_episode()
        assert engine.symbolic_theory.goal_features == frozenset(), (
            f"Expected empty goal_features after episode B, got {engine.symbolic_theory.goal_features}"
        )


def test_codebase_ontology_mutation_invariants() -> None:
    """Verify that no file outside cortex_causal.py mutates goal/barrier/walkable sets directly."""
    core_dir = Path(__file__).resolve().parents[3]
    cmd = [
        "rg",
        "-n",
        r"goal_features\.add|barrier_features\.add|walkable_features\.add",
        "hbllm/",
    ]
    proc = subprocess.run(cmd, cwd=str(core_dir), capture_output=True, text=True)
    assert proc.returncode == 0
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    for ln in lines:
        assert "cortex_causal.py" in ln, f"Forbidden direct mutation outside cortex_causal.py: {ln}"
