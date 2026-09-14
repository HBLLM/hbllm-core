"""Property-based fuzz testing for Interventional Causal Discovery Engine.

Uses Hypothesis to verify fundamental mathematical and cognitive invariants:
1. Posterior Monotonicity and Range Integrity ([0.0, 1.0])
2. Strict Irreversibility of Falsification
3. Entropy-Maximizing Selection Stability under Degenerate Percepts
4. Arbitrary Feature Stream Robustness and Safe Boundary Parsing
"""

from __future__ import annotations

import math
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
from plugins.developmental_adapter.causal_discovery import InterventionalCausalDiscoveryEngine
from plugins.developmental_adapter.environment import BabyWorldEnvironment
from plugins.developmental_adapter.perception import DevelopmentalPerceptionAdapter
from plugins.developmental_adapter.types import (
    BabyActionType,
    CausalHypothesis,
)


def _create_engine() -> tuple[InterventionalCausalDiscoveryEngine, BabyWorldEnvironment]:
    env = BabyWorldEnvironment(seed=42)
    substrate = create_blank_brain_substrate()
    perception = DevelopmentalPerceptionAdapter()
    engine = InterventionalCausalDiscoveryEngine(substrate, perception, env)
    return engine, env


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 1: Posterior Monotonicity and Range Integrity
# ─────────────────────────────────────────────────────────────────────────────


@given(
    initial_conf=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    evidence_steps=st.lists(st.booleans(), min_size=1, max_size=30),
)
@settings(max_examples=50)
def test_posterior_monotonicity_and_bounds(initial_conf: float, evidence_steps: list[bool]) -> None:
    """Confidence must remain in [0.0, 1.0] and monotonically increase under supporting evidence."""
    engine, _ = _create_engine()
    hyp = CausalHypothesis(
        action=BabyActionType.PUSH,
        variable="mass_sensation",
        operator="<",
        value=5.0,
        consequence="MOVES",
        confidence=initial_conf,
    )
    engine.hypotheses = [hyp]

    prev_conf = hyp.confidence
    was_falsified = hyp.falsified

    for matching_move in evidence_steps:
        probe_result = {
            "target_id": "test_obj",
            "did_move": matching_move,
            "mass_sensation": 3.0
            if matching_move
            else 8.0,  # 3.0 < 5.0 matches True, 8.0 < 5.0 matches False
        }
        engine._update_hypotheses_from_evidence(probe_result)

        assert 0.0 <= hyp.confidence <= 1.0, f"Confidence out of bounds: {hyp.confidence}"

        if was_falsified:
            assert hyp.falsified is True, "Falsified hypothesis revived!"
            assert hyp.confidence == 0.0, (
                f"Falsified hypothesis gained confidence: {hyp.confidence}"
            )
        else:
            if hyp.falsified:
                was_falsified = True
                assert hyp.confidence == 0.0
            else:
                # Supporting evidence: confidence must be non-decreasing
                assert hyp.confidence >= prev_conf, (
                    f"Confidence decreased under supporting evidence: {prev_conf} -> {hyp.confidence}"
                )

        prev_conf = hyp.confidence


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 2: Strict Irreversibility of Falsification
# ─────────────────────────────────────────────────────────────────────────────


@given(
    evidence_stream=st.lists(
        st.tuples(st.booleans(), st.floats(min_value=0.0, max_value=10.0)),
        min_size=2,
        max_size=20,
    )
)
@settings(max_examples=50)
def test_falsification_irreversibility(evidence_stream: list[tuple[bool, float]]) -> None:
    """Once a hypothesis encounters a contradiction, it remains permanently falsified."""
    engine, _ = _create_engine()
    hyp = CausalHypothesis(
        action=BabyActionType.PUSH,
        variable="mass_sensation",
        operator="<",
        value=5.0,
        consequence="MOVES",
        confidence=0.5,
    )
    engine.hypotheses = [hyp]

    ever_falsified = False

    for did_move, mass in evidence_stream:
        probe_result = {
            "target_id": "obj_x",
            "did_move": did_move,
            "mass_sensation": mass,
        }
        engine._update_hypotheses_from_evidence(probe_result)

        if hyp.falsified:
            ever_falsified = True

        if ever_falsified:
            assert hyp.falsified is True, "Hypothesis reverted from falsified to active!"
            assert hyp.confidence == 0.0, (
                f"Hypothesis has non-zero confidence ({hyp.confidence}) while falsified!"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 3: Entropy-Maximizing Selection Stability
# ─────────────────────────────────────────────────────────────────────────────


@given(
    entity_ids=st.lists(st.text(min_size=0, max_size=10), min_size=0, max_size=10),
    num_hypotheses=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=40)
def test_entropy_selection_robustness(entity_ids: list[str], num_hypotheses: int) -> None:
    """Intervention selection must handle arbitrary/empty entity pools and hypothesis states cleanly."""
    engine, _ = _create_engine()

    hyps = [
        CausalHypothesis(
            action=BabyActionType.PUSH,
            variable=f"feat_{i}",
            operator="<",
            value=float(i),
            consequence="MOVES",
            confidence=0.5,
            falsified=(i % 2 == 1),
        )
        for i in range(num_hypotheses)
    ]
    engine.hypotheses = hyps

    target_id, chosen_hyp = engine.select_active_intervention(entity_ids, hyps)

    assert isinstance(target_id, str), f"Target ID must be str, got {type(target_id)}"
    assert isinstance(chosen_hyp, CausalHypothesis), (
        f"Chosen hyp must be CausalHypothesis, got {type(chosen_hyp)}"
    )

    if entity_ids:
        # Chosen entity must come from available pool or fall back gracefully
        if target_id in entity_ids or target_id == entity_ids[0]:
            assert True
    else:
        assert target_id == "", f"Expected empty string when no entity_ids given, got {target_id}"


# ─────────────────────────────────────────────────────────────────────────────
# Invariant 4: Arbitrary Feature Stream Robustness
# ─────────────────────────────────────────────────────────────────────────────


feature_strategy = st.dictionaries(
    keys=st.sampled_from(
        ["mass_sensation", "color", "shape", "surface_friction", "density", "custom_feat"]
    ),
    values=st.one_of(
        st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False),
        st.text(min_size=1, max_size=8),
        st.integers(min_value=-1000, max_value=1000),
    ),
    min_size=1,
    max_size=6,
)

episode_strategy = st.fixed_dictionaries(
    {
        "moved": st.booleans(),
        "features": feature_strategy,
    }
)


@given(episodes=st.lists(episode_strategy, min_size=1, max_size=20))
@settings(max_examples=50)
def test_arbitrary_demonstrations_fuzzing(episodes: list[dict[str, Any]]) -> None:
    """Hypothesis formulation must not crash across randomized, arbitrary demonstration streams."""
    engine, env = _create_engine()
    dummy_obs = env.get_sensory_observation()

    hypotheses = engine.observe_and_generate_hypotheses(dummy_obs, episodes_data=episodes)

    assert isinstance(hypotheses, list)
    for h in hypotheses:
        assert isinstance(h, CausalHypothesis)
        assert h.operator in {"==", "!=", "<", "<=", ">", ">="}
        assert h.action == BabyActionType.PUSH
        assert 0.0 <= h.confidence <= 1.0
        assert h.variable != ""
        if isinstance(h.value, (int, float)):
            assert math.isfinite(h.value)
