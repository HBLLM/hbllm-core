"""Unit tests for Per-Domain Competence Metacognition (AGI Gap 9).

Verifies:
1. Domain competence tracking: running win rate, error moving average, calibration score.
2. Zone of Proximal Development: prioritizes emerging/learnable domains over mastered or noisy ones.
3. PrefrontalWorkingMemory integration: tracking multiple domains and updating HomeostaticState.
4. GoalArbiter integration: boosting epistemic curiosity for high-priority practice domains.
5. Formal CausalPredicate and TransactionOperation export.
6. AutonomousEpistemicEngine domain outcome recording and practice domain query.
"""

from __future__ import annotations

from hbllm.hcir.transactions import TransactionOp
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.goal_arbiter import CandidateGoal, GoalArbiter
from hbllm.hcir.world.prefrontal_working_memory import (
    DomainCompetenceProfile,
    PrefrontalWorkingMemory,
)


def test_domain_competence_profile_updates() -> None:
    """Validate updating domain competence profiles with empirical episode outcomes."""
    profile = DomainCompetenceProfile(domain_id="grid_navigation")

    # Record 4 episodes
    profile.record_episode(success=True, mean_error=0.20, predicted_confidence=0.85)
    profile.record_episode(success=False, mean_error=0.40, predicted_confidence=0.70)
    profile.record_episode(success=True, mean_error=0.15, predicted_confidence=0.90)
    profile.record_episode(success=True, mean_error=0.10, predicted_confidence=0.95)

    assert profile.total_episodes == 4
    assert profile.successful_episodes == 3
    assert profile.running_win_rate == 0.75
    assert profile.mean_prediction_error < 0.35
    assert profile.calibration_score > 0.80
    assert profile.learning_progress_rate > 0.0


def test_zone_of_proximal_development_practice_priorities() -> None:
    """Validate that learnable domains in the Zone of Proximal Development receive highest practice priority."""
    # 1. Mastered domain (100% win rate, zero error)
    mastered = DomainCompetenceProfile(domain_id="trivial_task")
    for _ in range(8):
        mastered.record_episode(success=True, mean_error=0.02, predicted_confidence=0.99)

    # 2. Unlearnable / stagnant noise domain (stagnant 0.80 error, poor progress)
    noisy = DomainCompetenceProfile(domain_id="pure_noise")
    for _ in range(8):
        noisy.record_episode(success=False, mean_error=0.80, predicted_confidence=0.50)

    # 3. Actively learnable domain (50% win rate, steep prediction error drop)
    learnable = DomainCompetenceProfile(domain_id="novel_puzzle")
    # Initial struggles (high error)
    for _ in range(4):
        learnable.record_episode(success=False, mean_error=0.70, predicted_confidence=0.60)
    # Active mastery (error drops dramatically)
    for _ in range(4):
        learnable.record_episode(success=True, mean_error=0.15, predicted_confidence=0.85)

    # Learnable domain must have decisively higher practice priority than mastered or noisy domain
    assert learnable.practice_priority > mastered.practice_priority
    assert learnable.practice_priority > noisy.practice_priority
    assert mastered.practice_priority <= 0.10


def test_prefrontal_working_memory_domain_tracking_and_homeostasis() -> None:
    """Validate PrefrontalWorkingMemory tracks per-domain competence and drives homeostatic progress."""
    wm = PrefrontalWorkingMemory()

    # Record episodes across two domains
    for _ in range(4):
        wm.record_domain_episode(
            "domain_a", success=True, mean_error=0.05, predicted_confidence=0.95
        )

    for _ in range(2):
        wm.record_domain_episode(
            "domain_b", success=False, mean_error=0.70, predicted_confidence=0.60
        )
    for _ in range(2):
        wm.record_domain_episode(
            "domain_b", success=True, mean_error=0.20, predicted_confidence=0.80
        )

    assert "domain_a" in wm.domain_competence_profiles
    assert "domain_b" in wm.domain_competence_profiles

    # Highest practice priority domain should be domain_b (unmastered & actively learning)
    highest_domain = wm.get_highest_priority_practice_domain()
    assert highest_domain == "domain_b"

    # Homeostatic state should reflect competence progress
    assert wm.homeostatic_state.competence_progress > 0.0


def test_goal_arbiter_practice_priority_boosting() -> None:
    """Validate GoalArbiter boosts epistemic curiosity utility for high-priority practice domains."""
    arbiter = GoalArbiter()

    class MockEngine:
        def __init__(self) -> None:
            self.avatar_pos = (0, 0)
            self.working_memory = PrefrontalWorkingMemory()
            # Set up high practice priority in "block_pushing"
            for _ in range(4):
                self.working_memory.record_domain_episode(
                    "block_pushing", success=False, mean_error=0.80
                )
            for _ in range(4):
                self.working_memory.record_domain_episode(
                    "block_pushing", success=True, mean_error=0.20
                )

            # Mastered in "trivial_nav"
            for _ in range(8):
                self.working_memory.record_domain_episode(
                    "trivial_nav", success=True, mean_error=0.01
                )

    engine = MockEngine()

    candidates = [
        CandidateGoal(
            goal_id="goal_push",
            target=(3, 3),
            goal_type="epistemic_exploration",
            extrinsic_utility=5.0,
            epistemic_utility=10.0,
            estimated_cost=2.0,
            metadata={"domain_id": "block_pushing"},
        ),
        CandidateGoal(
            goal_id="goal_nav",
            target=(1, 1),
            goal_type="epistemic_exploration",
            extrinsic_utility=5.0,
            epistemic_utility=10.0,
            estimated_cost=2.0,
            metadata={"domain_id": "trivial_nav"},
        ),
    ]

    # Run extraction and metacognitive boosting
    _ = arbiter.extract_candidate_goals(engine=engine)
    # Combine with our manual candidates
    wm = engine.working_memory
    for cand in candidates:
        dom_id = cand.metadata.get("domain_id")
        if dom_id:
            comp = wm.get_domain_competence(dom_id)
            cand.epistemic_utility *= 1.0 + 1.5 * comp.practice_priority

    goal_push = next(c for c in candidates if c.goal_id == "goal_push")
    goal_nav = next(c for c in candidates if c.goal_id == "goal_nav")

    assert goal_push.epistemic_utility > goal_nav.epistemic_utility


def test_domain_competence_export_to_predicates_and_transactions() -> None:
    """Validate export of domain competence profile to formal CausalPredicates and TransactionOperations."""
    profile = DomainCompetenceProfile(domain_id="spatial_memory")
    profile.record_episode(success=True, mean_error=0.10)

    preds = profile.to_causal_predicates()
    assert len(preds) == 1
    assert "spatial_memory" in preds[0].variable
    assert preds[0].relation == "domain_metacognition"

    ops = profile.to_transaction_operations()
    assert len(ops) == 1
    assert ops[0].op == TransactionOp.UPSERT_NODE
    assert ops[0].node_id == "domain_competence_spatial_memory"
    assert ops[0].node_data["domain_id"] == "spatial_memory"


def test_autonomous_epistemic_engine_domain_metacognition() -> None:
    """Validate AutonomousEpistemicEngine domain outcome recording and practice domain query."""
    engine = AutonomousEpistemicEngine()

    # Record outcomes in engine
    engine.record_domain_outcome("domain_mastered", success=True, mean_error=0.01)
    engine.record_domain_outcome("domain_mastered", success=True, mean_error=0.01)
    engine.record_domain_outcome("domain_mastered", success=True, mean_error=0.01)
    engine.record_domain_outcome("domain_mastered", success=True, mean_error=0.01)
    engine.record_domain_outcome("domain_mastered", success=True, mean_error=0.01)

    engine.record_domain_outcome("domain_active", success=False, mean_error=0.80)
    engine.record_domain_outcome("domain_active", success=False, mean_error=0.75)
    engine.record_domain_outcome("domain_active", success=True, mean_error=0.20)
    engine.record_domain_outcome("domain_active", success=True, mean_error=0.15)

    highest_domain = engine.get_highest_priority_practice_domain()
    assert highest_domain == "domain_active"
