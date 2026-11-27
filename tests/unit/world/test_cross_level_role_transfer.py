"""Cross-level semantic role transfer — unit tests.

Validates Fix 3 (Feature-to-Role Abstraction): when feature IDs change between
levels, the engine should transfer learned semantic roles (goal, barrier, key,
door, walkable) and conditionally gated affordances (key→door) to the new
feature IDs via role-level templates.

Invariants under test:
  - ``snapshot_conditional_affordance_templates`` captures feature→role → template
  - ``clear_feature_bindings`` clears feature→role but preserves role priors + templates
  - ``bind_feature_to_role`` lazily re-instantiates conditional affordance templates
  - Full episodic boundary round-trip: episode 1 → boundary → episode 2 transfers

Architecture note: This is a domain-agnostic role abstraction mechanism.
The "key", "door", "goal", "barrier" roles are universal semantic categories,
NOT task-specific solvers.
"""

from __future__ import annotations

from hbllm.hcir.world.cortex_causal import (
    CausalPredicate,
    HCIRSymbolicWorldTheory,
)

# ── Unit tests for HCIRSymbolicWorldTheory role abstraction API ──────────


class TestRoleAbstractionAPI:
    """Isolated tests for the semantic role abstraction layer on HCIRSymbolicWorldTheory."""

    def _make_theory(self) -> HCIRSymbolicWorldTheory:
        return HCIRSymbolicWorldTheory()

    def test_bind_and_get_role(self) -> None:
        """Basic bind + get roundtrip."""
        theory = self._make_theory()
        theory.bind_feature_to_role(3, "key")
        assert theory.get_role(3) == "key"
        assert 3 in theory.get_features_for_role("key")

    def test_rebind_replaces_old_role(self) -> None:
        """Rebinding a feature to a different role removes the old binding."""
        theory = self._make_theory()
        theory.bind_feature_to_role(3, "walkable")
        theory.bind_feature_to_role(3, "goal")
        assert theory.get_role(3) == "goal"
        assert 3 not in theory.get_features_for_role("walkable")
        assert 3 in theory.get_features_for_role("goal")

    def test_role_priors_accumulate(self) -> None:
        """Role priors count each bind (even if later overridden)."""
        theory = self._make_theory()
        theory.bind_feature_to_role(3, "goal")
        theory.bind_feature_to_role(8, "goal")
        priors = theory.role_priors
        assert priors.get("goal", 0) >= 2

    def test_clear_feature_bindings_preserves_priors_and_templates(self) -> None:
        """clear_feature_bindings clears feature→role but keeps priors + templates."""
        theory = self._make_theory()
        theory.bind_feature_to_role(3, "key")
        theory.bind_feature_to_role(2, "door")
        theory.induce_goal(8, evidence="test")
        # Create a conditional affordance: door(2) requires key(3)
        theory.induce_conditional_affordance(
            feature=2,
            condition=CausalPredicate(variable="inventory", operator="contains", value=3),
            evidence="test_key_door",
            prerequisite_feature=3,
        )
        # Snapshot before clearing
        theory.snapshot_conditional_affordance_templates()
        old_priors = dict(theory.role_priors)
        old_templates = list(theory._role_conditional_templates)

        # Clear
        theory.clear_feature_bindings()

        # Feature→role gone
        assert theory.get_role(3) is None
        assert theory.get_role(2) is None

        # Priors and templates survive
        assert theory.role_priors == old_priors
        assert theory._role_conditional_templates == old_templates

    def test_snapshot_captures_role_level_templates(self) -> None:
        """snapshot_conditional_affordance_templates lifts feature→feature to role→role."""
        theory = self._make_theory()
        theory.bind_feature_to_role(3, "key")
        theory.bind_feature_to_role(2, "door")
        theory.induce_conditional_affordance(
            feature=2,
            condition=CausalPredicate(variable="inventory", operator="contains", value=3),
            evidence="key_door_link",
            prerequisite_feature=3,
        )
        theory.snapshot_conditional_affordance_templates()

        assert len(theory._role_conditional_templates) == 1
        tmpl = theory._role_conditional_templates[0]
        assert tmpl["feature_role"] == "door"
        assert tmpl["prerequisite_role"] == "key"

    def test_lazy_template_instantiation_on_bind(self) -> None:
        """bind_feature_to_role lazily re-creates conditional affordances from templates."""
        theory = self._make_theory()

        # Episode 1: key(3) → door(2) → goal(8)
        theory.bind_feature_to_role(3, "key")
        theory.bind_feature_to_role(2, "door")
        theory.bind_feature_to_role(8, "goal")
        theory.induce_conditional_affordance(
            feature=2,
            condition=CausalPredicate(variable="inventory", operator="contains", value=3),
            evidence="ep1_key_door",
            prerequisite_feature=3,
        )

        # Level boundary
        theory.snapshot_conditional_affordance_templates()
        theory.clear_feature_bindings()
        # Also clear conditional affordances (they reference old feature IDs)
        theory._conditional_affordances.clear()

        # Episode 2: new features — key(7) → door(4) → goal(9)
        # Bind key first
        theory.bind_feature_to_role(7, "key")
        # Bind door — should trigger template instantiation since both sides are bound
        theory.bind_feature_to_role(4, "door")

        # Verify conditional affordance was re-created for door(4) requiring key(7)
        aff = theory.get_conditional_affordance(4)
        assert aff is not None, (
            "Conditional affordance should be re-instantiated for new door feature"
        )
        assert aff.prerequisite_feature == 7, (
            f"Expected prerequisite key feature 7, got {aff.prerequisite_feature}"
        )
        assert aff.feature == 4

    def test_lazy_template_instantiation_reverse_bind_order(self) -> None:
        """Template re-instantiation works regardless of bind order (door first, then key)."""
        theory = self._make_theory()

        # Episode 1 setup
        theory.bind_feature_to_role(3, "key")
        theory.bind_feature_to_role(2, "door")
        theory.induce_conditional_affordance(
            feature=2,
            condition=CausalPredicate(variable="inventory", operator="contains", value=3),
            evidence="ep1",
            prerequisite_feature=3,
        )

        theory.snapshot_conditional_affordance_templates()
        theory.clear_feature_bindings()
        theory._conditional_affordances.clear()

        # Episode 2: bind door FIRST, then key
        theory.bind_feature_to_role(4, "door")
        # At this point key is not yet bound, no instantiation yet
        assert theory.get_conditional_affordance(4) is None

        # Now bind key — should trigger instantiation
        theory.bind_feature_to_role(7, "key")
        aff = theory.get_conditional_affordance(4)
        assert aff is not None, (
            "Conditional affordance should be instantiated when prerequisite role binds"
        )
        assert aff.prerequisite_feature == 7

    def test_no_spurious_instantiation_without_templates(self) -> None:
        """No conditional affordances created when no templates exist."""
        theory = self._make_theory()
        assert len(theory._role_conditional_templates) == 0

        theory.bind_feature_to_role(7, "key")
        theory.bind_feature_to_role(4, "door")

        assert theory.get_conditional_affordance(4) is None

    def test_role_priors_survive_multiple_transitions(self) -> None:
        """Role priors accumulate across multiple level transitions."""
        theory = self._make_theory()

        # Level 1
        theory.bind_feature_to_role(3, "goal")
        theory.clear_feature_bindings()

        # Level 2
        theory.bind_feature_to_role(8, "goal")
        theory.clear_feature_bindings()

        # Level 3
        theory.bind_feature_to_role(12, "goal")

        assert theory.has_role_prior("goal")
        assert theory.role_priors["goal"] >= 3


class TestEpisodicBoundaryRoleTransfer:
    """Integration test: role transfer wired through the episodic boundary dispatch."""

    def test_consolidate_schema_snapshots_and_clears_on_new_level(self) -> None:
        """consolidate_schema_and_reset snapshots templates and clears bindings on is_new_level."""
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
        from hbllm.hcir.world.engine_state import ReadOnlyEngineView, StateDelta

        engine = AutonomousEpistemicEngine()
        theory = engine.symbolic_theory

        # Simulate episode 1: learn key(3)→door(2)→goal(8)
        theory.bind_feature_to_role(3, "key")
        theory.bind_feature_to_role(2, "door")
        theory.induce_goal(8, evidence="terminal_reward")
        theory.induce_conditional_affordance(
            feature=2,
            condition=CausalPredicate(variable="inventory", operator="contains", value=3),
            evidence="test_key_door",
            prerequisite_feature=3,
        )

        # Fire level boundary
        view = ReadOnlyEngineView(engine)
        delta = StateDelta()
        engine.episodic_cortex.consolidate_schema_and_reset(
            view, delta, retain_dynamics=True, is_new_level=True
        )
        engine.apply_delta(delta, author="test")

        # After boundary: feature bindings should be cleared
        assert theory.get_role(3) is None, "Old feature→role bindings should be cleared"
        assert theory.get_role(2) is None

        # But role priors survive
        assert theory.has_role_prior("key")
        assert theory.has_role_prior("door")
        assert theory.has_role_prior("goal")

        # Templates survive
        assert len(theory._role_conditional_templates) >= 1

        # Now simulate episode 2: new features get bound
        theory.bind_feature_to_role(7, "key")
        theory.bind_feature_to_role(4, "door")

        # Conditional affordance should be lazily re-instantiated
        aff = theory.get_conditional_affordance(4)
        assert aff is not None, (
            "After level boundary + rebind, conditional affordance should be re-created"
        )
        assert aff.prerequisite_feature == 7

    def test_full_clear_resets_everything_including_priors(self) -> None:
        """theory.clear() (cross-game reset) resets ALL role state including priors and templates."""
        theory = HCIRSymbolicWorldTheory()
        theory.bind_feature_to_role(3, "key")
        theory.bind_feature_to_role(2, "door")
        theory.induce_conditional_affordance(
            feature=2,
            condition=CausalPredicate(variable="inventory", operator="contains", value=3),
            evidence="test",
            prerequisite_feature=3,
        )
        theory.snapshot_conditional_affordance_templates()

        theory.clear()

        assert theory.get_role(3) is None
        assert len(theory._role_conditional_templates) == 0
        assert len(theory.role_priors) == 0

    def test_same_level_retry_preserves_feature_bindings(self) -> None:
        """On same-level retry (death, is_new_level=False), feature bindings are NOT cleared."""
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
        from hbllm.hcir.world.engine_state import ReadOnlyEngineView, StateDelta

        engine = AutonomousEpistemicEngine()
        theory = engine.symbolic_theory

        theory.bind_feature_to_role(3, "key")
        theory.bind_feature_to_role(8, "goal")

        view = ReadOnlyEngineView(engine)
        delta = StateDelta()
        engine.episodic_cortex.consolidate_schema_and_reset(
            view, delta, retain_dynamics=True, is_new_level=False
        )
        engine.apply_delta(delta, author="test")

        # On same-level retry, bindings should persist
        assert theory.get_role(3) == "key"
        assert theory.get_role(8) == "goal"

    def test_induce_conditional_affordance_auto_binds_roles_for_template(self) -> None:
        """induce_conditional_affordance binds default roles so template prerequisite_role is non-None."""
        theory = HCIRSymbolicWorldTheory()
        theory.induce_walkable(13, evidence="traversal")
        theory.induce_conditional_affordance(
            feature=12,
            condition=CausalPredicate(variable="trigger", operator="==", value=13),
            evidence="remote_mech:13_clears_12",
            prerequisite_feature=13,
        )
        theory.snapshot_conditional_affordance_templates()
        assert len(theory._role_conditional_templates) == 1
        tmpl = theory._role_conditional_templates[0]
        assert tmpl["prerequisite_role"] == "actuator"
        assert tmpl["feature_role"] == "barrier"

    def test_shared_engine_relabeled_grid_nav_recovers_without_label_poisoning(self) -> None:
        """Shared engine detects relabeled palette shifts across levels and matches/beats fresh."""
        import random

        from benchmarks.envs.grid_nav import GridNavEnvironment
        from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
        from hbllm.hcir.world.percept import (
            EmbodiedObservation,
            EpisodeBoundary,
            EpisodeBoundaryKind,
        )

        keys = [2, 3, 5, 8, 9]

        def _make_relabeled_env(seed: int) -> tuple[GridNavEnvironment, dict[int, int]]:
            env = GridNavEnvironment(max_steps=100)
            rng = random.Random(1000 + seed)
            perm = list(keys)
            rng.shuffle(perm)
            mapping = dict(zip(keys, perm, strict=True))
            return env, mapping

        def _remap(obs: EmbodiedObservation, mapping: dict[int, int]) -> EmbodiedObservation:
            arr = obs.lattice.copy()
            out = arr.copy()
            for orig, dst in mapping.items():
                out[arr == orig] = dst
            return EmbodiedObservation(
                lattice=out,
                available_actions=list(obs.available_actions),
                terminal=obs.terminal,
                context_id=obs.context_id,
                episode_id=obs.episode_id,
                entities=obs.entities,
                metadata=obs.metadata,
            )

        n_seeds = 10
        shared_engine = AutonomousEpistemicEngine()
        shared_wins = 0
        shared_steps: list[int] = []
        fresh_wins = 0

        for ep in range(n_seeds):
            env, mapping = _make_relabeled_env(ep)
            obs = _remap(env.reset(seed=ep), mapping)
            done, steps, info = False, 0, {}
            while not done and steps < 100:
                act, _ = shared_engine.step(obs)
                raw_obs, _, done, info = env.step(act)
                obs = _remap(raw_obs, mapping)
                steps += 1
            won = bool(info.get("success"))
            if won:
                shared_wins += 1
            shared_steps.append(steps)
            shared_engine.on_boundary(
                EpisodeBoundary(
                    kind=EpisodeBoundaryKind.GOAL_REACHED
                    if won
                    else EpisodeBoundaryKind.DEATH
                    if info.get("dead")
                    else EpisodeBoundaryKind.CONTEXT_RESET,
                    retain_dynamics=True,
                    same_level=False,
                )
            )

            env_f, mapping_f = _make_relabeled_env(ep)
            eng_f = AutonomousEpistemicEngine()
            obs_f = _remap(env_f.reset(seed=ep), mapping_f)
            done_f, steps_f, info_f = False, 0, {}
            while not done_f and steps_f < 100:
                act_f, _ = eng_f.step(obs_f)
                raw_f, _, done_f, info_f = env_f.step(act_f)
                obs_f = _remap(raw_f, mapping_f)
                steps_f += 1
            if info_f.get("success"):
                fresh_wins += 1

        assert shared_engine.symbolic_theory.labels_non_stationary is True
        assert shared_wins >= fresh_wins, (
            f"Shared engine collapsed under relabeling: shared={shared_wins}/{n_seeds} < fresh={fresh_wins}/{n_seeds}"
        )
        assert sum(1 for s in shared_steps[2:] if s == 100) <= 1, (
            f"Shared engine timed out repeatedly after label shift detection: steps={shared_steps}"
        )


def test_fuzzy_actuator_priors_and_grid_history_boundary_reset() -> None:
    import numpy as np

    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
    from hbllm.hcir.world.motor.effector import MotorCortexEffector
    from hbllm.hcir.world.percept import EpisodeBoundary, EpisodeBoundaryKind
    from hbllm.hcir.world.planning.goal_scheduler import EpistemicGoalScheduler

    # 1. EpistemicGoalScheduler scores graded actuator/switch priors positively
    priors = {7: {"actuator": 0.85}, 8: {"switch": 0.90}}
    pref_act, _, _ = EpistemicGoalScheduler._role_scores_from_priors(priors, 7, has_cargo_ctx=False)
    pref_sw, _, _ = EpistemicGoalScheduler._role_scores_from_priors(priors, 8, has_cargo_ctx=True)
    assert pref_act > 0.7
    assert pref_sw > 0.7

    # 2. MotorCortexEffector prioritizes entities with high fuzzy actuator prior over equal-area distractor
    engine = AutonomousEpistemicEngine()
    grid = np.zeros((8, 8), dtype=int)
    grid[1, 1] = 1  # avatar
    grid[2, 5] = 4  # distractor entity (area 1)
    grid[5, 2] = 7  # actuator entity (area 1)
    engine.avatar_feature = 1
    engine.avatar_features = {1}
    engine.avatar_pos = (1, 1)
    engine.bg_feature = 0
    engine.working_memory.concept_role_priors = {7: {"actuator": 0.95}}

    grounded = MotorCortexEffector.ground_effector_action(engine, grid, action=6)
    assert grounded is not None
    assert (grounded["y"], grounded["x"]) == (5, 2)

    # 3. LEVEL_COMPLETE boundary clears grid_history to prevent cross-level shape mismatch
    engine.grid_history.append(np.zeros((8, 8), dtype=int))
    engine.grid_history.append(np.zeros((8, 8), dtype=int))
    assert len(engine.grid_history) == 2
    engine.on_boundary(
        EpisodeBoundary(
            kind=EpisodeBoundaryKind.LEVEL_COMPLETE,
            retain_dynamics=True,
            same_level=False,
        )
    )
    assert len(engine.grid_history) == 0


def test_dirichlet_walkable_posterior_resists_exogenous_timeout_death() -> None:
    import numpy as np

    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
    from hbllm.hcir.world.causal.transition_kb import TransitionOutcome
    from hbllm.hcir.world.motor_calibration import ActionDynamicsModel

    engine = AutonomousEpistemicEngine()
    # Grid where perimeter border is color 4 and interior floor is color 3
    grid = np.full((12, 12), 4, dtype=int)
    grid[2:10, 2:10] = 3
    grid[5, 5] = 2  # avatar
    grid[5, 6] = 3  # floor ahead

    engine.avatar_feature = 2
    engine.avatar_features = {2}
    engine.avatar_pos = (5, 5)
    engine.bg_feature = 4
    engine.action_dynamics[4] = ActionDynamicsModel(
        action_id=4, delta_r=0, delta_c=1, confidence=1.0, probes_tested=5
    )

    # Record verified safe traversals on floor feature 3
    engine.symbolic_theory.induce_walkable(3, evidence="test_floor")
    engine.symbolic_theory.mark_verified_in_context("walkable", 3)
    for _ in range(20):
        engine.symbolic_theory.transition_kb.record_outcome(
            3, TransitionOutcome.TRAVERSED, scope="feature", weight=1.0
        )

    # Register target cell (5, 6) as previously visited + active colinear exogenous gauge (resource bar)
    engine.working_memory.visited_cells.add((5, 6))
    for step_idx in range(4):
        engine.working_memory.exogenous_tracker.observe_transition(
            [(10, 9 - step_idx)], action=4, avatar_locus=(5, 5)
        )
    assert engine.working_memory.exogenous_tracker.has_active_colinear_gauge() is True

    engine.prev_grid = grid.copy()
    engine.last_action = 4
    death_grid = grid.copy()
    death_grid[5, 5] = 3
    death_grid[5, 6] = 2

    engine.assimilate_feedback(death_grid, [1, 2, 3, 4], is_failure=True)

    # Floor feature 3 must remain walkable with high p_pass and must not be turned into a barrier
    assert engine.symbolic_theory.is_walkable(3) is True
    assert 3 not in engine.symbolic_theory.barrier_features
    assert (5, 6) not in engine.hazard_tracker.static_lethal_positions
    belief = engine.symbolic_theory.transition_kb.get_belief(3, scope="feature")
    assert belief is not None
    assert belief.dirichlet_posterior().p_pass > 0.9


def test_state_conditioned_lock_reopens_across_cyclic_transformer_states() -> None:
    import numpy as np

    from hbllm.hcir.world.assimilation.assimilator import EpistemicFeedbackAssimilator
    from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
    from hbllm.hcir.world.motor_calibration import ActionDynamicsModel
    from hbllm.hcir.world.perception.engine import PerceptionEngine

    engine = AutonomousEpistemicEngine()
    engine.avatar_feature = 2
    engine.avatar_features = {2}
    engine.bg_feature = 0
    engine.symbolic_theory.induce_walkable(0, evidence="bg")
    engine.action_dynamics[2] = ActionDynamicsModel(
        action_id=2, delta_r=1, delta_c=0, confidence=1.0, probes_tested=5
    )

    grid = np.zeros((10, 10), dtype=int)
    grid[3, 3] = 2  # avatar next to bounded conditional gate
    grid[4, 3] = 6  # bounded conditional gate
    grid[3, 7] = 8  # cyclic transformer pad
    grid[8, 1] = 2  # distant state indicator sharing avatar color

    # 1. Blocked against bounded gate under initial state S_0=0
    engine.avatar_pos = (3, 3)
    engine.prev_grid = grid.copy()
    engine.last_action = 2
    engine.failed_transitions.add(((3, 3), 2))
    engine.working_memory.habenular_ior.failed_transitions.add(((3, 3), 2))
    engine.exhausted_candidate_goals.add((4, 3))
    EpistemicFeedbackAssimilator._record_conditioned_block_if_gate(
        engine, grid, (3, 3), 2, bg=0, H=10, W=10
    )

    # 2. Avatar steps onto pad (3, 7) and distant indicator at (8, 2) mutates to 2 (State S_1)
    prev_pad = grid.copy()
    prev_pad[3, 3] = 0
    prev_pad[3, 6] = 2
    curr_pad = prev_pad.copy()
    curr_pad[3, 6] = 0
    curr_pad[3, 7] = 2
    curr_pad[8, 1] = 0
    curr_pad[8, 2] = 2  # indicator shifted in-place (<= 3.5 px) sharing avatar color 2

    engine.prev_grid = prev_pad
    engine.avatar_pos = (3, 7)
    diff = PerceptionEngine.compute_frame_diff(prev_pad, curr_pad)
    EpistemicFeedbackAssimilator._assimilate_state_mutations_and_cargo(
        engine,
        curr_pad,
        diff,
        action=4,
        action_data=None,
        is_effector_action=False,
        is_known_displacement=True,
        available_actions=[1, 2, 3, 4],
        is_success=False,
        is_failure=False,
        landing_cells_now=[(3, 7)],
        prev_avatar_pos=(3, 6),
        bg=0,
        H=10,
        W=10,
    )

    # Gate (3, 3) + action 2 & target (4, 3) must re-open under S_1 != 0, and pad 8 becomes actuator
    assert engine.working_memory.body_schema.latent_state_signature != 0
    assert ((3, 3), 2) not in engine.failed_transitions
    assert ((3, 3), 2) not in engine.working_memory.habenular_ior.failed_transitions
    assert (4, 3) not in engine.exhausted_candidate_goals
    assert 8 in engine.symbolic_theory.actuator_features

    # 3. Under S_1, if gate (3, 3)+action 2 still fails, transformer (3, 7) re-arms to cycle to S_2
    engine.exhausted_candidate_goals.add((3, 7))
    engine.symbolic_theory.mark_locus_explored((3, 7), reason="test")
    EpistemicFeedbackAssimilator._record_conditioned_block_if_gate(
        engine, curr_pad, (3, 3), 2, bg=0, H=10, W=10
    )
    assert (3, 7) not in engine.exhausted_candidate_goals

    # 4. Cycling back to an already-seen state with no newly reopened transitions marks transformer exhausted
    sig_s1 = engine.working_memory.body_schema.latent_state_signature
    engine.working_memory.body_schema.advance_latent_state(sig_s1 + 999, trigger_pos=(3, 7))
    engine.working_memory.body_schema.record_conditioned_block((3, 3), 2, target_pos=(4, 3))
    engine.working_memory.body_schema.advance_latent_state(sig_s1, trigger_pos=(3, 7))
    assert (3, 7) in engine.working_memory.body_schema.exhausted_transformers
