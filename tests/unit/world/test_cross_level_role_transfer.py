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
        from hbllm.hcir.world.experimental import restore_all as _restore_all

        engine = AutonomousEpistemicEngine()
        _restore_all(engine)
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
        from hbllm.hcir.world.experimental import restore_all as _restore_all

        engine = AutonomousEpistemicEngine()
        _restore_all(engine)
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
