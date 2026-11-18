from __future__ import annotations

from benchmarks.envs.grid_nav import GridNavEnvironment
from hbllm.hcir.grounding.contracts.dialogue import GoalProposal
from hbllm.hcir.grounding.contracts.references import GroundedEntityRef
from hbllm.hcir.grounding.features.composite_observation_adapter import CompositeObservationAdapter
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


class TestEndToEndLanguageGrounding:
    """Verifies end-to-end language grounding from raw sensory observation to dialogue goal proposal.

    The composed cognitive system observes a procedural grid environment, extracts visual entity
    components, learns perceptual concepts with derived semantic labels without hand-seeding, and
    binds natural language utterances to the exact perceived physical entities.
    """

    def test_end_to_end_perceptual_language_grounding_on_grid_nav(self) -> None:
        """Run composed system on GridNavEnvironment, let perception learn a concept

        for the red component, then publish an utterance 'push the red block left'
        through CompositeObservationAdapter.adapt_dialogue; assert a GoalProposal
        whose target is the GroundedEntityRef that perception created with zero hand-seeded aliases.
        """
        # 1. Instantiate procedural closed-loop environment
        env = GridNavEnvironment(size=10)
        obs = env.reset(seed=42)

        # 2. Boot composed cognitive system with verified governance and active grounding
        config = CognitiveSystemConfig(
            world_id="e2e_grounded_language_test",
            seed=42,
            enable_grounding=True,
            perception_mode="handcrafted",
        )
        system = build_cognitive_system(config)
        assert system.bridge is not None, "Grounding bridge must be attached"

        # 3. Observe the environment lattice (raw sensory data without oracle entity labels)
        action = system.engine.step(obs)
        assert action is not None

        # 4. Verify perception extracted entities and tracker registered candidates
        active_candidates = system.bridge.tracker.get_active_candidates()
        assert len(active_candidates) > 0, "Perception must track active candidates"

        # Find the candidate corresponding to the red component (value 2 in GridNav lattice)
        red_candidates = [
            c
            for c in active_candidates
            if c.metadata.get("color") == 2 or getattr(c, "color", None) == 2
        ]
        assert len(red_candidates) >= 1, "Perception must extract the red component"
        red_cand = red_candidates[0]
        red_cand_id = red_cand.candidate_ref.id

        # Verify concept store holds a concept referencing the red entity as exemplar
        matching_concepts = [
            c
            for c in system.bridge.concept_store.get_all_concepts()
            if red_cand_id in c.exemplar_entity_ids
        ]
        assert len(matching_concepts) >= 1, "Concept store must learn concept for red component"
        red_concept = matching_concepts[0]

        # Verify concept acquired linguistic aliases through sensory feature categorization (no hand-seeding)
        assert any("red" in alias for alias in red_concept.linguistic_aliases), (
            "Concept must naturally acquire 'red' linguistic alias from sensory categorization"
        )

        # 5. Publish utterance through CompositeObservationAdapter.adapt_dialogue
        adapter = CompositeObservationAdapter()
        envelope, _ = adapter.adapt_dialogue("push the red block left")
        assert envelope.metadata["text"] == "push the red block left"

        # 6. Process dialogue through the grounded cognitive system
        result = system.process_dialogue(envelope)

        # 7. Assertions on the produced GoalProposal
        assert len(result.goal_proposals) >= 1, "Interpretation must produce a GoalProposal"
        goal = result.goal_proposals[0]

        assert isinstance(goal, GoalProposal)
        assert goal.goal_intent == "push"
        assert isinstance(goal.target_entity, GroundedEntityRef)
        assert goal.target_entity.id in (red_cand_id, f"e_{red_cand_id}"), (
            f"GoalProposal target ({goal.target_entity.id}) must match "
            f"the GroundedEntityRef that perception created ({red_cand_id})"
        )
        assert goal.parameters.get("direction") == "left"
        assert goal.parameters.get("subject") == "red block"
