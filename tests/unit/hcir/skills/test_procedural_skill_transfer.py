from __future__ import annotations

import numpy as np

from hbllm.hcir.bytecode import Opcode
from hbllm.hcir.nodes import SkillNode
from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    DeclarativeNeuroSymbolicSkill,
    SkillEvaluationContext,
    SubgoalSequence,
    SymbolicSubgoal,
)
from hbllm.hcir.spatial_planner import SpatialActionIntent
from hbllm.hcir.workspace_tiers import TieredWorkspace
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.medial_prefrontal_schema import (
    GeneralizedTaskSchema,
    MedialPrefrontalSchemaEngine,
)
from hbllm.hcir.world.skill_program_synthesizer import (
    ActionTraceStep,
    SkillProgramSynthesizer,
)


class TestProceduralSkillTransferAndSchemaEngine:
    """Validates cross-context procedural skill persistence in TieredWorkspace.persistent,
    abstract task schema induction in MedialPrefrontalSchemaEngine, and zero-shot
    predicate-signature skill transfer across environments.
    """

    def test_skill_serialization_and_deserialization(self) -> None:
        """Verify serializing and reconstructing DeclarativeNeuroSymbolicSkills."""
        synthesizer = SkillProgramSynthesizer(min_support_traces=1)

        pattern = (
            SpatialActionIntent.NAVIGATE_TO,
            SpatialActionIntent.INTERACT,
            SpatialActionIntent.ACTUATE,
        )
        skill = synthesizer.synthesize_skill_from_pattern(
            pattern=pattern,
            skill_name="unlock_and_switch_skill",
            required_actions={1, 2, 4},
        )

        serialized = synthesizer.serialize_skill(skill)
        assert serialized["skill_name"] == "unlock_and_switch_skill"
        assert serialized["required_actions"] == [1, 2, 4]
        assert len(serialized["subgoals"]) == 3
        assert serialized["subgoals"][0]["intent"] == "NAVIGATE"
        assert serialized["subgoals"][1]["intent"] == "INTERACT"
        assert serialized["subgoals"][2]["intent"] == "ACTIVATE"

        # Deserialize into fresh skill instance
        restored = synthesizer.deserialize_skill(serialized)
        assert restored.skill_name == "unlock_and_switch_skill"
        assert isinstance(restored.signature, ActionAffordancePredicate)
        assert restored.signature.required == {1, 2, 4}
        assert len(restored.program.subgoals) == 3

        # Verify compilation to bytecode
        grid = np.zeros((5, 5), dtype=np.int32)
        stream = restored.compile_to_instruction_stream(grid, available_actions=[1, 2, 4])
        assert len(stream.instructions) == 9  # 3 instructions per subgoal (QUERY, ASSERT, EXECUTE)
        opcodes = [ins.opcode for ins in stream.instructions]
        assert opcodes == [
            Opcode.QUERY,
            Opcode.ASSERT,
            Opcode.EXECUTE,
            Opcode.QUERY,
            Opcode.ASSERT,
            Opcode.EXECUTE,
            Opcode.QUERY,
            Opcode.ASSERT,
            Opcode.EXECUTE,
        ]

    def test_skill_persistence_and_loading_in_tiered_workspace(self) -> None:
        """Verify committing procedural skills to workspace.persistent and restoring them."""
        workspace = TieredWorkspace()
        synthesizer = SkillProgramSynthesizer(min_support_traces=1)

        # Synthesize multiple skills
        skill_a = synthesizer.synthesize_skill_from_pattern(
            pattern=(SpatialActionIntent.NAVIGATE_TO, SpatialActionIntent.INTERACT),
            skill_name="navigate_interact_skill",
            required_actions={1, 2},
        )
        skill_b = synthesizer.synthesize_skill_from_pattern(
            pattern=(SpatialActionIntent.ALIGN, SpatialActionIntent.MANIPULATE),
            skill_name="align_manipulate_skill",
            required_actions={3, 4},
        )

        assert skill_a.skill_name == "navigate_interact_skill"
        assert skill_b.skill_name == "align_manipulate_skill"

        # Persist to persistent tier
        persisted_ids = synthesizer.persist_to_workspace(workspace, tenant_id="tenant_alpha")
        assert len(persisted_ids) == 2
        assert "skill_navigate_interact_skill" in persisted_ids
        assert "skill_align_manipulate_skill" in persisted_ids

        # Inspect nodes in persistent workspace graph
        nodes = workspace.persistent.graph.nodes
        assert "skill_navigate_interact_skill" in nodes
        assert "skill_align_manipulate_skill" in nodes

        node_a = nodes["skill_navigate_interact_skill"]
        assert isinstance(node_a, SkillNode)
        assert node_a.skill_name == "navigate_interact_skill"
        assert "serialized_skill" in node_a.properties

        # Load from persistent workspace using a fresh synthesizer instance
        fresh_synthesizer = SkillProgramSynthesizer(min_support_traces=1)
        assert len(fresh_synthesizer.synthesized_skills) == 0

        loaded_count = fresh_synthesizer.load_from_workspace(workspace, tenant_id="tenant_alpha")
        assert loaded_count == 2
        assert "navigate_interact_skill" in fresh_synthesizer.synthesized_skills
        assert "align_manipulate_skill" in fresh_synthesizer.synthesized_skills

        restored_a = fresh_synthesizer.synthesized_skills["navigate_interact_skill"]
        assert restored_a.skill_name == "navigate_interact_skill"
        assert len(restored_a.program.subgoals) == 2

    def test_medial_prefrontal_schema_engine_skill_registration(self) -> None:
        """Verify registering synthesized skills into abstract mPFC task schemas."""
        schema_engine = MedialPrefrontalSchemaEngine(match_threshold=0.30)
        synthesizer = SkillProgramSynthesizer(min_support_traces=1)

        skill = synthesizer.synthesize_skill_from_pattern(
            pattern=(SpatialActionIntent.NAVIGATE_TO, SpatialActionIntent.INTERACT),
            skill_name="barrier_passage",
            required_actions={1, 2},
        )

        schema = schema_engine.register_skill_schema(skill)
        assert isinstance(schema, GeneralizedTaskSchema)
        assert schema.schema_id == "schema_barrier_passage"
        assert "actor" in schema.canonical_roles
        assert "navigate" in schema.subgoal_sequence_hints[0]
        assert "interact" in schema.subgoal_sequence_hints[1]

    def test_predicate_signature_matching_and_transfer_query(self) -> None:
        """Verify querying transferable skills via predicate signatures."""
        schema_engine = MedialPrefrontalSchemaEngine(match_threshold=0.35)

        # Skill 1: Navigation / Interaction (actions 1, 2, 4)
        class NavigationSkill(DeclarativeNeuroSymbolicSkill):
            skill_name = "nav_skill"
            signature = ActionAffordancePredicate(required={1, 2, 4})
            program = SubgoalSequence(
                SymbolicSubgoal(intent=SpatialActionIntent.NAVIGATE_TO),
                SymbolicSubgoal(intent=SpatialActionIntent.INTERACT),
            )

        # Skill 2: Manipulation / Color Toggle (actions 5, 6)
        class ToggleSkill(DeclarativeNeuroSymbolicSkill):
            skill_name = "toggle_skill"
            signature = ActionAffordancePredicate(required={5, 6})
            program = SubgoalSequence(
                SymbolicSubgoal(intent=SpatialActionIntent.ACTUATE),
            )

        skill_1 = NavigationSkill()
        skill_2 = ToggleSkill()
        schema_engine.register_skill_schema(skill_1)
        schema_engine.register_skill_schema(skill_2)

        # Environment Alpha: actions [1, 2, 3, 4, 7]
        grid = np.zeros((10, 10), dtype=np.int32)
        ctx_alpha = SkillEvaluationContext(grid=grid, available_actions=[1, 2, 3, 4, 7])

        matches_alpha = schema_engine.query_transferable_skills(
            context=ctx_alpha,
            candidate_skills=[skill_1, skill_2],
            min_confidence=0.40,
        )

        assert len(matches_alpha) == 1
        best_skill, conf, reason = matches_alpha[0]
        assert best_skill.skill_name == "nav_skill"
        assert conf >= 0.80
        assert "predicate_signature_match" in reason

        # Environment Beta: actions [5, 6, 7]
        ctx_beta = SkillEvaluationContext(grid=grid, available_actions=[5, 6, 7])
        matches_beta = schema_engine.query_transferable_skills(
            context=ctx_beta,
            candidate_skills=[skill_1, skill_2],
            min_confidence=0.40,
        )

        assert len(matches_beta) == 1
        best_beta, conf_beta, _ = matches_beta[0]
        assert best_beta.skill_name == "toggle_skill"
        assert conf_beta >= 0.80

    def test_end_to_end_cross_context_skill_transfer_in_engine(self) -> None:
        """Verify full end-to-end procedural skill consolidation, persistence, and zero-shot transfer."""
        workspace = TieredWorkspace()
        engine = AutonomousEpistemicEngine(tiered_workspace=workspace)

        # Ensure faculties are active and live
        assert isinstance(engine.skill_synthesizer, SkillProgramSynthesizer)
        assert isinstance(engine.mpfc_schema_engine, MedialPrefrontalSchemaEngine)

        # Record a winning behavioral trace from Environment A
        trace_steps = [
            ActionTraceStep(
                intent=SpatialActionIntent.NAVIGATE_TO,
                target_query={"role": "key"},
                parameters={"action_id": 1, "reward": 0.0},
            ),
            ActionTraceStep(
                intent=SpatialActionIntent.NAVIGATE_TO,
                target_query={"role": "door"},
                parameters={"action_id": 2, "reward": 0.0},
            ),
            ActionTraceStep(
                intent=SpatialActionIntent.INTERACT,
                target_query={"role": "goal"},
                parameters={"action_id": 4, "reward": 1.0},
            ),
        ]

        engine.skill_synthesizer.record_winning_trace(
            steps=trace_steps,
            required_actions={1, 2, 4},
            terminal_reward=1.0,
        )

        # Consolidate memory: DreamCoder induction + persistence + mPFC registration
        res = engine.consolidate_episodic_memory(terminal_reward=1.0)
        assert len(res["induced_skills"]) > 0
        induced_skill = res["induced_skills"][0]

        # Verify skill was committed to persistent workspace
        persisted_nodes = [
            n for n in workspace.persistent.graph.nodes.values() if isinstance(n, SkillNode)
        ]
        assert len(persisted_nodes) >= 1

        # Now simulate entering a completely new Environment B
        grid_env_b = np.zeros((8, 8), dtype=np.int32)
        grid_env_b[1, 1] = 2  # Agent
        grid_env_b[6, 6] = 3  # Goal
        actions_env_b = [1, 2, 3, 4, 7]  # Actions 1, 2, 4 are available

        transferred = engine.query_transferable_skills(
            grid=grid_env_b,
            available_actions=actions_env_b,
            min_confidence=0.40,
        )

        assert len(transferred) >= 1
        transferred_skill, confidence, reason = transferred[0]
        assert confidence >= 0.80
        assert transferred_skill.skill_name == induced_skill.skill_name

        # Verify zero-shot executable bytecode generation for Environment B
        stream = transferred_skill.compile_to_instruction_stream(
            grid=grid_env_b,
            available_actions=actions_env_b,
        )
        assert len(stream.instructions) > 0
        assert stream.author == transferred_skill.skill_name
