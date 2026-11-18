"""Unit tests for Advanced Advancements:
Track A: Curricular Skill Graph & Subgoal Decomposition (BA10)
Track B: Structural Causal Model (SCM) & Pearl's do(a) Interventions
Track D: Cerebellar Temporal SNN Pacing & Hazard Forecasting
"""

from __future__ import annotations

import numpy as np

from hbllm.hcir.skills.curriculum import (
    CurricularSkillGraph,
    SkillTier,
)
from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    DeclarativeNeuroSymbolicSkill,
    SubgoalSequence,
    SymbolicSubgoal,
)
from hbllm.hcir.spatial_planner import SpatialActionIntent
from hbllm.hcir.world.frontopolar_subgoal_stack import (
    FrontopolarSubgoal,
    FrontopolarSubgoalStack,
    SubgoalType,
)


class TestCurricularSkillGraphAndSubgoalDecomposition:
    """Tests for Track A: Curricular Skill Graph and Frontopolar Subgoal Decomposition."""

    def test_curricular_skill_registration_and_query(self):
        graph = CurricularSkillGraph()

        # 1. Register micro-skill (move/navigate)
        micro_skill = DeclarativeNeuroSymbolicSkill()
        micro_skill.skill_name = "MicroNavigate"
        micro_skill.signature = ActionAffordancePredicate(required={1, 2})
        micro_skill.program = SubgoalSequence(
            SymbolicSubgoal(intent=SpatialActionIntent.NAVIGATE, target_query={"role": "corridor"})
        )
        graph.register_skill(micro_skill, tier=SkillTier.MICRO, target_roles={"corridor"})

        # 2. Register meso-skill (key acquisition)
        meso_skill = DeclarativeNeuroSymbolicSkill()
        meso_skill.skill_name = "AcquireKey"
        meso_skill.signature = ActionAffordancePredicate(required={1, 2, 3})
        meso_skill.program = SubgoalSequence(
            SymbolicSubgoal(intent=SpatialActionIntent.NAVIGATE, target_query={"role": "key"}),
            SymbolicSubgoal(intent=SpatialActionIntent.INTERACT, target_query={"role": "key"}),
        )
        graph.register_skill(
            meso_skill,
            tier=SkillTier.MESO,
            dependencies={"MicroNavigate"},
            target_roles={"key"},
        )

        grid = np.zeros((10, 10))
        # Query with actions {1, 2}
        matches_micro = graph.find_matching_skills(grid, available_actions={1, 2})
        assert len(matches_micro) == 1
        assert matches_micro[0].skill_name == "MicroNavigate"

        # Query with actions {1, 2, 3} targeting "key"
        matches_key = graph.find_matching_skills(
            grid, available_actions={1, 2, 3}, target_role="key"
        )
        assert len(matches_key) == 1
        assert matches_key[0].skill_name == "AcquireKey"
        assert matches_key[0].tier == SkillTier.MESO

    def test_macro_skill_composition_and_decomposition(self):
        graph = CurricularSkillGraph()

        skill_a = DeclarativeNeuroSymbolicSkill()
        skill_a.skill_name = "KeySkill"
        skill_a.signature = ActionAffordancePredicate(required={1})
        skill_a.program = SubgoalSequence(
            SymbolicSubgoal(intent=SpatialActionIntent.NAVIGATE, target_query={"role": "key"})
        )
        graph.register_skill(skill_a, tier=SkillTier.MESO, target_roles={"key"})

        skill_b = DeclarativeNeuroSymbolicSkill()
        skill_b.skill_name = "DoorSkill"
        skill_b.signature = ActionAffordancePredicate(required={2})
        skill_b.program = SubgoalSequence(
            SymbolicSubgoal(intent=SpatialActionIntent.ACTUATE, target_query={"role": "door"})
        )
        graph.register_skill(skill_b, tier=SkillTier.MESO, target_roles={"door"})

        macro = graph.compose_macro_skill("SolveLevelMacro", ["KeySkill", "DoorSkill"])
        assert macro is not None
        assert macro.skill_name == "SolveLevelMacro"
        assert len(macro.program.subgoals) == 2

        # Decompose
        children = graph.decompose_macro_skill("SolveLevelMacro")
        assert len(children) == 2
        assert [c.skill_name for c in children] == ["KeySkill", "DoorSkill"]

    def test_frontopolar_subgoal_skill_binding_and_decomposition(self):
        stack = FrontopolarSubgoalStack(max_depth=8)

        parent_sg = FrontopolarSubgoal(
            subgoal_id="sg_master_0",
            subgoal_type=SubgoalType.DELIVER_TO_GOAL,
            target_entity_pos=(1, 1),
            target_destination=(8, 8),
            priority=1.0,
        )
        stack.push_subgoal(parent_sg)

        # Build declarative skill with 2 subgoals
        skill = DeclarativeNeuroSymbolicSkill()
        skill.skill_name = "TwoStagePush"
        skill.program = SubgoalSequence(
            SymbolicSubgoal(
                intent=SpatialActionIntent.NAVIGATE,
                target_query={"role": "barrier", "pos": (3, 3)},
            ),
            SymbolicSubgoal(
                intent=SpatialActionIntent.ACTUATE,
                target_query={"role": "goal", "pos": (8, 8)},
            ),
        )

        parent_sg.bind_skill(skill)
        assert parent_sg.bound_skill_name == "TwoStagePush"

        # Decompose in stack
        decomposed = stack.decompose_subgoal_with_skill("sg_master_0", skill)
        assert len(decomposed) == 2
        assert decomposed[0].subgoal_type == SubgoalType.CLEAR_OBSTACLE
        assert decomposed[0].target_entity_pos == (3, 3)
        assert decomposed[1].subgoal_type == SubgoalType.DELIVER_TO_GOAL
        assert decomposed[1].target_destination == (8, 8)
        assert stack.depth == 2
