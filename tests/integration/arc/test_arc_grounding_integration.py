from __future__ import annotations

"""Integration tests for ARC-AGI Plugin with SGE Grounding.

Verifies that MyAgent runs on ARC environments with the SGE CorticalGroundingBridge,
tracking entities, extracting features, and assigning concepts across steps.
"""

import sys
import types

# Ensure arc_agi module is stubbed if not installed
if "arc_agi" not in sys.modules:
    try:
        import arc_agi  # noqa: F401
    except ModuleNotFoundError:
        stub = types.ModuleType("arc_agi")
        stub.Arcade = object  # type: ignore[attr-defined]
        sys.modules["arc_agi"] = stub

from hbllm.drivers.base import DriverAction, DriverInput
from plugins.arc_agi_adapter.local_env import LocalEnv
from plugins.arc_agi_adapter.my_agent import CognitiveAgentAdapter, MyAgent


class TestARCGroundingIntegration:
    """Tests MyAgent in LocalEnv with grounding bridge attached."""

    def test_my_agent_has_grounding_bridge(self) -> None:
        agent = MyAgent(enable_grounding=True)
        assert agent.grounding_bridge is not None
        assert agent.internal_engine.grounding_bridge is agent.grounding_bridge

    def test_arc_steps_track_entities_and_form_concepts(self) -> None:
        agent = MyAgent(enable_grounding=True)
        adapter = CognitiveAgentAdapter(agent)
        env = LocalEnv(0).make("nav_basic")
        frame = env.reset()
        adapter.reset_episode()

        bridge = agent.grounding_bridge
        assert bridge is not None

        # Execute 5 environment steps
        for step_idx in range(5):
            adapter.observe(
                DriverInput(
                    raw_data=None,
                    metadata={"frame_obj": frame, "StepCounter": step_idx},
                )
            )
            action: DriverAction = adapter.decide(frame.available_actions)
            assert action.action_id in frame.available_actions

            frame = env.step(action.action_id, action.parameters or {})
            adapter.update(
                action,
                types.SimpleNamespace(success=True, state_delta=None),
            )

        # After 5 steps, verify grounding activity
        active_candidates = bridge.tracker.get_active_candidates()
        assert len(active_candidates) >= 1, "Entities should be actively tracked"

        # Verify feature extraction and concept assignment
        assert len(bridge.state.last_feature_vectors) >= 1, (
            "Features should be extracted from entities"
        )
        assert len(bridge.state.last_concept_assignments) >= 1, "Concepts should be assigned"

        # Verify concepts are recorded in store
        all_concepts = bridge.concept_store.get_all_concepts()
        assert len(all_concepts) >= 1, "At least one concept should be formed in store"

    def test_arc_game_switch_retains_grounding_bridge(self) -> None:
        agent = MyAgent(enable_grounding=True)
        env = LocalEnv(0).make("nav_basic")
        frame = env.reset()

        adapter = CognitiveAgentAdapter(agent)
        adapter.observe(
            DriverInput(
                raw_data=None,
                metadata={"frame_obj": frame, "StepCounter": 0},
            )
        )
        _ = adapter.decide(frame.available_actions)

        # Verify initial bridge exists
        assert agent.grounding_bridge is not None

        # Simulate game switch
        agent.game_id = "new_game_42"
        frame_new = env.reset()
        adapter.observe(
            DriverInput(
                raw_data=None,
                metadata={"frame_obj": frame_new, "StepCounter": 0},
            )
        )
        _ = adapter.decide(frame_new.available_actions)

        # Grounding bridge should still be attached on the new internal engine
        assert agent.grounding_bridge is not None
