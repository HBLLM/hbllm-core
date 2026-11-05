"""Regression: each non-terminal env step must trigger exactly ONE feedback
assimilation through the engine (the decide()/pipeline seam AND MyAgent's
explicit death/level-complete calls must not double-assimilate the same step).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from hbllm.drivers.base import DriverAction, DriverFeedback, DriverInput
from plugins.arc_agi_adapter.local_env import LocalEnv  # noqa: E402
from plugins.arc_agi_adapter.my_agent import CognitiveAgentAdapter, MyAgent


def _run_steps(n_steps: int = 8) -> list[int]:
    """Drive CognitiveAgentAdapter + real MyAgent + real engine through
    nav_basic; return assimilate_feedback call counts per env step."""
    import types

    # run_all_25_games imports the ARC-AGI-3 SDK at module level; the test only
    # needs CognitiveAgentAdapter (driven against the local env, not Arcade).
    if "arc_agi" not in sys.modules:
        try:
            import arc_agi  # noqa: F401
        except ModuleNotFoundError:
            stub = types.ModuleType("arc_agi")
            stub.Arcade = object  # type: ignore[attr-defined]
            sys.modules["arc_agi"] = stub

    agent = MyAgent()
    adapter = CognitiveAgentAdapter(agent)
    engine = agent.internal_engine

    # Spy the single seam every assimilation flows through.
    calls: list[int] = [0]
    orig = engine.assimilate_feedback

    def spy(*a, **k):
        calls[0] += 1
        return orig(*a, **k)

    engine.assimilate_feedback = spy

    env = LocalEnv(0).make("nav_basic")
    frame = env.reset()
    adapter.reset_episode()

    # Drive the adapter the same way run_all_25_games does: observe -> decide -> step -> update.
    per_step = []

    for _ in range(n_steps):
        calls[0] = 0
        adapter.observe(
            DriverInput(
                raw_data=None,
                metadata={"frame_obj": frame, "StepCounter": getattr(env, "steps", 0)},
            )
        )
        action: DriverAction = adapter.decide(frame.available_actions)
        next_frame = env.step(
            action.action_id,
            action.parameters or {},
        )
        adapter.update(
            action,
            DriverFeedback(
                success=True,
                causal_delta=np.asarray(next_frame.grid),
                info={
                    "StepCounter": getattr(env, "steps", 0),
                    "levels_completed": next_frame.levels_completed,
                },
            ),
        )
        per_step.append(calls[0])
        frame = next_frame
    return per_step


def test_single_assimilation_per_nonterminal_step():
    counts = _run_steps(8)
    # First step may be 0 (no previous frame to assimilate against);
    # every subsequent non-terminal step must assimilate exactly once.
    for i, c in enumerate(counts):
        expected = 0 if i == 0 else 1
        assert c == expected, (
            f"step {i}: expected {expected} assimilation(s), got {c} "
            f"(counts={counts}) — double-assimilation detected"
        )
