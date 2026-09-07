"""
Unit and adversarial tests for BabyAI Tier 5: Causal Obstacle Relocation & Unblocking (Unblock).

Verifies:
1. Detection of critical path obstruction by movable objects.
2. Causal unblocking detour planning (pickup obstacle -> relocate to free tile -> drop).
3. Primary task completion following successful unblocking.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add core/ and core/plugins/ to sys.path
_test_dir = Path(__file__).resolve().parent
_plugin_dir = _test_dir.parent
_plugins_root = _plugin_dir.parent
_core_root = _plugins_root.parent

for p in [str(_core_root), str(_plugins_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from babyai_adapter import (
    BabyAIActionAdapter,
    BabyAIMissionParser,
    BabyAIPerceptionAdapter,
    MiniGridAction,
    create_blocked_level,
)


def test_hermetic_unblocking_mechanics() -> None:
    """Verify that an agent can unblock an obstructed doorway and reach the goal."""
    # Door at (4, 2) is blocked by purple ball at (3, 2)
    # Key is at (2, 1), target red box is at (7, 2)
    env = create_blocked_level(
        mission="pick up the red box",
        target_pos=(7, 2),
        door_pos=(4, 2),
        blocker_pos=(3, 2),
        blocker=("ball", "purple"),
        key_pos=(2, 1),
        agent_pos=(1, 1),
        room_width=9,
        room_height=5,
    )

    adapter = BabyAIPerceptionAdapter()
    planner = BabyAIActionAdapter(room_size=(9, 5))
    parser = BabyAIMissionParser()

    obs = env.gen_obs()
    for _ in range(4):
        adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)
        obs, r, term, trunc, info = env.step(int(MiniGridAction.LEFT))
    adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)

    goal = parser.parse(env.mission)

    # Closed-loop execution: agent reasons step-by-step
    success = False
    for step_num in range(60):
        adapter.ingest_observation(obs, known_agent_pos=env.agent_pos)
        act = planner.plan_next_action(adapter.graph, goal)
        obs, r, term, trunc, info = env.step(int(act))
        if term and r > 0.0:
            success = True
            break

    assert success, "Agent failed to unblock doorway and pick up target box"
