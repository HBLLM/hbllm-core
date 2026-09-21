"""Unit and regression tests for ARC3SpatialCognitiveAgent and Driver Management Layer."""

from arc_agi import Arcade
from arcengine import GameAction

from hbllm.drivers import BaseDriver, DriverManager
from plugins.arc_agi_adapter.arc_driver import ArcadeDriver
from plugins.arc_agi_adapter.arc_spatial_agent import ARC3SpatialCognitiveAgent


def test_driver_management_registration() -> None:
    """Verify ArcadeDriver registers with DriverManager."""
    manager = DriverManager()
    driver = ArcadeDriver()
    assert isinstance(driver, BaseDriver)
    manager.register(driver)
    retrieved = manager.get_driver("arcade_driver")
    assert retrieved is driver
    assert manager.get_driver("arcade_driver").name == "arcade_driver"


def test_arc_spatial_agent_wa30_all_levels() -> None:
    """Verify ARC3SpatialCognitiveAgent solves wa30 across all 3 levels."""
    client = Arcade()
    env = client.make("wa30", render_mode=None)
    fd = env.reset()
    agent = ARC3SpatialCognitiveAgent()

    for lvl in range(3):
        completed = False
        for s in range(110):
            if getattr(fd, "levels_completed", 0) > lvl:
                completed = True
                break
            if not hasattr(fd, "frame") or len(fd.frame) == 0:
                break
            avail = getattr(fd, "available_actions", [1, 2, 3, 4, 5])
            curr_grid = fd.frame[0]
            act, conf = agent.plan_next_action(curr_grid, avail)
            prev_grid = curr_grid
            fd = env.step(getattr(GameAction, f"ACTION{act}"))
            curr_grid = fd.frame[0] if len(fd.frame) > 0 else prev_grid
            agent.update_causal_dynamics(act, prev_grid, curr_grid)

        if not completed and getattr(fd, "levels_completed", 0) > lvl:
            completed = True
        assert completed, f"Level {lvl} failed to complete within step budget"
        agent.reset_episode(retain_dynamics=True)
        try:
            fd = env.step(GameAction.ACTION5)
        except Exception:
            pass

    assert getattr(fd, "levels_completed", 0) == 3
