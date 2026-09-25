from __future__ import annotations

import logging
import os
import sys
from typing import Any

import numpy as np

# Ensure HBLLM Core is on sys.path
for root, dirs, _ in os.walk("/kaggle/input"):
    if "hbllm" in dirs:
        if root not in sys.path:
            sys.path.insert(0, root)
        break

# Local development fallback
core_path = "/Users/Dumith_Salinda/Projects/HBLLM/core"
if os.path.exists(core_path) and core_path not in sys.path:
    sys.path.insert(0, core_path)

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState

logging.getLogger("hbllm").setLevel(logging.ERROR)
logging.getLogger("arc_agi").setLevel(logging.ERROR)

from plugins.arc_agi_adapter.inductive_learner import InductiveHCIRAgent


class MyAgent(Agent):
    """The competitive ARC-AGI-3 Agent for Kaggle powered by HBLLM Core Cognitive Architecture."""

    MAX_ACTIONS = 1000

    def __init__(self, *args: Any, disable_archetypes: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.disable_archetypes = disable_archetypes
        self.internal_agent = InductiveHCIRAgent(disable_archetypes=disable_archetypes)
        self.last_grid: np.ndarray | None = None
        self.current_game_id: str | None = None
        self.current_levels_completed: int = 0

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        win_levels = getattr(latest_frame, "win_levels", 1) or 1
        return latest_frame.state is GameState.WIN or latest_frame.levels_completed >= win_levels

    def _extract_grid(self, latest_frame: Any, frames: Any = None) -> np.ndarray:
        if isinstance(latest_frame, np.ndarray):
            return latest_frame[-1] if latest_frame.ndim == 3 else latest_frame
        if hasattr(latest_frame, "frame"):
            f = latest_frame.frame
            if isinstance(f, np.ndarray):
                return f[-1] if f.ndim == 3 else f
            if isinstance(f, (list, tuple)) and len(f) > 0:
                if isinstance(f[-1], np.ndarray):
                    return f[-1]
                try:
                    return np.asarray(f[-1], dtype=int)
                except Exception:
                    pass
        if hasattr(latest_frame, "grid"):
            g = latest_frame.grid
            if isinstance(g, np.ndarray):
                return g
            if isinstance(g, (list, tuple)) and len(g) > 0:
                if isinstance(g[-1], np.ndarray):
                    return g[-1]
                try:
                    return np.asarray(g[-1], dtype=int)
                except Exception:
                    pass
        if hasattr(latest_frame, "image"):
            im = latest_frame.image
            if isinstance(im, np.ndarray):
                return im
        return np.zeros((64, 64), dtype=int)

    def _extract_available_actions(self, latest_frame: Any) -> list[int]:
        if hasattr(latest_frame, "available_actions"):
            raw = latest_frame.available_actions
            if isinstance(raw, (list, set, tuple)):
                acts = []
                for a in raw:
                    if hasattr(a, "value") and isinstance(a.value, int):
                        acts.append(a.value)
                    elif isinstance(a, int):
                        acts.append(a)
                if acts:
                    return sorted(list(set(acts)))
        return [1, 2, 3, 4, 5, 6, 7]

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        # Framework contract: First call or after a death -> reset the level
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self.internal_agent.reset_episode()
            self.last_grid = None
            return GameAction.RESET

        grid = self._extract_grid(latest_frame, frames)
        available_actions = self._extract_available_actions(latest_frame)

        # Detect level completion / transition
        lvl_completed = getattr(latest_frame, "levels_completed", 0)
        if lvl_completed > self.current_levels_completed:
            self.current_levels_completed = lvl_completed
            self.internal_agent.reset_episode(retain_dynamics=True)
            self.last_grid = None

        if self.last_grid is not None and self.last_grid.shape == grid.shape:
            diff_ratio = float(np.mean(self.last_grid != grid))
            if diff_ratio > 0.50:
                self.internal_agent.reset_episode(retain_dynamics=True)

        self.last_grid = grid.copy()

        # Update state and level
        state_obj = latest_frame.state
        self.internal_agent.last_frame_state = getattr(state_obj, "name", None) or str(state_obj)
        if hasattr(self.internal_agent, "current_level"):
            self.internal_agent.current_level = lvl_completed

        # Automatically hydrate game-specific knowledge if available
        gid = getattr(self, "game_id", None) or getattr(latest_frame, "game_id", None)
        if gid and isinstance(gid, str):
            base_gid = gid.split("-")[0].strip()
            if self.current_game_id != base_gid:
                self.current_game_id = base_gid
                import glob
                from pathlib import Path

                found_kdir = None
                for root_cand in [
                    Path.cwd(),
                    Path("/Users/Dumith_Salinda/Projects/HBLLM/core"),
                    Path("/kaggle/input/hbllm-kaggle-dataset"),
                ]:
                    kdir = root_cand / "data" / "cognitive_memory" / "arc_agi_3"
                    if (kdir / f"{base_gid}_knowledge_graph.json").exists():
                        found_kdir = kdir
                        break
                if found_kdir is None:
                    matches = glob.glob("/kaggle/input/**/arc_agi_3", recursive=True)
                    if matches:
                        found_kdir = Path(matches[0])
                if found_kdir and (found_kdir / f"{base_gid}_knowledge_graph.json").exists():
                    self.internal_agent.load_knowledge(found_kdir, game_id=base_gid)

        try:
            action_id, conf = self.internal_agent.plan_next_action(grid, available_actions)
            action_data = getattr(self.internal_agent, "last_action_data", None)
        except Exception:
            action_id = available_actions[0] if available_actions else 1
            action_data = None

        try:
            if hasattr(GameAction, f"ACTION{action_id}"):
                act_enum = getattr(GameAction, f"ACTION{action_id}")
            elif hasattr(GameAction, str(action_id)):
                act_enum = getattr(GameAction, str(action_id))
            else:
                act_enum = GameAction(action_id)
        except Exception:
            act_enum = GameAction.ACTION1

        # Complex action (ACTION6) coordinates
        if act_enum.is_complex() or action_id == 6:
            if (
                not isinstance(action_data, dict)
                or "x" not in action_data
                or "y" not in action_data
            ):
                H, W = grid.shape
                fallback_x, fallback_y = W // 2, H // 2
                if (
                    hasattr(self.internal_agent, "current_target_pos")
                    and self.internal_agent.current_target_pos is not None
                ):
                    fallback_y, fallback_x = (
                        int(round(self.internal_agent.current_target_pos[0])),
                        int(round(self.internal_agent.current_target_pos[1])),
                    )
                elif (
                    hasattr(self.internal_agent, "current_actor_pos")
                    and self.internal_agent.current_actor_pos is not None
                ):
                    fallback_y, fallback_x = (
                        int(round(self.internal_agent.current_actor_pos[0])),
                        int(round(self.internal_agent.current_actor_pos[1])),
                    )
                coords = {"x": max(0, min(W - 1, fallback_x)), "y": max(0, min(H - 1, fallback_y))}
            else:
                coords = {"x": int(action_data["x"]), "y": int(action_data["y"])}
            act_enum.set_data(coords)

        return act_enum
