from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────
# CRITICAL: Ensure HBLLM Core is importable BEFORE any other import.
# On Kaggle, the dataset zip extracts to /kaggle/input/<dataset-slug>/
# and we need the ROOT of the extracted zip on sys.path so that
# `from hbllm.xxx import yyy` and `from plugins.xxx import yyy` both work.
# ─────────────────────────────────────────────────────────────────────────

_hbllm_resolved = False
_hbllm_root = None

# 1. Fast explicit path check (covers 99% of Kaggle cases)
_KAGGLE_DATASET_CANDIDATES = [
    "/kaggle/input/hbllm-kaggle-dataset",
    "/kaggle/input/hbllm-core",
    "/kaggle/input/hbllm",
    "/kaggle/input/datasets/dumithrathnayaka/hbllm-kaggle-dataset",
]

for _cand in _KAGGLE_DATASET_CANDIDATES:
    if os.path.isdir(os.path.join(_cand, "hbllm")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        _hbllm_resolved = True
        _hbllm_root = Path(_cand)
        print(f"[HBLLM] Resolved core from: {_cand}", flush=True)
        break

# 2. Recursive fallback: walk /kaggle/input (limited depth to avoid slowness)
if not _hbllm_resolved:
    for _root, _dirs, _ in os.walk("/kaggle/input"):
        if _root.count(os.sep) - "/kaggle/input".count(os.sep) > 2:
            _dirs.clear()  # Don't descend too deep
            continue
        if "hbllm" in _dirs and "plugins" in _dirs:
            if _root not in sys.path:
                sys.path.insert(0, _root)
            _hbllm_resolved = True
            _hbllm_root = Path(_root)
            print(f"[HBLLM] Resolved core from walk: {_root}", flush=True)
            break

# 3. Local development fallback
try:
    _here = Path(__file__).resolve().parent
except Exception:
    _here = Path.cwd()

if not _hbllm_resolved:
    for _cand_local in [_here.parent, _here.parent.parent, Path.cwd()]:
        if (_cand_local / "hbllm").exists() and str(_cand_local) not in sys.path:
            sys.path.insert(0, str(_cand_local))
            _hbllm_resolved = True
            _hbllm_root = _cand_local
            break

# 4. Locate ARC-AGI-3-Agents framework if present
for _root, _dirs, _ in os.walk("/kaggle"):
    if _root.count(os.sep) > 5:
        _dirs.clear()
        continue
    if "ARC-AGI-3-Agents" in _dirs:
        _p = os.path.join(_root, "ARC-AGI-3-Agents")
        if _p not in sys.path:
            sys.path.insert(0, _p)
        break

try:
    from agents.agent import Agent
except ImportError:

    class Agent:  # type: ignore[no-redef]
        """Fallback base Agent class when running standalone outside framework."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


try:
    from arcengine import FrameData, GameAction, GameState
except ImportError:
    from enum import Enum

    class GameState(Enum):  # type: ignore[no-redef]
        NOT_PLAYED = "NOT_PLAYED"
        NOT_FINISHED = "NOT_FINISHED"
        GAME_OVER = "GAME_OVER"
        WIN = "WIN"

    class GameAction(Enum):  # type: ignore[no-redef]
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

        def is_complex(self) -> bool:
            return self.value == 6

        def set_data(self, data: Any) -> None:
            self.data = data

    class FrameData:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.state = GameState.NOT_PLAYED
            self.frame = np.zeros((64, 64), dtype=int)
            self.levels_completed = 0
            self.win_levels = 1
            self.available_actions = [1, 2, 3, 4, 5, 6, 7]


logging.getLogger("hbllm").setLevel(logging.ERROR)
logging.getLogger("arc_agi").setLevel(logging.ERROR)

from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine


class MyAgent(Agent):
    """The competitive ARC-AGI-3 Agent for Kaggle powered by HBLLM Core Cognitive Architecture."""

    MAX_ACTIONS = 1000

    def __init__(
        self,
        card_id: str = "default_card",
        game_id: str = "default_game",
        agent_name: str = "myagent",
        ROOT_URL: str = "http://gateway:8001",
        record: bool = False,
        arc_env: Any = None,
        *args: Any,
        disable_archetypes: bool = False,
        instructions: Any = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(
                card_id, game_id, agent_name, ROOT_URL, record, arc_env, *args, **kwargs
            )
        except Exception:
            try:
                super().__init__(*args, **kwargs)
            except Exception:
                pass
        self.card_id = card_id
        self.game_id = game_id
        self.agent_name = agent_name
        self.ROOT_URL = ROOT_URL
        self.record = record
        self.arc_env = arc_env
        self.disable_archetypes = disable_archetypes
        self.internal_engine = AutonomousEpistemicEngine(instructions=instructions)
        self.internal_agent = self.internal_engine
        self.last_grid: np.ndarray | None = None
        self.current_game_id: str | None = None
        self.current_levels_completed: int = 0

    def load_instructions(self, instructions: Any) -> None:
        """Load executive cognitive directives into the agent's internal epistemic engine."""
        if hasattr(self.internal_engine, "load_instructions"):
            self.internal_engine.load_instructions(instructions)

    def reset_episode(self, retain_dynamics: bool = False, is_new_level: bool = False) -> None:
        self.last_grid = None
        if hasattr(self.internal_agent, "reset_episode"):
            try:
                self.internal_agent.reset_episode(
                    retain_dynamics=retain_dynamics, is_new_level=is_new_level
                )
            except TypeError:
                self.internal_agent.reset_episode(retain_dynamics=retain_dynamics)
        if hasattr(self.internal_agent, "prev_grid"):
            self.internal_agent.prev_grid = None

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
        state_obj = getattr(latest_frame, "state", None)
        state_name = getattr(state_obj, "name", str(state_obj))

        # Explicit ARC contract: NOT_PLAYED or GAME_OVER resets
        if state_obj is GameState.NOT_PLAYED or state_name == "NOT_PLAYED":
            self.internal_agent.reset_episode(retain_dynamics=False)
            self.last_grid = None
            if hasattr(self.internal_agent, "prev_grid"):
                self.internal_agent.prev_grid = None
            return GameAction.RESET

        if state_obj is GameState.GAME_OVER or state_name == "GAME_OVER":
            grid = self._extract_grid(latest_frame, frames)
            avail = self._extract_available_actions(latest_frame)
            if (
                hasattr(self.internal_engine, "assimilate_feedback")
                and getattr(self.internal_engine, "prev_grid", None) is not None
            ):
                self.internal_engine.assimilate_feedback(grid, avail, is_lost=True)
            # Per competition semantics with ONLY_RESET_LEVELS=true,
            # reset retries the same level: retain learned dynamics and lethal memories
            self.reset_episode(retain_dynamics=True, is_new_level=False)
            return GameAction.RESET

        # Check full_reset signal from ARC-AGI environment
        if getattr(latest_frame, "full_reset", False):
            self.internal_agent.reset_episode(retain_dynamics=False)
            self.last_grid = None
            if hasattr(self.internal_agent, "prev_grid"):
                self.internal_agent.prev_grid = None

        # Game ID extraction and internal cognitive management BEFORE planning
        raw_gid = getattr(latest_frame, "game_id", None) or getattr(self, "game_id", None)
        if raw_gid in ("default_game", "", None):
            raw_gid = getattr(latest_frame, "game_id", None)
        if raw_gid and isinstance(raw_gid, str):
            base_gid = raw_gid.split("-")[0].strip()
            if self.current_game_id != base_gid:
                print(
                    f"[ARC] Switching game: {self.current_game_id} -> {base_gid} (raw={raw_gid})",
                    flush=True,
                )
                self.current_game_id = base_gid
                self.current_levels_completed = 0
                self.internal_agent.reset_episode(retain_dynamics=False)
                self.last_grid = None
                if hasattr(self.internal_agent, "prev_grid"):
                    self.internal_agent.prev_grid = None

        grid = self._extract_grid(latest_frame, frames)
        available_actions = self._extract_available_actions(latest_frame)

        # Detect level completion / transition or reset
        lvl_completed = getattr(latest_frame, "levels_completed", 0)
        if lvl_completed != self.current_levels_completed:
            if lvl_completed > self.current_levels_completed:
                print(
                    f"[ARC] Level completed: {self.current_levels_completed} -> {lvl_completed} (game={self.current_game_id})",
                    flush=True,
                )
                if (
                    hasattr(self.internal_engine, "assimilate_feedback")
                    and getattr(self.internal_engine, "prev_grid", None) is not None
                ):
                    self.internal_engine.assimilate_feedback(grid, available_actions, is_win=True)
            else:
                print(
                    f"[ARC] Game reset to level: {self.current_levels_completed} -> {lvl_completed} (game={self.current_game_id})",
                    flush=True,
                )
            self.current_levels_completed = lvl_completed
            self.reset_episode(retain_dynamics=True, is_new_level=True)

        self.last_grid = grid.copy()

        # Synchronize working level and frame state
        self.internal_agent.last_frame_state = state_name
        if hasattr(self.internal_agent, "current_level"):
            self.internal_agent.current_level = lvl_completed
        if hasattr(self.internal_agent, "spatial_cognitive_agent"):
            self.internal_agent.spatial_cognitive_agent.current_level = lvl_completed

        # Diagnostic telemetry per step
        step_idx = getattr(self.internal_agent, "step_counter", 0)
        if step_idx < 5 or step_idx % 10 == 0:
            print(
                f"[ARC] game={raw_gid} base={self.current_game_id} step={step_idx} "
                f"lvl={lvl_completed} state={state_name} actions={available_actions}",
                flush=True,
            )

        # Execute cognitive planning with resilient error recovery
        try:
            if hasattr(self.internal_engine, "decide"):
                _arc_schema = [
                    {"action_id": 1, "name": "MOVE_UP"},
                    {"action_id": 2, "name": "MOVE_DOWN"},
                    {"action_id": 3, "name": "MOVE_LEFT"},
                    {"action_id": 4, "name": "MOVE_RIGHT"},
                    {"action_id": 5, "name": "INTERACT"},
                    {"action_id": 6, "name": "CLICK_CELL", "parameters": {"x": "int", "y": "int"}},
                    {"action_id": 7, "name": "ACTION7", "parameters": {"x": "int", "y": "int"}},
                ]
                action_id, action_data = self.internal_engine.decide(
                    grid, available_actions, action_schemas=_arc_schema
                )
                conf = 1.0
            else:
                action_id, conf = self.internal_agent.plan_next_action(grid, available_actions)
                action_data = getattr(self.internal_agent, "last_action_data", None)
            if step_idx < 5 or step_idx % 10 == 0:
                print(
                    f"[ARC] -> action_id={action_id} conf={conf:.3f} data={action_data}", flush=True
                )
        except Exception as e:
            import traceback

            print(
                f"[HBLLM ERROR] plan_next_action failed game={raw_gid} base={self.current_game_id} "
                f"step={step_idx} lvl={lvl_completed} state={state_name}: {e}. Falling back to default available action.",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
            action_id = available_actions[0] if available_actions else 1
            action_data = None

        # Resolve GameAction enum
        try:
            if hasattr(GameAction, f"ACTION{action_id}"):
                act_enum = getattr(GameAction, f"ACTION{action_id}")
            elif hasattr(GameAction, "from_id"):
                act_enum = GameAction.from_id(int(action_id))
            elif hasattr(GameAction, str(action_id)):
                act_enum = getattr(GameAction, str(action_id))
            else:
                act_enum = GameAction[f"ACTION{action_id}"]
        except Exception as e:
            print(
                f"[HBLLM FATAL] Failed to resolve GameAction for action_id={action_id}: {e}",
                file=sys.stderr,
                flush=True,
            )
            raise

        # Parameterized / complex action coordinate attachment
        if (hasattr(act_enum, "is_complex") and act_enum.is_complex()) or (
            action_data
            and isinstance(action_data, dict)
            and "x" in action_data
            and "y" in action_data
        ):
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
            if hasattr(act_enum, "set_data"):
                act_enum.set_data(coords)
            act_enum.data = coords

        return act_enum
