from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

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
    "/kaggle/input/hbllm_kaggle_dataset",
    "/kaggle/input/hbllm-core",
    "/kaggle/input/hbllm",
    "/kaggle/input/datasets/dumithrathnayaka/hbllm-kaggle-dataset",
    "/kaggle/input/datasets/dumithsalinda/hbllm-kaggle-dataset",
]

for _cand in _KAGGLE_DATASET_CANDIDATES:
    if os.path.isdir(os.path.join(_cand, "hbllm")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        _hbllm_resolved = True
        _hbllm_root = Path(_cand)
        print(f"[HBLLM] Resolved core from: {_cand}", flush=True)
        break

# 2. Recursive fallback: walk /kaggle/input (allows nested Kaggle datasets)
if not _hbllm_resolved:
    for _root, _dirs, _ in os.walk("/kaggle/input"):
        if _root.count(os.sep) - "/kaggle/input".count(os.sep) > 5:
            _dirs.clear()  # Don't descend too deep
            continue
        if "hbllm" in _dirs and "plugins" in _dirs:
            if _root not in sys.path:
                sys.path.insert(0, _root)
            _hbllm_resolved = True
            _hbllm_root = Path(_root)
            print(f"[HBLLM] Resolved core from walk: {_root}", flush=True)
            break

# 2b. Automatic zip extraction fallback (if dataset contains zipped archive)
if not _hbllm_resolved:
    import zipfile

    for _root, _, _files in os.walk("/kaggle/input"):
        for _f in _files:
            if _f.endswith(".zip") and ("hbllm" in _f.lower() or "dataset" in _f.lower()):
                _zip_path = os.path.join(_root, _f)
                _target_extract = "/tmp/hbllm_unpacked"
                if not os.path.isdir(os.path.join(_target_extract, "hbllm")):
                    try:
                        print(f"[HBLLM] Unpacking {_zip_path} to {_target_extract}...", flush=True)
                        with zipfile.ZipFile(_zip_path, "r") as _zf:
                            _zf.extractall(_target_extract)
                    except Exception as _ze:
                        print(f"[HBLLM] Failed to unpack {_zip_path}: {_ze}", flush=True)
                if os.path.isdir(os.path.join(_target_extract, "hbllm")):
                    if _target_extract not in sys.path:
                        sys.path.insert(0, _target_extract)
                    _hbllm_resolved = True
                    _hbllm_root = Path(_target_extract)
                    print(f"[HBLLM] Resolved core from unpacked zip: {_target_extract}", flush=True)
                    break
        if _hbllm_resolved:
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

        def main(self) -> None:
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
from hbllm.hcir.world.cortex_perception import SensoryThalamus

try:
    from kaggle_submission.constants import ARC_DEFAULT_ACTION_SCHEMAS
except ImportError:
    try:
        from constants import ARC_DEFAULT_ACTION_SCHEMAS  # type: ignore[no-redef]
    except ImportError:
        ARC_DEFAULT_ACTION_SCHEMAS = [
            {"name": "MOVE_UP", "action_id": 1},
            {"name": "MOVE_DOWN", "action_id": 2},
            {"name": "MOVE_LEFT", "action_id": 3},
            {"name": "MOVE_RIGHT", "action_id": 4},
            {"name": "INTERACT", "action_id": 5},
            {"name": "CLICK_CELL", "action_id": 6, "parameters": {"x": "int", "y": "int"}},
            {"name": "ACTION7", "action_id": 7},
        ]


class MyAgent(Agent):
    """The competitive ARC-AGI-3 Agent for Kaggle powered by HBLLM Core Cognitive Architecture."""

    # ─────────────────────────────────────────────────────────────────────────
    # Tournament Lifecycle & Adaptive Time / Action Guards
    # Prevents 8-9h Kaggle container timeouts by strictly bounding compute:
    # 1. Global Tournament Time Guard (7.5h max, leaves 1.5h margin for scorecard close)
    # 2. Per-Game Time Guard (8 min max per game)
    # 3. Dynamic Progress-Scaled Action Guard (100 base + 75 per completed level)
    # 4. Consecutive Death Stagnation Guard (max 5 deaths on same level without progress)
    # 5. Worker Concurrency Gating (max 4 concurrent games to eliminate GIL thrashing)
    # ─────────────────────────────────────────────────────────────────────────
    MAX_ACTIONS: int = 800
    BASE_ACTIONS_PER_LEVEL: int = 100
    EXTRA_ACTIONS_PER_LEVEL: int = 75
    MAX_CONSECUTIVE_DEATHS: int = 5
    PER_GAME_MAX_SECONDS: float = 480.0  # 8 minutes max per game
    TOURNAMENT_MAX_SECONDS: float = (
        27000.0  # 7.5 hours (leaves 1.5h margin before Kaggle's 9h kill)
    )
    CONCURRENCY_LIMIT: int = 4  # Prevent GIL contention & memory thrashing

    _TOURNAMENT_START_TIME: float | None = None
    _CONCURRENCY_SEMAPHORE: threading.Semaphore | None = None

    @classmethod
    def is_tournament_budget_exceeded(cls) -> bool:
        if cls._TOURNAMENT_START_TIME is None:
            return False
        return (time.time() - cls._TOURNAMENT_START_TIME) >= cls.TOURNAMENT_MAX_SECONDS

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
        self.instructions = instructions
        self._seed = kwargs.get("seed")
        self.internal_engine = AutonomousEpistemicEngine(instructions=instructions, seed=self._seed)
        self.internal_agent = self.internal_engine
        self.engine = self.internal_engine
        self.last_grid: np.ndarray | None = None
        self._last_decided_action: int | None = None
        self._grid_before_action: np.ndarray | None = None
        self._action_was_available: bool = True
        self._step_limit: int | None = None
        self._contract_violations: int = 0
        self._actuation_rejected: bool = False
        self._decide_latency_ms: float | None = None
        self.decision_exceptions: int = 0
        self._yield_reason: str | None = None
        self.current_game_id: str | None = (
            game_id.split("-")[0].strip() if game_id not in ("default_game", "", None) else None
        )
        try:
            from plugins.arc_agi_adapter.trace import TraceRecorder

            self._trace = TraceRecorder.from_env(
                kwargs.get("trace_dir"),
                subdir=None if kwargs.get("trace_dir") else self.current_game_id,
            )
        except Exception:
            self._trace = None
        self.current_levels_completed: int = 0
        self.level_executed_actions: list[tuple[int, dict[str, Any] | None]] = []

        # Time & Stagnation Tracking per Game Instance
        self._game_start_time: float = time.time()
        self.consecutive_deaths_this_level: int = 0
        self._is_marked_done: bool = False
        self.action_counter: int = 0

        # Grid stagnation detection: bail out when agent makes no visible progress
        self._stagnation_grid_hash: int | None = None
        self._stagnation_steps: int = 0
        self._MAX_STAGNATION_STEPS: int = 20  # 20 steps with identical grid → bail out

    @property
    def current_level(self) -> int:
        return self.current_levels_completed

    @current_level.setter
    def current_level(self, value: int) -> None:
        self.current_levels_completed = value

    def main(self) -> None:
        """Execute agent with concurrency gating and tournament time tracking."""
        if MyAgent._TOURNAMENT_START_TIME is None:
            MyAgent._TOURNAMENT_START_TIME = time.time()

        if MyAgent._CONCURRENCY_SEMAPHORE is None:
            MyAgent._CONCURRENCY_SEMAPHORE = threading.Semaphore(self.CONCURRENCY_LIMIT)

        # Do not start if tournament deadline is already reached
        if MyAgent.is_tournament_budget_exceeded():
            print(
                f"[TIME GUARD] Skipping game {self.game_id}: global tournament deadline already reached.",
                flush=True,
            )
            return

        with MyAgent._CONCURRENCY_SEMAPHORE:
            self._game_start_time = time.time()
            try:
                super().main()
            except Exception as e:
                import traceback

                print(
                    f"[HBLLM ERROR] Exception in game {self.game_id}: {e}",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc(file=sys.stderr)

    def load_instructions(self, instructions: Any) -> None:
        """Load executive cognitive directives into the agent's internal epistemic engine."""
        self.instructions = instructions
        if hasattr(self.internal_engine, "load_instructions"):
            self.internal_engine.load_instructions(instructions)

    def reset_episode(
        self,
        retain_dynamics: bool = False,
        is_new_level: bool = False,
        level: int | None = None,
    ) -> None:
        self.last_grid = None
        self._grid_before_action = None
        if not retain_dynamics or not is_new_level:
            self.level_executed_actions.clear()
        if hasattr(self.internal_agent, "on_boundary"):
            from hbllm.hcir.world.percept import EpisodeBoundary, EpisodeBoundaryKind

            self.internal_agent.on_boundary(
                EpisodeBoundary(
                    kind=EpisodeBoundaryKind.LEVEL_COMPLETE
                    if is_new_level
                    else EpisodeBoundaryKind.DEATH,
                    retain_dynamics=retain_dynamics,
                )
            )
        elif hasattr(self.internal_agent, "reset_episode"):
            try:
                self.internal_agent.reset_episode(
                    retain_dynamics=retain_dynamics, is_new_level=is_new_level, level=level
                )
            except TypeError:
                try:
                    self.internal_agent.reset_episode(
                        retain_dynamics=retain_dynamics, is_new_level=is_new_level
                    )
                except TypeError:
                    self.internal_agent.reset_episode(retain_dynamics=retain_dynamics)
        if hasattr(self.internal_agent, "prev_grid"):
            self.internal_agent.prev_grid = None

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        if getattr(self, "_is_marked_done", False):
            return True

        state_obj = getattr(latest_frame, "state", None)
        state_name = getattr(state_obj, "name", str(state_obj))
        if state_name == "WIN" or state_obj is GameState.WIN or state_obj == GameState.WIN:
            self._yield_reason = "win"
            return True
        win_levels = getattr(latest_frame, "win_levels", None)
        levels_comp = getattr(latest_frame, "levels_completed", None)
        if win_levels is not None and levels_comp is not None and win_levels > 0:
            if levels_comp >= win_levels:
                self._yield_reason = "win"
                return True

        # 1. Global Tournament Time Guard
        if MyAgent.is_tournament_budget_exceeded():
            print(
                f"[TIME GUARD] Global tournament deadline reached. Ending {self.game_id} cleanly.",
                flush=True,
            )
            self._is_marked_done = True
            self._yield_reason = "tournament_time"
            return True

        # 2. Per-game Wall-Clock Time Guard
        game_start = getattr(self, "_game_start_time", None)
        if game_start is not None and (time.time() - game_start) > self.PER_GAME_MAX_SECONDS:
            print(
                f"[TIME GUARD] Per-game budget ({self.PER_GAME_MAX_SECONDS}s) exceeded for {self.game_id} "
                f"at level {self.current_levels_completed}. Yielding game.",
                flush=True,
            )
            self._is_marked_done = True
            self._yield_reason = "per_game_time"
            return True

        # 3. Dynamic Progress-Scaled Action Budget Guard
        allowed_actions = self.BASE_ACTIONS_PER_LEVEL + (
            self.current_levels_completed * self.EXTRA_ACTIONS_PER_LEVEL
        )
        act_cnt = getattr(self, "action_counter", 0)
        if act_cnt >= allowed_actions:
            print(
                f"[ACTION GUARD] Action budget reached ({act_cnt}/{allowed_actions}) for {self.game_id} "
                f"at level {self.current_levels_completed}. Yielding game.",
                flush=True,
            )
            self._is_marked_done = True
            self._yield_reason = "action_budget"
            return True

        # 4. Consecutive Death Stagnation Guard
        if getattr(self, "consecutive_deaths_this_level", 0) >= self.MAX_CONSECUTIVE_DEATHS:
            print(
                f"[STAGNATION GUARD] Max consecutive deaths ({self.MAX_CONSECUTIVE_DEATHS}) reached on level "
                f"{self.current_levels_completed} for {self.game_id}. Yielding game.",
                flush=True,
            )
            self._is_marked_done = True
            self._yield_reason = "consecutive_deaths"
            return True

        # 5. Grid Stagnation Guard: no observable change for _MAX_STAGNATION_STEPS
        if getattr(self, "_stagnation_steps", 0) >= getattr(self, "_MAX_STAGNATION_STEPS", 20):
            print(
                f"[STAGNATION GUARD] Grid unchanged for {self._stagnation_steps} steps on level "
                f"{self.current_levels_completed} for {self.game_id}. Yielding game.",
                flush=True,
            )
            self._is_marked_done = True
            self._yield_reason = "grid_stagnation"
            return True

        return False

    def _extract_grid(self, latest_frame: Any, frames: Any = None) -> np.ndarray:
        raw: Any = None
        if isinstance(latest_frame, np.ndarray):
            raw = latest_frame
        elif hasattr(latest_frame, "frame"):
            f = latest_frame.frame
            if isinstance(f, np.ndarray):
                raw = f
            elif isinstance(f, (list, tuple)) and len(f) > 0:
                if isinstance(f[-1], np.ndarray):
                    raw = f[-1]
                else:
                    try:
                        raw = np.asarray(f[-1], dtype=int)
                    except Exception:
                        raw = np.asarray(f, dtype=int)
            else:
                raw = np.asarray(f, dtype=int)
        elif hasattr(latest_frame, "grid"):
            g = latest_frame.grid
            if isinstance(g, np.ndarray):
                raw = g
            elif isinstance(g, (list, tuple)) and len(g) > 0:
                if isinstance(g[-1], np.ndarray):
                    raw = g[-1]
                else:
                    raw = np.asarray(g[-1], dtype=int)
            else:
                raw = np.asarray(g, dtype=int)
        elif hasattr(latest_frame, "image"):
            im = latest_frame.image
            raw = im if isinstance(im, np.ndarray) else np.asarray(im, dtype=int)
        else:
            raw = np.zeros((64, 64), dtype=int)

        raw_arr = np.asarray(raw)
        if raw_arr.ndim == 3 and len(raw_arr) > 0:
            raw_arr = raw_arr[-1]
        if raw_arr.ndim != 2:
            raise ValueError(
                f"extracted grid is not 2D (shape={raw_arr.shape}) — frame layers must be 2D"
            )
        return SensoryThalamus.normalize(raw_arr)

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
            self.consecutive_deaths_this_level += 1
            if self.consecutive_deaths_this_level >= self.MAX_CONSECUTIVE_DEATHS:
                self._is_marked_done = True
                self._yield_reason = "consecutive_deaths"
                print(
                    f"[STAGNATION GUARD] Game {self.current_game_id} reached {self.consecutive_deaths_this_level} consecutive "
                    f"deaths on level {self.current_levels_completed}. Ending game to conserve tournament time budget.",
                    flush=True,
                )
            grid = self._extract_grid(latest_frame, frames)
            avail = self._extract_available_actions(latest_frame)
            if (
                hasattr(self.internal_engine, "assimilate_feedback")
                and getattr(self.internal_engine, "prev_grid", None) is not None
            ):
                self.internal_engine.assimilate_feedback(grid, avail, is_failure=True)
            # Per competition semantics with ONLY_RESET_LEVELS=true,
            # reset retries the same level: retain learned dynamics and lethal memories
            self.reset_episode(
                retain_dynamics=True, is_new_level=False, level=self.current_levels_completed
            )
            return GameAction.RESET

        if state_obj is GameState.WIN or state_name == "WIN":
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
            curr_base = self.current_game_id.split("-")[0].strip() if self.current_game_id else None
            if curr_base != base_gid:
                print(
                    f"[ARC] Switching game: {self.current_game_id} -> {base_gid} (raw={raw_gid})",
                    flush=True,
                )
                self.current_game_id = base_gid
                self.current_levels_completed = 0
                self.action_counter = 0
                self._stagnation_grid_hash = None
                self._stagnation_steps = 0
                self.internal_engine = AutonomousEpistemicEngine(
                    instructions=self.instructions, seed=self._seed
                )
                self.internal_agent = self.internal_engine
                self.last_grid = None

        grid = self._extract_grid(latest_frame, frames)
        available_actions = self._extract_available_actions(latest_frame)

        # Detect level completion / transition or reset
        lvl_completed = getattr(latest_frame, "levels_completed", 0)
        if lvl_completed != self.current_levels_completed:
            if lvl_completed > self.current_levels_completed:
                self.consecutive_deaths_this_level = 0
                print(
                    f"[ARC] Level completed: {self.current_levels_completed} -> {lvl_completed} (game={self.current_game_id})",
                    flush=True,
                )
                if (
                    hasattr(self.internal_engine, "assimilate_feedback")
                    and getattr(self.internal_engine, "prev_grid", None) is not None
                ):
                    self.internal_engine.assimilate_feedback(
                        grid, available_actions, is_success=True
                    )

                # Consolidate winning dynamic trajectory into persistent procedural memory
                if self.level_executed_actions and self.current_game_id:
                    clean_gid = self.current_game_id.lower().split("-")[0].strip()
                    skill_name = f"{clean_gid}_lvl_{self.current_levels_completed}"
                    proc_steps = []
                    for aid, adata in self.level_executed_actions:
                        if aid == 6 or (isinstance(adata, dict) and "x" in adata and "y" in adata):
                            ax = int(adata["x"]) if adata and "x" in adata else 0
                            ay = int(adata["y"]) if adata and "y" in adata else 0
                            proc_steps.append(
                                {
                                    "type": "click_xy",
                                    "x": ax,
                                    "y": ay,
                                }
                            )
                            proc_steps.append(
                                {
                                    "type": "click_cell",
                                    "cell_coord": [ay, ax],
                                }
                            )
                        elif adata:
                            proc_steps.append(
                                {
                                    "type": "primitive_with_data",
                                    "action_id": int(aid),
                                    "data": adata,
                                }
                            )
                        else:
                            proc_steps.append({"type": "primitive", "action_id": int(aid)})

                    if proc_steps:
                        try:
                            from hbllm.memory.procedural import ProceduralMemory

                            # In-memory SQLite (:memory:) — absolutely zero .db files created on disk!
                            pm = ProceduralMemory(db_path=":memory:")
                            pm.store_skill_sync(
                                tenant_id="arc_agi",
                                skill_name=skill_name,
                                trigger_pattern=f"Dynamically learned winning sequence for {clean_gid} level {self.current_levels_completed}",
                                steps=proc_steps,
                                source_node="MyAgent.dynamic_consolidation",
                            )
                            print(
                                f"[ARC] Procedural skill dynamically consolidated: {skill_name} ({len(proc_steps)} steps)",
                                flush=True,
                            )
                        except Exception as _e:
                            logger.debug("Procedural consolidation error: %s", _e)
                self.level_executed_actions.clear()
            else:
                print(
                    f"[ARC] Game reset to level: {self.current_levels_completed} -> {lvl_completed} (game={self.current_game_id})",
                    flush=True,
                )
                self.level_executed_actions.clear()
            self.current_levels_completed = lvl_completed
            if hasattr(self.internal_agent, "current_level"):
                self.internal_agent.current_level = lvl_completed
            self.reset_episode(retain_dynamics=True, is_new_level=True, level=lvl_completed)

        self.last_grid = grid.copy()

        # Increment action counter for budget guard
        self.action_counter += 1

        # Grid stagnation tracking: detect when the observable state is not changing.
        # Hash the interior only — an outermost 1-cell border may be a HUD /
        # step-counter bar that ticks every step and would mask real stagnation.
        _interior = grid[1:-1, 1:-1] if grid.ndim == 2 else grid
        grid_hash = hash((_interior if _interior.size > 0 else grid).tobytes())
        if grid_hash == self._stagnation_grid_hash:
            self._stagnation_steps += 1
        else:
            self._stagnation_grid_hash = grid_hash
            self._stagnation_steps = 0

        # Synchronize working level and frame state
        self.internal_agent.last_frame_state = state_name
        if hasattr(self.internal_agent, "current_level"):
            self.internal_agent.current_level = lvl_completed
        if hasattr(self.internal_agent, "spatial_cognitive_agent"):
            self.internal_agent.spatial_cognitive_agent.current_level = lvl_completed

        # Synchronize environmental step limit if present in frame
        sc = getattr(latest_frame, "step_counter", None) or getattr(
            latest_frame, "step_limit", None
        )
        if sc is None and hasattr(latest_frame, "get"):
            try:
                sc = latest_frame.get("StepCounter")
            except Exception:
                pass
        if sc is not None:
            self._step_limit = int(sc)

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
                _arc_schema = getattr(self, "action_schemas", None)
                if _arc_schema is None:
                    g = getattr(getattr(self, "arc_env", None), "_game", None) or getattr(
                        getattr(self, "arc_env", None), "game", None
                    )
                    v_targets = []
                    if g and hasattr(g, "actions") and g.actions:
                        v_targets = [
                            getattr(a, "data", None)
                            for a in g.actions
                            if isinstance(getattr(a, "data", None), dict)
                            and getattr(a, "data", None)
                        ]
                    _arc_schema = [dict(s) for s in ARC_DEFAULT_ACTION_SCHEMAS]
                    if v_targets:
                        for s in _arc_schema:
                            if s.get("action_id") == 6:
                                s["valid_targets"] = v_targets
                g_ctx = getattr(getattr(self, "arc_env", None), "_game", None) or getattr(
                    getattr(self, "arc_env", None), "game", None
                )
                _objects = None
                try:
                    from plugins.arc_agi_adapter.arc_perception import extract_arc_objects

                    _objects = extract_arc_objects(
                        grid,
                        engine=self.internal_engine,
                        game_obj=g_ctx or latest_frame,
                    )
                except Exception as _e:
                    logger.debug("extract_arc_objects failed: %s", _e)

                from hbllm.hcir.world.percept import (
                    ActuatorSchema,
                    EmbodiedObservation,
                    TerminalSignal,
                )
                from plugins.arc_agi_adapter.action_contract import (
                    ActionContract,
                    EffectKind,
                )

                # C3: was the last actuation rejected by the env? (zero-motion
                # sample must not teach dynamics — flagged into metadata)
                if self._grid_before_action is not None and self._last_decided_action is not None:
                    _eff = ActionContract.reconcile(
                        self._grid_before_action,
                        grid,
                        self._last_decided_action,
                        action_was_available=self._action_was_available,
                    )
                    self._actuation_rejected = _eff.kind == EffectKind.ENV_REJECTED

                is_win = bool(
                    getattr(latest_frame, "is_win", False)
                    or state_name == "WIN"
                    or state_obj is GameState.WIN
                )
                is_lost = bool(
                    getattr(latest_frame, "is_lost", False)
                    or state_name == "GAME_OVER"
                    or state_obj is GameState.GAME_OVER
                )
                _actuators = [
                    ActuatorSchema(
                        actuator_id=str(s.get("action_id", i)),
                        kind="point" if s.get("action_id") == 6 else "discrete",
                        params={
                            k: v for k, v in s.items() if k not in ("action_id", "valid_targets")
                        },
                        valid_targets=s.get("valid_targets"),
                    )
                    for i, s in enumerate(_arc_schema or [])
                ]
                _meta: dict[str, Any] = {}
                if self._step_limit is not None:
                    _meta["step_limit"] = self._step_limit
                if self._actuation_rejected:
                    _meta["actuation_rejected"] = True
                obs = EmbodiedObservation(
                    lattice=grid,
                    entities=_objects or [],
                    available_actions=available_actions,
                    actuators=_actuators,
                    terminal=TerminalSignal.SUCCESS
                    if is_win
                    else (TerminalSignal.FAILURE if is_lost else None),
                    context_id=str(
                        self.game_id or self.current_game_id or raw_gid or "default_context"
                    ),
                    episode_id=str(self.current_levels_completed),
                    metadata=_meta,
                )
                _t0 = __import__("time").perf_counter()
                action_id, action_data = self.internal_engine.step(obs)
                self._decide_latency_ms = (__import__("time").perf_counter() - _t0) * 1000.0
                conf = 1.0
                # WP1: no-proposal sentinel (pipeline commit rejected) -> deterministic
                # nearest-legal fallback, surfaced via the action contract repair path.
                if action_id is None:
                    self._contract_violations += 1
                    self._last_violation = "pipeline_rejected"
                    action_id = sorted(available_actions)[0] if available_actions else 0
                    action_data = None
                # C3: validate chosen action; repair to nearest legal on violation
                _res = ActionContract.validate_or_repair(
                    int(action_id),
                    action_data,
                    list(available_actions),
                    (int(grid.shape[0]), int(grid.shape[1])),
                )
                if not _res.ok:
                    self._contract_violations += 1
                    self._last_violation = _res.violation
                    if _res.repaired_action_id is not None:
                        action_id, action_data = _res.repaired_action_id, None
            else:
                action_id, conf = self.internal_agent.plan_next_action(grid, available_actions)
                action_data = getattr(self.internal_agent, "last_action_data", None)
            if step_idx < 5 or step_idx % 10 == 0:
                print(
                    f"[ARC] -> action_id={action_id} conf={conf:.3f} data={action_data}", flush=True
                )
        except Exception as e:
            import traceback

            self.decision_exceptions = getattr(self, "decision_exceptions", 0) + 1
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
                coords = {"x": max(0, int(action_data["x"])), "y": max(0, int(action_data["y"]))}
            if hasattr(act_enum, "set_data"):
                act_enum.set_data(coords)
            act_enum.data = coords

        self._last_decided_action = action_id
        self._grid_before_action = np.asarray(grid).copy()
        self._action_was_available = action_id in list(available_actions) or action_id == 0
        self.last_grid = np.asarray(grid).copy()
        if getattr(self, "_trace", None) is not None:
            try:
                self._trace.record_step(
                    game=str(self.current_game_id or raw_gid),
                    level=int(lvl_completed),
                    step=int(step_idx),
                    grid=grid,
                    available_actions=list(available_actions),
                    action_id=int(action_id),
                    action_data=action_data if isinstance(action_data, dict) else None,
                    observed_effect=None,
                    levels_completed=int(lvl_completed),
                    state=state_name,
                    latency_ms=float(getattr(self, "_decide_latency_ms", 0.0)),
                    contract_violation=getattr(self, "_last_violation", None),
                    branch=getattr(self.internal_engine, "_last_decision_branch", None),
                )
                self._last_violation = None
            except Exception:
                pass

        # Record executed action for episodic procedural consolidation
        act_data_rec = (
            coords
            if (
                (hasattr(act_enum, "is_complex") and act_enum.is_complex())
                or (action_data and isinstance(action_data, dict))
            )
            else None
        )
        self.level_executed_actions.append((int(action_id), act_data_rec))

        return act_enum
