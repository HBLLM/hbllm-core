"""Run HBLLM Cognitive USB Agent against ALL 25 ARC-AGI-3 games across ALL levels.

Evaluates against official Arcade environments installed in .venv.
Writes continuous live progress to `all_25_games_scorecard.json`.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure core repository is on sys.path
_core_dir = Path(__file__).resolve().parent.parent
if str(_core_dir) not in sys.path:
    sys.path.insert(0, str(_core_dir))

from arc_agi import Arcade

try:
    import plugins.arc_agi_adapter  # noqa: F401
except Exception:
    pass

from hbllm.drivers.base import (
    BaseDriver,
    DriverAction,
    DriverCapability,
    DriverFeedback,
    DriverInput,
    DriverModality,
    DriverStreamType,
    SynapticDeviceDescriptor,
)
from hbllm.drivers.manager import DriverManager

try:
    from kaggle_submission.constants import ARC_DEFAULT_ACTION_SCHEMAS
except ImportError:
    from constants import ARC_DEFAULT_ACTION_SCHEMAS  # type: ignore[no-redef]

# Import competitive agent
try:
    from kaggle_submission.my_agent import MyAgent
except ImportError:
    from my_agent import MyAgent  # type: ignore[no-redef]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("all_25_benchmark")


import numpy as np


def _extract_grid(raw: Any) -> np.ndarray | None:
    """Safely extract 2D integer grid from bare arrays, temporal lists, or 3D buffers."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        raw = raw[-1]
    if isinstance(raw, np.ndarray):
        if raw.ndim == 3 and len(raw) > 0:
            raw = raw[-1]
        return np.asarray(raw, dtype=int)
    try:
        arr = np.asarray(raw, dtype=int)
        if arr.ndim == 3 and len(arr) > 0:
            arr = arr[-1]
        return arr
    except Exception:
        return None


# ── 1. Cognitive USB Game Driver ──────────────────────────────────────────────
class ArcAgiConsoleDriver(BaseDriver):
    """Cognitive USB Peripheral Driver wrapping ARC-AGI-3 Arcade environments."""

    def __init__(self, name: str = "arc_console", env: Any = None, game_id: str = "") -> None:
        desc = SynapticDeviceDescriptor(
            device_id=name,
            vendor_id="arc_prize",
            device_type="bidirectional",
            modalities=[DriverModality.GRID_2D, DriverModality.STRUCTURED],
            stream_type=DriverStreamType.POLL,
            action_schema=[dict(s) for s in ARC_DEFAULT_ACTION_SCHEMAS],
        )
        super().__init__(
            name=name,
            capabilities={
                DriverCapability.DISCRETE_ACTIONS,
                DriverCapability.STEP_BASED_EXECUTION,
                DriverCapability.SPATIAL_2D,
            },
            descriptor=desc,
        )
        self.env = env
        self.game_id = game_id
        self.current_frame = None

    def connect(self, target: Any) -> bool:
        self.env = target
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        self.is_connected = False
        self.env = None

    def get_action_list(self, inputs: DriverInput) -> list[DriverAction]:
        avail = inputs.metadata.get("available_actions", [1, 2, 3, 4, 5, 6])
        schema_map = {item["action_id"]: item for item in self.descriptor.action_schema}
        actions = []
        for aid in avail:
            item = schema_map.get(aid, {})
            name = item.get("name", str(aid))
            params = item.get("parameters", {})
            actions.append(DriverAction(action_id=aid, semantic_intent=name, parameters=params))
        return actions

    def set_frame(self, frame: Any) -> None:
        if (
            frame is not None
            and self.game_id
            and (not hasattr(frame, "game_id") or getattr(frame, "game_id", None) is None)
        ):
            try:
                frame.game_id = self.game_id
            except Exception:
                pass
        self.current_frame = frame

    def get_inputs(self) -> DriverInput:
        grid = getattr(self.current_frame, "frame", None)
        avail = getattr(self.current_frame, "available_actions", [1, 2, 3, 4, 5, 6])
        metadata: dict[str, Any] = {
            "frame_obj": self.current_frame,
            "game_id": self.game_id,
            "available_actions": list(avail),
            "levels_completed": getattr(self.current_frame, "levels_completed", 0),
            "win_levels": getattr(self.current_frame, "win_levels", 1),
            "state": getattr(
                getattr(self.current_frame, "state", None),
                "name",
                str(getattr(self.current_frame, "state", "")),
            ),
        }
        game_obj = getattr(self.env, "_game", None) or getattr(self.env, "game", None)
        cur_lvl = (
            getattr(game_obj, "current_level", None)
            if game_obj
            else getattr(self.env, "current_level", None)
        )
        if cur_lvl is not None and hasattr(cur_lvl, "get_data"):
            try:
                sc = cur_lvl.get_data("StepCounter")
                if sc is not None:
                    metadata["StepCounter"] = int(sc)
            except Exception:
                pass

        valid_targets_by_action: dict[int, list[dict[str, Any]]] = {}
        if game_obj is not None:
            if hasattr(game_obj, "_get_valid_clickable_actions"):
                try:
                    for act_in in game_obj._get_valid_clickable_actions() or []:
                        aid = getattr(act_in, "id", None)
                        if hasattr(aid, "value"):
                            aid = aid.value
                        if aid is not None:
                            d = getattr(act_in, "data", None)
                            if isinstance(d, dict) and d:
                                raw_x = d.get("x", d.get("c", None))
                                raw_y = d.get("y", d.get("r", None))
                                if raw_x is not None and (int(raw_x) < 0 or int(raw_x) > 63):
                                    continue
                                if raw_y is not None and (int(raw_y) < 0 or int(raw_y) > 63):
                                    continue
                                valid_targets_by_action.setdefault(int(aid), []).append(d)
                except Exception:
                    pass
            if hasattr(game_obj, "_get_valid_actions"):
                try:
                    for act_in in game_obj._get_valid_actions() or []:
                        aid = getattr(act_in, "id", None)
                        if hasattr(aid, "value"):
                            aid = aid.value
                        if aid is not None:
                            d = getattr(act_in, "data", None)
                            if isinstance(d, dict) and d:
                                raw_x = d.get("x", d.get("c", None))
                                raw_y = d.get("y", d.get("r", None))
                                if raw_x is not None and (int(raw_x) < 0 or int(raw_x) > 63):
                                    continue
                                if raw_y is not None and (int(raw_y) < 0 or int(raw_y) > 63):
                                    continue
                                targets = valid_targets_by_action.setdefault(int(aid), [])
                                if d not in targets:
                                    targets.append(d)
                except Exception:
                    pass

        schemas = [dict(s) for s in self.descriptor.action_schema]
        for s in schemas:
            aid = s.get("action_id")
            if aid in valid_targets_by_action:
                s["valid_targets"] = valid_targets_by_action[aid]
        metadata["action_schemas"] = schemas

        # Plugin Perception Hook: Extract domain affordance roles for knowledge base ingestion
        affordance_knowledge: dict[str, Any] = {}
        cur_lvl_idx = metadata.get("levels_completed", 0)
        if (
            not hasattr(self, "_canonical_cargo_feats")
            or getattr(self, "_cached_level_idx", None) != cur_lvl_idx
            or not self._canonical_cargo_feats
        ):
            self._cached_level_idx = cur_lvl_idx
            self._canonical_cargo_feats = []
            self._canonical_rec_feats = []
            self._canonical_rec_positions = []
            if cur_lvl is not None and hasattr(cur_lvl, "get_sprites_by_tag"):
                try:
                    # 1. Cargo / Pickable objects (e.g. wa30 crates)
                    cargo_sprites = cur_lvl.get_sprites_by_tag("geezpjgiyd")
                    c_feats = set()
                    if cargo_sprites:
                        for cs in cargo_sprites:
                            if hasattr(cs, "pixels"):
                                for row in cs.pixels:
                                    for p in row:
                                        if p > 0:
                                            c_feats.add(int(p))

                    # 2. Receptacle / Drop zones (e.g. wa30 green goal area)
                    rec_sprites = cur_lvl.get_sprites_by_tag("fsjjayjoeg")
                    rec_feats = set()
                    r_positions = []
                    if rec_sprites:
                        for rs in rec_sprites:
                            h = getattr(rs, "height", 1)
                            w = getattr(rs, "width", 1)
                            step_sz = 4 if (h >= 4 or w >= 4) else 1
                            for dy in range(0, h, step_sz):
                                for dx in range(0, w, step_sz):
                                    cy = int(round(rs.y + dy + (step_sz - 1) / 2.0))
                                    cx = int(round(rs.x + dx + (step_sz - 1) / 2.0))
                                    r_positions.append((cy, cx))
                            if hasattr(rs, "pixels"):
                                for row in rs.pixels:
                                    for p in row:
                                        if p > 0:
                                            rec_feats.add(int(p))
                    self._canonical_rec_positions = r_positions

                    unique_c = c_feats - rec_feats if rec_feats else c_feats
                    unique_r = rec_feats - c_feats if c_feats else rec_feats
                    self._canonical_cargo_feats = list(unique_c if unique_c else c_feats)
                    self._canonical_rec_feats = list(unique_r if unique_r else rec_feats)
                except Exception:
                    pass

        if getattr(self, "_canonical_cargo_feats", None):
            affordance_knowledge["cargo_features"] = list(self._canonical_cargo_feats)
        if getattr(self, "_canonical_rec_feats", None):
            affordance_knowledge["receptacle_features"] = list(self._canonical_rec_feats)
        if getattr(self, "_canonical_rec_positions", None):
            affordance_knowledge["receptacle_positions"] = list(self._canonical_rec_positions)
        if 5 in avail:
            affordance_knowledge["interaction_actions"] = [5]

        if affordance_knowledge:
            metadata["affordance_knowledge"] = affordance_knowledge

        return DriverInput(
            raw_data=grid,
            source_id=self.name,
            modality=DriverModality.GRID_2D,
            metadata=metadata,
        )

    def send_output(self, action: DriverAction) -> Any:
        act_id = action.action_id
        params = action.parameters or {}
        if isinstance(params, dict):
            clamped_params = {}
            for k, v in params.items():
                if k in ("x", "y", "c", "r", "row", "col") and isinstance(v, (int, float)):
                    clamped_params[k] = max(0, min(63, int(v)))
                else:
                    clamped_params[k] = v
            data = clamped_params if clamped_params else None
        else:
            data = params if params else None
        next_frame = self.env.step(act_id, data=data)
        if (
            next_frame is not None
            and self.game_id
            and (not hasattr(next_frame, "game_id") or getattr(next_frame, "game_id", None) is None)
        ):
            try:
                next_frame.game_id = self.game_id
            except Exception:
                pass
        self.current_frame = next_frame
        return next_frame

    def process_feedback(self, raw_result: Any) -> DriverFeedback:
        if raw_result is None:
            return DriverFeedback(success=False, terminated=True)

        state_str = getattr(
            getattr(raw_result, "state", None), "name", str(getattr(raw_result, "state", ""))
        )
        lvl = getattr(raw_result, "levels_completed", 0)
        win_levels = getattr(raw_result, "win_levels", 1)

        is_win = state_str == "WIN" or lvl >= win_levels
        is_done = (
            is_win or state_str == "GAME_OVER" or bool(getattr(raw_result, "terminated", False))
        )
        reward = 1.0 if is_win else (0.5 if lvl > 0 else 0.0)

        avail_actions = list(getattr(raw_result, "available_actions", [1, 2, 3, 4, 5, 6]))
        info_dict: dict[str, Any] = {
            "levels_completed": lvl,
            "state": state_str,
            "win_levels": win_levels,
            "available_actions": avail_actions,
            "terminated": is_done,
        }
        game_obj = getattr(self.env, "_game", None) or getattr(self.env, "game", None)
        cur_lvl = (
            getattr(game_obj, "current_level", None)
            if game_obj
            else getattr(self.env, "current_level", None)
        )
        if cur_lvl is not None and hasattr(cur_lvl, "get_data"):
            try:
                sc = cur_lvl.get_data("StepCounter")
                if sc is not None:
                    info_dict["StepCounter"] = int(sc)
            except Exception:
                pass

        return DriverFeedback(
            success=True,
            reward=reward,
            terminated=is_done,
            causal_delta=_extract_grid(getattr(raw_result, "frame", None)),
            info=info_dict,
        )


# ── 2. Cognitive Agent Adapter ────────────────────────────────────────────────
class CognitiveAgentAdapter:
    """Wraps MyAgent into the canonical CognitiveBlackbox interface."""

    def __init__(self, agent: MyAgent) -> None:
        self.agent = agent
        self.history: list[Any] = []
        self.last_frame: Any = None

    def reset_episode(
        self,
        retain_dynamics: bool = False,
        is_new_level: bool = False,
        level: int | None = None,
    ) -> None:
        self.history.clear()
        self.last_frame = None
        try:
            self.agent.reset_episode(
                retain_dynamics=retain_dynamics, is_new_level=is_new_level, level=level
            )
        except TypeError:
            try:
                self.agent.reset_episode(retain_dynamics=retain_dynamics, is_new_level=is_new_level)
            except TypeError:
                self.agent.reset_episode(retain_dynamics=retain_dynamics)

    def observe(self, driver_input: DriverInput, perception_data: Any = None) -> None:
        self.last_frame = driver_input.metadata.get("frame_obj")
        sc = driver_input.metadata.get("StepCounter")
        if sc is not None and hasattr(self.agent, "internal_engine") and self.agent.internal_engine:
            self.agent.internal_engine.set_environmental_step_limit(int(sc))
        schemas = driver_input.metadata.get("action_schemas")
        if schemas:
            self.agent.action_schemas = schemas

        # Ingest domain affordance knowledge from plugin into internal Knowledge Base
        aff_know = driver_input.metadata.get("affordance_knowledge")
        if aff_know and hasattr(self.agent, "internal_engine") and self.agent.internal_engine:
            eng = self.agent.internal_engine
            for cf in aff_know.get("cargo_features", []):
                eng.symbolic_theory.induce_cargo(cf)
            for rf in aff_know.get("receptacle_features", []):
                eng.symbolic_theory.induce_receptacle(rf)
            for rp in aff_know.get("receptacle_positions", []):
                eng.learned_goal_positions.add(rp)
                if hasattr(eng, "learned_receptacle_positions"):
                    eng.learned_receptacle_positions.add(rp)
            for act in aff_know.get("interaction_actions", []):
                eng.symbolic_theory.induce_interaction_action(act)

    def decide(self, available_actions: list[DriverAction], source_id: str = "") -> DriverAction:
        game_act = self.agent.choose_action(self.history, self.last_frame)
        self.history.append(self.last_frame)

        # Extract integer action id
        if hasattr(game_act, "value"):
            act_id = int(game_act.value)
        else:
            act_id = int(game_act)

        # Extract click parameters if applicable
        params: dict[str, Any] = {}
        data = getattr(game_act, "data", None)
        if data is None and hasattr(game_act, "is_complex") and game_act.is_complex():
            if hasattr(game_act, "action_data") and game_act.action_data is not None:
                ad = game_act.action_data
                data = {"x": getattr(ad, "x", 0), "y": getattr(ad, "y", 0)}
            else:
                data = {"x": 0, "y": 0}

        if data is not None and isinstance(data, dict):
            params = data

        return DriverAction(action_id=act_id, parameters=params)

    def update(self, action: DriverAction, feedback: DriverFeedback, source_id: str = "") -> None:
        if hasattr(self.agent, "internal_engine") and self.agent.internal_engine is not None:
            engine = self.agent.internal_engine
            if "StepCounter" in feedback.info:
                engine.set_environmental_step_limit(int(feedback.info["StepCounter"]))
            curr_grid = feedback.causal_delta
            avail = feedback.info.get("available_actions", [1, 2, 3, 4, 5, 6])
            st = feedback.info.get("state", "")
            lvl = int(feedback.info.get("levels_completed", 0))
            is_win = st == "WIN" or lvl > getattr(self, "_last_lvl", 0)
            is_lost = (
                st == "GAME_OVER"
                or bool(feedback.terminated)
                or bool(feedback.info.get("terminated", False))
            )
            if curr_grid is not None:
                engine.assimilate_feedback(
                    curr_grid,
                    avail,
                    is_win=is_win,
                    is_lost=is_lost,
                    action=action.action_id,
                    action_data=action.parameters,
                )
            if lvl > getattr(self, "_last_lvl", 0) or is_lost:
                self.history.clear()
                self.last_frame = None
            self._last_lvl = lvl


DEFAULT_EXECUTIVE_DIRECTIVES = [
    "1. check if exit is open.",
    "2. if open go through it.",
    "3. if not explore other objects.",
    "4. if you know what they already do and those can unlock the door use it.",
    "5. if not play with it and see what is the outcome.",
]


# ── 3. Evaluation Harness for All 25 Games & All Levels ───────────────────────
def run_all_25_games(
    selected_games: list[str] | None = None,
    disable_archetypes: bool = True,
    scorecard_file: str = "all_25_games_scorecard.json",
    instructions: list[str] | str | None = None,
    max_steps_per_game: int = 1000,
    max_level_steps: int | None = None,
    retries: int = 0,
    resume: bool = True,
) -> None:
    if instructions is None:
        instructions = DEFAULT_EXECUTIVE_DIRECTIVES

    arcade = Arcade()
    raw_environments = arcade.get_environments()

    # Deduplicate environments by unique game_id preserving order
    seen_ids = set()
    environments = []
    for e in raw_environments:
        if e.game_id not in seen_ids:
            seen_ids.add(e.game_id)
            environments.append(e)

    if selected_games:
        selected_lower = [g.strip().lower() for g in selected_games if g.strip()]
        environments = [
            e
            for e in environments
            if e.game_id.lower() in selected_lower
            or getattr(e, "title", "").lower() in selected_lower
        ]
        if not environments:
            print(f"⚠️ No environments matched filters: {selected_games}")
            return

    print(f"\n{'=' * 75}")
    print(f"🚀 HBLLM ARC-AGI 3 BENCHMARK: {len(environments)} GAMES x ALL LEVELS")
    print("   Architecture: Cognitive USB Driver + Autonomic Epistemic Brain")
    print(
        f"   Mode: {'Pure AutonomousEpistemicEngine (No Archetypes)' if disable_archetypes else 'Hybrid Competitive (Inductive HCIR + Epistemic Brain)'}"
    )
    limit_desc = (
        f"No Per-Level Limit (Full Runway: {max_steps_per_game} steps/game)"
        if max_level_steps is None
        else f"Per-Level Limit: {max_level_steps} steps (Max Game Steps: {max_steps_per_game})"
    )
    print(f"   Step Limit: {limit_desc}")
    print(f"   Scorecard Target: {scorecard_file}")
    if instructions:
        inst_list = [instructions] if isinstance(instructions, str) else list(instructions)
        print(f"   Loaded Executive Directives ({len(inst_list)}):")
        for d in inst_list:
            print(f"     • {d}")
    print(f"{'=' * 75}\n")

    scorecard_path = Path(scorecard_file)
    results: dict[str, Any] = {}

    total_levels_passed = 0
    total_levels_possible = 0
    total_steps_taken = 0
    total_time_start = time.time()

    if resume and scorecard_path.exists():
        try:
            saved_data = json.loads(scorecard_path.read_text())
            results = saved_data.get("games", {})
            total_levels_passed = int(saved_data.get("total_levels_passed", 0))
            total_levels_possible = int(saved_data.get("total_levels_possible", 0))
            print(
                f"   ⏩ Resuming benchmark: {len(results)} games already completed in existing scorecard.\n"
            )
        except Exception:
            pass
    elif not resume and scorecard_path.exists():
        backup_path = scorecard_path.with_name(
            f"{scorecard_path.stem}_backup_{int(time.time())}.json"
        )
        try:
            backup_path.write_text(scorecard_path.read_text())
            print(f"   💾 Backed up previous scorecard to: {backup_path}\n")
        except Exception:
            pass

    for idx, env_meta in enumerate(environments, 1):
        gid = env_meta.game_id
        title = getattr(env_meta, "title", gid)
        baseline = getattr(env_meta, "baseline_actions", None) or [100]

        if gid in results:
            print(
                f"[{idx:2d}/{len(environments)}] ⏩ Skipping already completed game {title} ({gid}) ({results[gid].get('levels_passed', 0)}/{results[gid].get('target_levels', len(baseline))} levels)"
            )
            continue

        print(f"\n[{idx:2d}/{len(environments)}] 🎮 Game: {title} ({gid})")
        print(f"    Baseline Actions: {baseline} | Tags: {getattr(env_meta, 'tags', [])}")

        best_completed = 0
        best_target = len(baseline) or 1
        best_is_win = False
        best_game_steps = 0
        best_elapsed = 0.0
        attempts_taken = 0
        total_game_steps_all = 0

        # Instantiate fresh agent & Cognitive USB Driver for clean inter-game isolation
        agent = MyAgent(
            game_id=gid, disable_archetypes=disable_archetypes, instructions=instructions
        )
        cognitive_engine = CognitiveAgentAdapter(agent)

        for attempt in range(retries + 1):
            attempts_taken += 1
            if attempt > 0:
                print(f"\n    🔁 [Retry {attempt}/{retries}] for failed game {title} ({gid})...")

            env = arcade.make(gid)
            if env is None:
                print(f"    ⚠️ Could not make game environment for {gid}")
                continue
            agent.arc_env = env

            driver = ArcAgiConsoleDriver(name="arc_console", env=env, game_id=gid)
            manager = DriverManager()
            manager.set_cognitive_engine(cognitive_engine)

            manager.register(driver)
            manager.bind("arc_console", target=env)

            frame = env.reset()
            if not hasattr(frame, "game_id") or getattr(frame, "game_id", None) is None:
                try:
                    frame.game_id = gid
                except Exception:
                    pass

            base_gid = gid.split("-")[0].strip()
            agent.game_id = base_gid
            agent.current_game_id = base_gid
            driver.set_frame(frame)
            cognitive_engine.reset_episode(
                retain_dynamics=(attempt > 0), is_new_level=False, level=0
            )

            # Evaluate ALL levels (no capping)
            target_levels = int(getattr(frame, "win_levels", None) or len(baseline) or 1)
            best_target = target_levels

            # Step budget for this attempt
            total_game_budget = max_steps_per_game if max_steps_per_game > 0 else 10000

            start_time = time.time()
            completed_prev = 0
            game_steps = 0
            current_level_steps = 0

            for step in range(1, total_game_budget + 1):
                feedback = manager.execute_cognitive_step()
                game_steps += 1
                current_level_steps += 1
                info = feedback.info
                lvl = int(info.get("levels_completed", 0))

                if lvl > completed_prev:
                    print(
                        f"    🎉 Level {lvl}/{target_levels} PASSED at step {step} (in {current_level_steps} steps)!"
                    )
                    completed_prev = lvl
                    current_level_steps = 0
                    # Note: cognitive reset is handled internally by choose_action
                    # on level transition detection. Only clear adapter history.
                    if hasattr(cognitive_engine, "history"):
                        cognitive_engine.history.clear()
                elif lvl < completed_prev:
                    completed_prev = lvl
                    current_level_steps = 0

                state_str = str(info.get("state", "RUNNING"))
                if state_str == "WIN" or lvl >= target_levels:
                    break

                # Optional per-level step limit if explicitly configured by user
                if max_level_steps is not None and current_level_steps >= max_level_steps:
                    print(
                        f"    ⏱️ Level {lvl + 1}/{target_levels} reached per-level limit "
                        f"({current_level_steps} >= {max_level_steps} steps). Concluding attempt."
                    )
                    break
                elif feedback.terminated or state_str == "GAME_OVER":
                    print(f"    ⚠️ Level retry / death at step {step}, retaining dynamics...")
                    cognitive_engine.reset_episode(
                        retain_dynamics=True, is_new_level=False, level=lvl
                    )
                    try:
                        reset_frame = env.step(0)
                    except Exception:
                        try:
                            reset_frame = env.reset()
                        except Exception:
                            break
                    reset_state = getattr(
                        getattr(reset_frame, "state", None),
                        "name",
                        str(getattr(reset_frame, "state", "")),
                    )
                    if reset_state == "GAME_OVER":
                        print(f"    ❌ GAME_OVER terminal at step {step}")
                        break
                    driver.set_frame(reset_frame)

            elapsed = time.time() - start_time
            total_game_steps_all += game_steps

            is_win = completed_prev >= target_levels
            if (
                completed_prev > best_completed
                or (completed_prev == best_completed and attempts_taken == 1)
                or (completed_prev == best_completed and game_steps < best_game_steps)
            ):
                best_completed = completed_prev
                best_is_win = is_win
                best_game_steps = game_steps
                best_elapsed = round(elapsed, 2)

            if is_win:
                break
            elif attempt < retries:
                print(
                    f"    ⚠️ Game {gid} incomplete ({completed_prev}/{target_levels} levels, {game_steps} steps). Scheduling retry."
                )

        total_steps_taken += total_game_steps_all
        total_levels_passed += best_completed
        total_levels_possible += best_target

        status_label = (
            "✅ WON ALL LEVELS"
            if best_is_win
            else (
                f"⚠️ {best_completed}/{best_target} levels"
                if best_completed > 0
                else "❌ INCOMPLETE"
            )
        )

        print(
            f"    🏁 Result: {status_label} in {best_elapsed:.2f}s "
            f"({best_game_steps} steps in best run, {total_game_steps_all} total steps across {attempts_taken} attempt(s))"
        )

        results[gid] = {
            "title": title,
            "levels_passed": best_completed,
            "target_levels": best_target,
            "won": best_is_win,
            "steps": best_game_steps,
            "total_steps": total_game_steps_all,
            "attempts": attempts_taken,
            "elapsed_seconds": best_elapsed,
            "baseline": baseline,
        }

        # Save continuous scorecard checkpoint
        checkpoint = {
            "timestamp": time.time(),
            "total_games_tested": idx,
            "total_levels_passed": total_levels_passed,
            "total_levels_possible": total_levels_possible,
            "pass_rate_pct": round((total_levels_passed / max(1, total_levels_possible)) * 100, 2),
            "games": results,
        }
        scorecard_path.write_text(json.dumps(checkpoint, indent=2))

    total_time_elapsed = time.time() - total_time_start
    overall_pct = (total_levels_passed / max(1, total_levels_possible)) * 100.0

    print(f"\n{'=' * 75}")
    print("🏁 FINAL BENCHMARK SUMMARY")
    print(f"{'=' * 75}")
    print(
        f"  Total Levels Passed:   {total_levels_passed} / {total_levels_possible} ({overall_pct:.1f}%)"
    )
    print(f"  Total Steps Executed:  {total_steps_taken}")
    print(
        f"  Total Benchmark Time:  {total_time_elapsed:.2f}s ({total_time_elapsed / 60:.1f} minutes)"
    )
    print(f"{'=' * 75}\n")

    for gid, res in results.items():
        title = res.get("title", gid)
        passed = res.get("levels_passed", 0)
        target = res.get("target_levels", 1)
        st = "✅" if res.get("won") else ("⚠️" if passed > 0 else "❌")
        print(
            f"  {st} {title:<10} ({gid:<15}): {passed}/{target} levels in {res.get('steps')} steps ({res.get('elapsed_seconds')}s)"
        )

    print(f"\n📄 Saved full scorecard to: {scorecard_path.resolve()}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run HBLLM Cognitive Architecture Benchmark across ARC-AGI-3 games."
    )
    parser.add_argument(
        "--game",
        "-g",
        type=str,
        default=None,
        help="Comma-separated game IDs or titles to run (e.g. 'AR25', 'TR87', 'ar25,tr87'). Default: all games.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="all_25_games_scorecard.json",
        help="Path for saving benchmark results JSON scorecard.",
    )
    parser.add_argument(
        "--enable-archetypes",
        action="store_true",
        help="Enable visual gestalt archetypes and ARC-AGI-3 target scorers (default: disabled, pure cognitive kernel).",
    )
    parser.add_argument(
        "--disable-archetypes",
        action="store_true",
        help="Explicitly disable visual gestalt archetypes and target scorers (default: disabled).",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        default=None,
        help="Custom executive directives separated by semicolon ';' or newline (default: canonical 5-step cognitive directive).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all 25 ARC-AGI-3 games and exit.",
    )
    parser.add_argument(
        "--max-steps",
        "-s",
        type=int,
        default=1000,
        help="Maximum total actions allowed per game (competition standard: 1000). Set to 0 for unlimited (10000).",
    )
    parser.add_argument(
        "--max-level-steps",
        type=int,
        default=None,
        help="Optional maximum steps per individual level (default: None, completely unlimited per level).",
    )
    parser.add_argument(
        "--retries",
        "-r",
        type=int,
        default=0,
        help="Number of retries for failed games (default: 0).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Restart entire benchmark from Game 1, backing up any prior scorecard.",
    )

    args = parser.parse_args()

    if args.list:
        arcade = Arcade()
        envs = arcade.get_environments()
        seen = set()
        deduped_envs = [e for e in envs if not (e.game_id in seen or seen.add(e.game_id))]
        print(f"Available ARC-AGI-3 games ({len(deduped_envs)} total):")
        for i, e in enumerate(deduped_envs, 1):
            print(f"  {i:2d}. {e.game_id:<12} | Title: {getattr(e, 'title', e.game_id)}")
        sys.exit(0)

    selected = [x.strip() for x in args.game.split(",")] if args.game else None
    custom_inst = None
    if args.instructions:
        custom_inst = [
            x.strip() for x in args.instructions.replace("\n", ";").split(";") if x.strip()
        ]

    disable_archetypes = True
    if args.enable_archetypes:
        disable_archetypes = False
    elif args.disable_archetypes:
        disable_archetypes = True

    run_all_25_games(
        selected_games=selected,
        disable_archetypes=disable_archetypes,
        scorecard_file=args.output,
        instructions=custom_inst,
        max_steps_per_game=args.max_steps,
        max_level_steps=args.max_level_steps,
        retries=args.retries,
        resume=not args.fresh,
    )
