"""Offline open-loop trace replay (WP0.1/WP1 golden gate).

Feeds recorded frames into a FRESH agent per game and compares the chosen
action ids (and ACTION6 payloads) with what the recording agent emitted.
Group by ``game`` so each game replays on a fresh agent like the runner.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReplayReport:
    total_steps: int = 0
    first_divergence_step: int | None = None
    action_agreement_ratio: float = 0.0
    per_game: dict[str, dict[str, Any]] = field(default_factory=dict)


def _default_agent_factory(game_id: str, seed: int) -> Any:
    sys.path.insert(0, "kaggle_submission")
    from my_agent import MyAgent

    return MyAgent(card_id="replay", game_id=game_id, seed=seed)


def _mk_frame(grid: Any, rec: dict) -> Any:
    """Rebuild a LocalFrame-compatible object the agent saw at decision time."""
    import numpy as np

    from plugins.arc_agi_adapter.local_env import GameState, LocalFrame

    g2d = np.asarray(grid)
    win_levels = rec.get("win_levels")
    try:
        from plugins.arc_agi_adapter.local_env import LocalEnv

        spec_game = LocalEnv(0).make(rec["game"])
        if win_levels is None:
            win_levels = spec_game.win_levels
    except Exception:
        if win_levels is None:
            win_levels = 1
    # The recorded npz IS the normalized grid the engine consumed — feed it
    # verbatim (both as the frame layer and the grid attr).
    return LocalFrame(
        state=GameState[rec.get("state", "NOT_FINISHED")],
        frame=[g2d.tolist()],
        grid=g2d.tolist(),
        levels_completed=int(rec.get("levels_completed", 0)),
        win_levels=int(win_levels),
        available_actions=list(rec.get("available_actions", [])),
        step_count=int(rec.get("step", 0)),
    )


def replay_trace(
    trace_path: str | Path,
    agent_factory: Callable[[str, int], Any] | None = None,
    seed: int = 0,
) -> ReplayReport:
    """Replay a recorded trace open-loop; per-game agreement + divergence."""
    import numpy as np

    factory = agent_factory or _default_agent_factory
    trace_path = Path(trace_path)
    frames_dir = trace_path.parent / "frames"

    recs_by_game: dict[str, list[dict]] = defaultdict(list)
    with open(trace_path) as fp:
        for line in fp:
            rec = json.loads(line)
            recs_by_game[rec["game"]].append(rec)

    report = ReplayReport()
    for gid, recs in recs_by_game.items():
        agent = factory(gid, seed)
        agree = 0
        first_div = None
        for rec in recs:
            npz = frames_dir / f"{rec['frame_hash']}.npz"
            if not npz.exists():
                continue
            grid = np.load(npz)["grid"]
            frame = _mk_frame(grid, rec)
            act = agent.choose_action([frame], frame)
            aid = int(getattr(act, "value", act))
            adata = getattr(act, "data", None)
            report.total_steps += 1
            agreed = aid == int(rec["action_id"]) and (
                aid != 6 or (adata or {}) == (rec.get("action_data") or {})
            )
            agree += int(agreed)
            if not agreed and first_div is None:
                first_div = rec["step"]
        n = len(recs)
        report.per_game[gid] = {
            "steps": n,
            "agreement": agree / max(1, n),
            "first_divergence_step": first_div,
        }
        if first_div is not None and report.first_divergence_step is None:
            report.first_divergence_step = first_div
    if report.total_steps:
        report.action_agreement_ratio = (
            sum(g["agreement"] * g["steps"] for g in report.per_game.values()) / report.total_steps
        )
    return report


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: python -m plugins.arc_agi_adapter.replay <trace.jsonl> [--seed 0]")
        return 2
    seed = 0
    if "--seed" in argv:
        i = argv.index("--seed")
        seed = int(argv[i + 1])
    rep = replay_trace(argv[0], seed=seed)
    for gid, g in sorted(rep.per_game.items()):
        print(
            f"{gid:20s} steps={g['steps']:4d} agreement={g['agreement']:.3f} first_div={g['first_divergence_step']}"
        )
    print(f"OVERALL steps={rep.total_steps} agreement={rep.action_agreement_ratio:.3f}")
    return 0 if rep.action_agreement_ratio == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
