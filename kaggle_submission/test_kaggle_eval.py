"""Benchmark MyAgent (from kaggle_submission.submission) on ARC-AGI-3 games.

Simulates the exact evaluation loop executed in the Kaggle notebook.
"""

import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from arc_agi import Arcade

try:
    from kaggle_submission.my_agent import MyAgent
except ImportError:
    from kaggle_submission.submission import MyAgent


def test_kaggle_submission_agent(games: list[str], max_levels: int = 2, max_steps: int = 120):
    arcade = Arcade()
    agent = MyAgent()

    scorecard = {}
    total_passed = 0
    total_possible = 0

    print(f"🚀 Running Kaggle Submission Evaluation across {len(games)} games...")
    print(f"Games: {', '.join(games)}\n")

    for idx, gid in enumerate(games, 1):
        env = arcade.make(gid)
        if env is None:
            print(f"[{idx}/{len(games)}] ⚠️ Could not make game {gid}")
            continue

        frame = env.reset()
        agent.reset_episode()
        history = []

        target_levels = min(getattr(frame, "win_levels", 1) or 1, max_levels)
        total_possible += target_levels

        start_t = time.time()
        completed_prev = 0

        for s in range(max_steps * target_levels):
            action = agent.choose_action(history, frame)
            history.append(frame)

            data = getattr(action, "data", None)
            if data is None and hasattr(action, "action_data") and action.action_data is not None:
                ad = action.action_data
                data = {"x": getattr(ad, "x", 0), "y": getattr(ad, "y", 0)}
            if data is None and hasattr(action, "is_complex") and action.is_complex():
                data = {"x": 0, "y": 0}

            frame = env.step(action, data=data)
            if frame is None:
                print(f"  [{gid:<4}] Step returned None at step {s + 1}")
                break

            lvl_completed = getattr(frame, "levels_completed", 0)
            if lvl_completed > completed_prev:
                print(
                    f"  [{gid:<4}] 🎉 Level {lvl_completed}/{target_levels} passed at step {s + 1}!"
                )
                completed_prev = lvl_completed
                # Notify agent of level completion
                agent.reset_episode(retain_dynamics=True)

            state_str = getattr(
                getattr(frame, "state", None), "name", str(getattr(frame, "state", ""))
            )
            if state_str == "WIN" or lvl_completed >= target_levels:
                break
            elif state_str == "GAME_OVER":
                print(f"  [{gid:<4}] ❌ Game over at step {s + 1}")
                break

        elapsed = time.time() - start_t
        scorecard[gid] = f"{completed_prev}/{target_levels}"
        total_passed += completed_prev
        status = "✅ WIN" if completed_prev >= target_levels else "❌ INCOMPLETE"
        print(
            f"[{idx}/{len(games)}] {gid:<6}: {completed_prev}/{target_levels} levels ({status}) in {elapsed:.2f}s ({len(history)} steps)"
        )

    print("\n" + "=" * 60)
    print(
        f"🏁 FINAL KAGGLE AGENT SCORECARD: {total_passed}/{total_possible} levels ({total_passed / max(1, total_possible) * 100:.1f}%)"
    )
    print("=" * 60)
    for gid, sc in scorecard.items():
        print(f"  {gid:<8}: {sc}")
    print("=" * 60)


if __name__ == "__main__":
    test_games = ["ft09", "ls20", "sc25", "cn04", "dc22", "ka59", "g50t", "bp35", "tr87", "tn36"]
    if len(sys.argv) > 1:
        test_games = [g.strip() for g in sys.argv[1].split(",")]
    test_kaggle_submission_agent(test_games)
