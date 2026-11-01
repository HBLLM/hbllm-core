"""Per-step trace recorder (WP0.1): JSONL records + de-duplicated npz frames."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class TraceRecorder:
    """Writes one JSONL record per decide() step; frames dedup'd to frames/<hash>.npz."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "frames").mkdir(exist_ok=True)
        self._fp = open(self.path / "trace.jsonl", "w")

    def record_step(
        self,
        *,
        game: str,
        level: int,
        step: int,
        grid: Any,
        available_actions: list[int],
        action_id: int,
        action_data: dict[str, Any] | None,
        predicted_effect: Any = None,
        observed_effect: Any = None,
        levels_completed: int = 0,
        state: str = "NOT_FINISHED",
        latency_ms: float = 0.0,
        stage_latency_ms: dict[str, float] | None = None,
        stagnation_steps: int = 0,
        boundary: str | None = None,
        contract_violation: str | None = None,
        branch: str | None = None,
    ) -> str:
        import numpy as np

        arr = np.asarray(grid, dtype=np.int32)
        frame_hash = hashlib.blake2b(arr.tobytes(), digest_size=12).hexdigest()
        fpath = self.path / "frames" / f"{frame_hash}.npz"
        if not fpath.exists():
            np.savez_compressed(fpath, grid=arr)
        rec = {
            "game": game,
            "level": level,
            "step": step,
            "frame_hash": frame_hash,
            "frame_shape": list(arr.shape),
            "available_actions": list(available_actions),
            "action_id": action_id,
            "action_data": action_data,
            "predicted_effect": predicted_effect,
            "observed_effect": observed_effect,
            "levels_completed": levels_completed,
            "state": state,
            "latency_ms": latency_ms,
            "stage_latency_ms": stage_latency_ms or {},
            "stagnation_steps": stagnation_steps,
            "boundary": boundary,
            "contract_violation": contract_violation,
            "branch": branch,
        }
        self._fp.write(json.dumps(rec) + "\n")
        self._fp.flush()
        return frame_hash

    def close(self) -> None:
        self._fp.close()

    @staticmethod
    def from_env(default: str | None = None) -> TraceRecorder | None:
        d = os.environ.get("HBLLM_TRACE_DIR") or default
        return TraceRecorder(d) if d else None
