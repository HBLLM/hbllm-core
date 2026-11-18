"""Model architecture — custom transformer for HBLLM."""

from __future__ import annotations

from pathlib import Path

_exp_path = Path(__file__).resolve().parents[2] / "experiments" / "legacy_llm" / "model"
if _exp_path.exists() and str(_exp_path) not in __path__:
    __path__.append(str(_exp_path))


__all__: list[str] = []
