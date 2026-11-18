"""Data pipeline — download, clean, deduplicate, tokenize, and shard."""

from __future__ import annotations

from pathlib import Path

_exp_path = Path(__file__).resolve().parents[2] / "experiments" / "legacy_llm" / "data"
if _exp_path.exists() and str(_exp_path) not in __path__:
    __path__.append(str(_exp_path))


from hbllm.data.interaction_miner import InteractionMiner

__all__ = ["InteractionMiner"]
