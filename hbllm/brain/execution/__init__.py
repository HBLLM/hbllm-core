"""Brain execution nodes — translation boundary between cognitive OS and Execution OS."""

from __future__ import annotations

from pathlib import Path

_exp_path = Path(__file__).resolve().parents[3] / "experiments" / "legacy_brain" / "execution"
if _exp_path.exists() and str(_exp_path) not in __path__:
    __path__.append(str(_exp_path))
