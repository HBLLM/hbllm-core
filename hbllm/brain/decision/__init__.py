"""A19 Intrinsic Curiosity, Active Epistemic Discovery & Decision Policy Package.

Provides EpistemicGap scanning, Shannon entropy quantification, Bayesian posterior updates,
discriminative EpistemicProbes, expected information gain, Value-of-Information (VoI),
multi-criteria DecisionEngine, and rational inaction.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ActiveDiscoveryLoop": ("hbllm.brain.decision.discovery_loop", "ActiveDiscoveryLoop"),
    "EpistemicGap": ("hbllm.brain.decision.gap", "EpistemicGap"),
    "EpistemicGapScanner": ("hbllm.brain.decision.gap", "EpistemicGapScanner"),
    "HypothesisOption": ("hbllm.brain.decision.gap", "HypothesisOption"),
    "CandidateKind": ("hbllm.brain.decision.policy", "CandidateKind"),
    "DecisionCandidate": ("hbllm.brain.decision.policy", "DecisionCandidate"),
    "DecisionEngine": ("hbllm.brain.decision.policy", "DecisionEngine"),
    "DecisionResult": ("hbllm.brain.decision.policy", "DecisionResult"),
    "DecisionType": ("hbllm.brain.decision.policy", "DecisionType"),
    "EpistemicProbe": ("hbllm.brain.decision.probe", "EpistemicProbe"),
    "ProbeGenerator": ("hbllm.brain.decision.probe", "ProbeGenerator"),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod_path, attr = _EXPORTS[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))
