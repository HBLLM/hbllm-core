"""A19 Intrinsic Curiosity, Active Epistemic Discovery & Decision Policy Package.

Provides EpistemicGap scanning, Shannon entropy quantification, Bayesian posterior updates,
discriminative EpistemicProbes, expected information gain, Value-of-Information (VoI),
multi-criteria DecisionEngine, and rational inaction.
"""

from __future__ import annotations

from pathlib import Path

_exp_path = Path(__file__).resolve().parents[3] / "experiments" / "legacy_brain" / "decision"
if _exp_path.exists() and str(_exp_path) not in __path__:
    __path__.append(str(_exp_path))


from hbllm.brain.decision.discovery_loop import ActiveDiscoveryLoop
from hbllm.brain.decision.gap import (
    EpistemicGap,
    EpistemicGapScanner,
    HypothesisOption,
)
from hbllm.brain.decision.policy import (
    CandidateKind,
    DecisionCandidate,
    DecisionEngine,
    DecisionResult,
    DecisionType,
)
from hbllm.brain.decision.probe import EpistemicProbe, ProbeGenerator

__all__ = [
    "ActiveDiscoveryLoop",
    "CandidateKind",
    "DecisionCandidate",
    "DecisionEngine",
    "DecisionResult",
    "DecisionType",
    "EpistemicGap",
    "EpistemicGapScanner",
    "EpistemicProbe",
    "HypothesisOption",
    "ProbeGenerator",
]
