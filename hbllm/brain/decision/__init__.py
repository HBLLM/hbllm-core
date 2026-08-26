"""A19 Intrinsic Curiosity, Active Epistemic Discovery & Decision Policy Package.

Provides EpistemicGap scanning, Shannon entropy quantification, Bayesian posterior updates,
discriminative EpistemicProbes, expected information gain, Value-of-Information (VoI),
multi-criteria DecisionEngine, and rational inaction.
"""

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
