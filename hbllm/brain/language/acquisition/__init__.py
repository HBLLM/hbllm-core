"""A17 Grounded Language Learning Package.

Provides autonomous lexicon acquisition through cross-situational observation,
ostensive teaching, state transition grounding, contrastive differentiation,
and error-driven revision over HCIR.
"""

from hbllm.brain.language.acquisition.contrastive_learner import (
    ContrastiveLearner,
    ContrastiveRelation,
)
from hbllm.brain.language.acquisition.cross_situational_learner import CrossSituationalLearner
from hbllm.brain.language.acquisition.grounded_lexicon import (
    GroundedLexicon,
    GroundingResult,
    RealizationResult,
)
from hbllm.brain.language.acquisition.lexical_hypothesis import (
    EvidenceSourceType,
    LexicalCandidate,
    LexicalCandidateStatus,
    LexicalEvidence,
    LexicalHypothesisSet,
    LexicalSense,
    LexicalTargetType,
)
from hbllm.brain.language.acquisition.lexicon_acquisition_loop import (
    AcquisitionCycleResult,
    LexiconAcquisitionLoop,
)
from hbllm.brain.language.acquisition.ostensive_teacher import OstensiveTeacher
from hbllm.brain.language.acquisition.scoring import (
    apply_evidence_to_candidate,
    update_candidate_status,
)

__all__ = [
    "AcquisitionCycleResult",
    "ContrastiveLearner",
    "ContrastiveRelation",
    "CrossSituationalLearner",
    "EvidenceSourceType",
    "GroundedLexicon",
    "GroundingResult",
    "LexicalCandidate",
    "LexicalCandidateStatus",
    "LexicalEvidence",
    "LexicalHypothesisSet",
    "LexicalSense",
    "LexicalTargetType",
    "LexiconAcquisitionLoop",
    "OstensiveTeacher",
    "RealizationResult",
    "apply_evidence_to_candidate",
    "update_candidate_status",
]
