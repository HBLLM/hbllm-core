"""A17 Grounded Language Learning Package.

Provides autonomous lexicon acquisition through cross-situational observation,
ostensive teaching, state transition grounding, contrastive differentiation,
and error-driven revision over HCIR.
"""

from hbllm.hcir.language.acquisition.contrastive_learner import (
    ContrastiveLearner,
    ContrastiveRelation,
)
from hbllm.hcir.language.acquisition.cross_situational_learner import CrossSituationalLearner
from hbllm.hcir.language.acquisition.grounded_lexicon import (
    GroundedLexicon,
    GroundingResult,
    RealizationResult,
)
from hbllm.hcir.language.acquisition.lexical_hypothesis import (
    EvidenceSourceType,
    LexicalCandidate,
    LexicalCandidateStatus,
    LexicalEvidence,
    LexicalHypothesisSet,
    LexicalSense,
    LexicalTargetType,
)
from hbllm.hcir.language.acquisition.lexicon_acquisition_loop import (
    AcquisitionCycleResult,
    LexiconAcquisitionLoop,
)
from hbllm.hcir.language.acquisition.ostensive_teacher import OstensiveTeacher
from hbllm.hcir.language.acquisition.scoring import (
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
