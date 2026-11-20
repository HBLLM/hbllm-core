"""Tamil Surface Realizer for A16 non-LLM multilingual cognition.

Converts cognitive EpistemicState and SemanticFrames into natural Tamil utterances
calibrated to epistemic confidence levels.
Implements the LanguageRealizer protocol via the Universal Declarative Engine.
"""

from __future__ import annotations

from pathlib import Path

from hbllm.hcir.language.core.epistemic_policy import EpistemicRealizationPolicy
from hbllm.hcir.language.core.profile import LanguageProfile
from hbllm.hcir.language.core.universal_engine import UniversalLanguageRealizer

_TA_PROFILE_PATH = Path(__file__).resolve().parent.parent / "profiles" / "ta.json"


class TamilRealizer(UniversalLanguageRealizer):
    """Generates calibrated Tamil surface utterances using Universal Engine."""

    def __init__(self, policy: EpistemicRealizationPolicy | None = None) -> None:
        profile = LanguageProfile.from_file(_TA_PROFILE_PATH)
        super().__init__(profile=profile, epistemic_policy=policy)
