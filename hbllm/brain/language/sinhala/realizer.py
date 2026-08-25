"""Sinhala Surface Realizer for A16.

Converts cognitive EpistemicState and SemanticFrames into natural Sinhala utterances
calibrated to epistemic confidence levels.
Implements the LanguageRealizer protocol.
"""

from __future__ import annotations

import logging

from hbllm.brain.language.core.epistemic_policy import (
    CognitiveEpistemicState,
    EpistemicRealizationPolicy,
    EpistemicVerbalizationLevel,
)
from hbllm.brain.language.core.semantic_frame import (
    FrameType,
    SemanticFrame,
    ThematicRole,
)

logger = logging.getLogger(__name__)


class SinhalaRealizer:
    """Generates calibrated Sinhala surface utterances from cognitive state."""

    def __init__(self, policy: EpistemicRealizationPolicy | None = None) -> None:
        self._policy = policy or EpistemicRealizationPolicy()
        self._concept_map = {
            "ball": "බෝලය",
            "table": "මේසය",
            "box": "පෙට්ටිය",
            "cup": "කෝප්පය",
            "robot": "රොබෝ",
        }
        self._color_map = {
            "red": "රතු",
            "blue": "නිල්",
            "green": "කොළ",
        }
        self._postp_map = {
            "located_on": "මත",
            "located_in": "තුළ",
            "below": "යට",
            "near": "ළඟ",
        }

    def realize(
        self,
        epistemic_state: CognitiveEpistemicState,
        original_frame: SemanticFrame | None = None,
    ) -> str:
        """Verbalize CognitiveEpistemicState into calibrated Sinhala."""
        level = self._policy.evaluate(epistemic_state)
        subj_sin = self._concept_map.get(epistemic_state.target_subject, epistemic_state.target_subject)
        obj_sin = self._concept_map.get(epistemic_state.target_object or "", epistemic_state.target_object or "")
        postp = self._postp_map.get(epistemic_state.target_predicate, "මත")
        if epistemic_state.target_predicate in ("located_in", "in"):
            postp = "මත" if obj_sin in ("මේසය", "මේසේ", "පුටුව", "බිම") else "තුළ"

        # 1. Unknown / Insufficient evidence
        if level == EpistemicVerbalizationLevel.INSUFFICIENT_EVIDENCE:
            return f"{subj_sin} කොහෙද තියෙන්නේ කියලා තීරණය කිරීමට ප්‍රමාණවත් සාක්ෂි නැත."

        # 2. Contradiction
        if level == EpistemicVerbalizationLevel.CONTRADICTED:
            return f"{subj_sin} පිළිබඳව පරස්පර සාක්ෂි තිබේ."

        # 3. Verification (Yes/No answer)
        if original_frame and original_frame.query_target == "verification":
            if epistemic_state.raw_belief_value is True:
                if level == EpistemicVerbalizationLevel.CERTAIN:
                    return f"ඔව්, {subj_sin} {obj_sin} {postp} තියෙනවා."
                elif level == EpistemicVerbalizationLevel.PROBABLE:
                    return f"ඔව්, {subj_sin} බොහෝ විට {obj_sin} {postp} තියෙන්න පුළුවන්."
                else:
                    return f"මම හිතන්නේ {subj_sin} {obj_sin} {postp} තියෙන්න පුළුවන්."
            else:
                return f"නැත, {subj_sin} {obj_sin} {postp} නැත."

        # 4. Spatial / Location Query Answer
        if level == EpistemicVerbalizationLevel.CERTAIN:
            return f"{subj_sin} {obj_sin} {postp} තියෙනවා."
        elif level == EpistemicVerbalizationLevel.PROBABLE:
            return f"{subj_sin} බොහෝ විට {obj_sin} {postp} තියෙන්න පුළුවන්."
        elif level == EpistemicVerbalizationLevel.PLAUSIBLE:
            return f"මම හිතන්නේ {subj_sin} {obj_sin} {postp} තියෙන්න පුළුවන්."
        else:
            return f"{subj_sin} {obj_sin} {postp} තියෙනවද කියලා මට විශ්වාස නැත."

    def realize_frame(self, frame: SemanticFrame) -> str:
        """Realize a SemanticFrame directly into Sinhala text (for interlingual translation)."""
        theme_ref = frame.get_role(ThematicRole.THEME) or frame.get_role(ThematicRole.AGENT)
        loc_ref = frame.get_role(ThematicRole.LOCATION) or frame.get_role(ThematicRole.DESTINATION)

        subj_concept: str = (theme_ref.concept_name if (theme_ref and theme_ref.concept_name) else "entity")
        obj_concept: str = (loc_ref.concept_name if (loc_ref and loc_ref.concept_name) else "entity")

        subj_sin = self._concept_map.get(subj_concept, subj_concept)
        obj_sin = self._concept_map.get(obj_concept, obj_concept)

        prop_str = ""
        if theme_ref and theme_ref.properties:
            color = theme_ref.properties.get("color")
            if color:
                prop_str = f"{self._color_map.get(color, color)} "

        postp = self._postp_map.get(frame.predicate, "මත")

        if frame.frame_type == FrameType.ASSERTION:
            return f"{prop_str}{subj_sin} {obj_sin} {postp} තියෙනවා."
        elif frame.frame_type == FrameType.COMMAND:
            return f"{subj_sin} {obj_sin} වෙත ගෙනයන්න."

        return f"{subj_sin} {obj_sin} {postp} තියෙනවා."
