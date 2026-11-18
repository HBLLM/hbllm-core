"""Tamil Surface Realizer for A16 non-LLM multilingual cognition.

Converts cognitive EpistemicState and SemanticFrames into natural Tamil utterances
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


class TamilRealizer:
    """Generates calibrated Tamil surface utterances from cognitive state."""

    def __init__(self, policy: EpistemicRealizationPolicy | None = None) -> None:
        self._policy = policy or EpistemicRealizationPolicy()
        self._concept_map = {
            "ball": "பந்து",
            "table": "மேசை",
            "box": "பெட்டி",
            "cup": "கோப்பை",
            "robot": "ரோபோ",
            "door": "கதவு",
            "front_door": "முன் கதவு",
            "back_door": "பின் கதவு",
            "gate": "வாயில்",
            "arm": "கை",
            "robot_arm": "ரோபோ கை",
        }
        self._color_map = {
            "red": "சிவப்பு",
            "blue": "நீலம்",
            "green": "பச்சை",
        }
        self._postp_map = {
            "located_on": "மீது",
            "located_in": "உள்ளே",
            "below": "கீழே",
            "near": "அருகில்",
        }

    def realize(
        self,
        epistemic_state: CognitiveEpistemicState,
        original_frame: SemanticFrame | None = None,
    ) -> str:
        """Verbalize CognitiveEpistemicState into calibrated Tamil."""
        level = self._policy.evaluate(epistemic_state)
        subj_ta = self._concept_map.get(
            epistemic_state.target_subject, epistemic_state.target_subject
        )
        obj_ta = self._concept_map.get(
            epistemic_state.target_object or "", epistemic_state.target_object or ""
        )
        postp = self._postp_map.get(epistemic_state.target_predicate, "மீது")
        if epistemic_state.target_predicate in ("located_in", "in"):
            postp = "மீது" if obj_ta in ("மேசை", "நாற்காலி", "தரை") else "உள்ளே"

        # 1. Unknown / Insufficient evidence
        if level == EpistemicVerbalizationLevel.INSUFFICIENT_EVIDENCE:
            return f"{subj_ta} எங்கே உள்ளது என்பதை தீர்மானிக்க போதுமான ஆதாரம் இல்லை."

        # 2. Contradiction
        if level == EpistemicVerbalizationLevel.CONTRADICTED:
            return f"{subj_ta} குறித்து முரண்பட்ட சான்றுகள் உள்ளன."

        # 3. Verification (Yes/No answer)
        if original_frame and original_frame.query_target == "verification":
            if epistemic_state.raw_belief_value is True:
                if level == EpistemicVerbalizationLevel.CERTAIN:
                    return f"ஆம், {subj_ta} {obj_ta} {postp} இருக்கிறது."
                elif level == EpistemicVerbalizationLevel.PROBABLE:
                    return f"ஆம், {subj_ta} அநேகமாக {obj_ta} {postp} இருக்கலாம்."
                else:
                    return f"நான் நினைக்கிறேன் {subj_ta} {obj_ta} {postp} இருக்கலாம்."
            else:
                return f"இல்லை, {subj_ta} {obj_ta} {postp} இல்லை."

        # 4. Spatial / Location Query Answer
        if level == EpistemicVerbalizationLevel.CERTAIN:
            return f"{subj_ta} {obj_ta} {postp} இருக்கிறது."
        elif level == EpistemicVerbalizationLevel.PROBABLE:
            return f"{subj_ta} அநேகமாக {obj_ta} {postp} இருக்கலாம்."
        elif level == EpistemicVerbalizationLevel.PLAUSIBLE:
            return f"நான் நினைக்கிறேன் {subj_ta} {obj_ta} {postp} இருக்கலாம்."
        else:
            return f"{subj_ta} {obj_ta} {postp} இருக்கிறதா என்று எனக்கு உறுதியாக தெரியவில்லை."

    def realize_frame(self, frame: SemanticFrame) -> str:
        """Realize a SemanticFrame directly into Tamil text (for interlingual translation)."""
        theme_ref = frame.get_role(ThematicRole.THEME) or frame.get_role(ThematicRole.AGENT)
        loc_ref = frame.get_role(ThematicRole.LOCATION) or frame.get_role(ThematicRole.DESTINATION)
        patient_ref = frame.get_role(ThematicRole.PATIENT)

        if frame.frame_type == FrameType.ASSERTION:
            subj = "பொருள்"
            color_adj = ""
            if theme_ref:
                if "color" in theme_ref.properties:
                    color_adj = self._color_map.get(theme_ref.properties["color"], "") + " "
                c_name = theme_ref.concept_name or ""
                subj = self._concept_map.get(c_name, c_name) or subj

            loc = "மேசை"
            postp = self._postp_map.get(frame.predicate, "மீது")
            if loc_ref:
                loc_c_name = loc_ref.concept_name or ""
                loc = self._concept_map.get(loc_c_name, loc_c_name) or loc

            return f"{color_adj}{subj} {loc} {postp} இருக்கிறது."

        elif frame.frame_type == FrameType.COMMAND:
            patient_name = "பொருள்"
            if patient_ref:
                p_c_name = patient_ref.concept_name or ""
                patient_name = self._concept_map.get(p_c_name, p_c_name) or patient_name
            verb_map = {
                "open": "திறக்கவும்",
                "lock": "பூட்டவும்",
                "close": "மூடவும்",
                "rotate": "சுழற்றவும்",
                "move": "நகர்த்தவும்",
                "push": "தள்ளவும்",
                "stop": "நிறுத்தவும்",
            }
            tamil_verb = verb_map.get(frame.predicate, "செய்க")
            return f"{patient_name} {tamil_verb}"

        return "செயல்பாடு முடிந்தது."
