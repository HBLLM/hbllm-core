"""English Surface Realizer for A16.

Converts cognitive EpistemicState and SemanticFrames into natural, grammatically correct
English sentences with epistemically calibrated hedges.
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


class EnglishRealizer:
    """Generates calibrated English surface utterances from cognitive state."""

    def __init__(self, policy: EpistemicRealizationPolicy | None = None) -> None:
        self._policy = policy or EpistemicRealizationPolicy()

    def realize(
        self,
        epistemic_state: CognitiveEpistemicState,
        original_frame: SemanticFrame | None = None,
    ) -> str:
        """Verbalize a CognitiveEpistemicState into an epistemically calibrated English answer."""
        level = self._policy.evaluate(epistemic_state)
        subj = epistemic_state.target_subject
        obj = epistemic_state.target_object or ""
        pred = epistemic_state.target_predicate

        # Prep preposition phrasing
        prep = "on"
        if pred in ("located_in", "in"):
            prep = "on" if obj.lower() in ("table", "shelf", "floor", "desk", "chair") else "in"
        elif pred in ("below", "under"):
            prep = "under"
        elif pred in ("near",):
            prep = "near"

        # 1. Unknown / Insufficient Evidence
        if level == EpistemicVerbalizationLevel.INSUFFICIENT_EVIDENCE:
            if original_frame and original_frame.query_target == "location":
                return f"I do not have enough evidence to determine where the {subj} is."
            elif original_frame and original_frame.query_target == "property":
                return f"I do not know the {pred} of the {subj}."
            return f"I do not have enough evidence to determine the state of the {subj}."

        # 2. Contradiction
        if level == EpistemicVerbalizationLevel.CONTRADICTED:
            return f"There is conflicting evidence regarding the {subj}."

        # 3. Verification (Yes/No answer)
        if original_frame and original_frame.query_target == "verification":
            if epistemic_state.raw_belief_value is True:
                if level == EpistemicVerbalizationLevel.CERTAIN:
                    return f"Yes, the {subj} is {prep} the {obj}."
                elif level == EpistemicVerbalizationLevel.PROBABLE:
                    return f"Yes, the {subj} is probably {prep} the {obj}."
                else:
                    return f"I think the {subj} may be {prep} the {obj}."
            else:
                return f"No, the {subj} is not {prep} the {obj}."

        # 4. Property Query ("What color is the ball?")
        if original_frame and original_frame.query_target == "property" or pred == "color":
            if level == EpistemicVerbalizationLevel.CERTAIN:
                return f"The {subj} is {obj}."
            elif level == EpistemicVerbalizationLevel.PROBABLE:
                return f"The {subj} is probably {obj}."
            else:
                return f"I think the {subj} may be {obj}."

        # 5. Spatial / Location Query ("Where is the ball?")
        if level == EpistemicVerbalizationLevel.CERTAIN:
            return f"The {subj} is {prep} the {obj}."
        elif level == EpistemicVerbalizationLevel.PROBABLE:
            return f"The {subj} is probably {prep} the {obj}."
        elif level == EpistemicVerbalizationLevel.PLAUSIBLE:
            return f"I think the {subj} may be {prep} the {obj}."
        else:
            return f"I am not certain, but the {subj} might be {prep} the {obj}."

    def realize_frame(self, frame: SemanticFrame) -> str:
        """Realize a SemanticFrame directly into declarative English (for interlingual translation)."""
        theme_ref = frame.get_role(ThematicRole.THEME) or frame.get_role(ThematicRole.AGENT)
        loc_ref = (
            frame.get_role(ThematicRole.LOCATION)
            or frame.get_role(ThematicRole.DESTINATION)
            or frame.get_role(ThematicRole.PATIENT)
        )

        subj = theme_ref.concept_name if theme_ref else "entity"
        obj = loc_ref.concept_name if loc_ref else "entity"

        # Check properties on theme
        prop_str = ""
        if theme_ref and theme_ref.properties:
            color = theme_ref.properties.get("color")
            if color:
                prop_str = f"{color} "

        prep = "on"
        if frame.predicate in ("located_in", "in"):
            prep = "in"
        elif frame.predicate in ("below", "under"):
            prep = "under"

        if frame.frame_type == FrameType.ASSERTION:
            if frame.predicate in ("located_on", "located_in", "below", "near", "above"):
                return f"The {prop_str}{subj} is {prep} the {obj}."
            elif frame.predicate == "is_property":
                props = theme_ref.properties if theme_ref else {}
                val = list(props.values())[0] if props else "present"
                return f"The {subj} is {val}."
            elif frame.predicate:
                return f"The {subj} {frame.predicate} the {obj}."

        elif frame.frame_type == FrameType.COMMAND:
            return f"Move the {subj} to the {obj}."

        return f"The {subj} is {obj}."
