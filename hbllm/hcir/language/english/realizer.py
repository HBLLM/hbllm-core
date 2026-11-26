"""English Surface Realizer for A16.

Converts cognitive EpistemicState and SemanticFrames into natural, grammatically correct
English sentences with epistemically calibrated hedges.
Implements the LanguageRealizer protocol.
"""

from __future__ import annotations

import logging

from hbllm.hcir.language.core.epistemic_policy import (
    CognitiveEpistemicState,
    EpistemicRealizationPolicy,
    EpistemicVerbalizationLevel,
)
from hbllm.hcir.language.core.semantic_frame import (
    FrameType,
    SemanticFrame,
    ThematicRole,
)
from hbllm.hcir.language.english.morphology import (
    INVARIANT_S_NOUNS,
    singularize_noun,
)

logger = logging.getLogger(__name__)

MASS_NOUNS = frozenset(
    {
        "oxygen",
        "hydrogen",
        "nitrogen",
        "carbon",
        "water",
        "air",
        "gravity",
        "friction",
        "matter",
        "code",
        "software",
        "hardware",
        "energy",
        "heat",
        "rust",
        "iron",
        "gold",
        "silver",
        "wood",
    }
)


class EnglishRealizer:
    """Generates calibrated English surface utterances from cognitive state."""

    def __init__(self, policy: EpistemicRealizationPolicy | None = None) -> None:
        self._policy = policy or EpistemicRealizationPolicy()

    @classmethod
    def _format_noun(cls, word: str, title: bool = False) -> str:
        """Format noun with correct indefinite article ('a'/'an') or bare noun for proper/mass nouns."""
        if not word:
            return ""
        clean = word.strip()
        lower = clean.lower()

        # If already starts with an article
        if lower.startswith(("a ", "an ", "the ")):
            return clean.capitalize() if title else clean

        # Proper noun check: capitalized or known proper name
        if clean[0].isupper() or lower in (
            "gandalf",
            "darcy",
            "mr. darcy",
            "saruman",
            "elizabeth",
            "elizabeth bennet",
        ):
            return clean.capitalize() if title else clean

        # Mass / uncountable noun check
        if lower in MASS_NOUNS or lower in INVARIANT_S_NOUNS:
            return clean.capitalize() if title else clean

        # Plural noun check (e.g. reptiles, dogs, mammals, programs, birds)
        if lower.endswith("s") and not lower.endswith(("ss", "us", "is", "ics")):
            sing = singularize_noun(clean)
            if sing != lower:
                art = "An" if sing.lower()[0] in "aeiou" else "A"
                if not title:
                    art = art.lower()
                return f"{art} {sing}"

        # Countable common noun: prefix 'a' or 'an'
        art = "An" if lower[0] in "aeiou" else "A"
        if not title:
            art = art.lower()
        return f"{art} {clean}"

    @staticmethod
    def _inflect_3s_simple(verb: str) -> str:
        v = verb.strip().lower()
        if not v:
            return v
        if v.endswith(("sh", "ch", "s", "x", "z", "o")):
            return v + "es"
        if v.endswith("y") and len(v) > 2 and v[-2] not in "aeiou":
            return v[:-1] + "ies"
        return v + "s"

    @staticmethod
    def _format_svo_subject(subj: str, original_frame: SemanticFrame | None) -> str:
        clean = subj.strip()
        if not clean:
            return "It"
        if original_frame is not None:
            for role in (ThematicRole.THEME, ThematicRole.AGENT):
                ref = original_frame.get_role(role)
                if ref and ref.raw_text:
                    raw_w = ref.raw_text.strip()
                    if (
                        raw_w
                        and raw_w[0].isupper()
                        and not raw_w.lower().startswith(("a ", "an ", "the "))
                    ):
                        return raw_w[0].upper() + clean[1:] if len(clean) > 1 else raw_w
        return clean[0].upper() + clean[1:]

    def realize(
        self,
        epistemic_state: CognitiveEpistemicState,
        original_frame: SemanticFrame | None = None,
    ) -> str:
        """Verbalize a CognitiveEpistemicState into an epistemically calibrated English answer."""
        level, deg_hedge = self._policy.evaluate_two_dimensional(epistemic_state)
        subj = epistemic_state.target_subject
        obj = epistemic_state.target_object or ""
        pred = epistemic_state.target_predicate
        q_target = original_frame.query_target if original_frame else None

        # Prep preposition phrasing
        prep = "on"
        if pred in ("located_in", "in"):
            prep = "on" if obj.lower() in ("table", "shelf", "floor", "desk", "chair") else "in"
        elif pred in ("below", "under", "located_under", "located_beneath"):
            prep = "under"
        elif pred in ("above", "located_above"):
            prep = "above"
        elif pred in ("near", "located_near", "located_beside"):
            prep = "near"
        if deg_hedge is not None:
            prep = f"{deg_hedge} {prep}"

        # 0. Specialized identity, causal, and requirement answers
        if pred == "identity" and obj:
            return f"{obj}." if not obj.endswith(".") else obj

        is_verification = bool(original_frame and q_target == "verification")

        if pred in ("causes", "enables") and obj and not is_verification:
            if q_target in ("enables", "enable"):
                return f"{subj.capitalize()} enables {obj}."
            if q_target == "cause":
                return f"{subj.capitalize()} causes {obj}."
            if ("." in obj or "\n" in obj) and len(obj) > 30:
                return obj
            if pred == "enables":
                return f"{subj.capitalize()} enables {obj}."

        if pred == "requires" and obj and not is_verification:
            if q_target in ("procedure", "requirement"):
                if ("." in obj or "\n" in obj) and len(obj) > 30:
                    return obj
                return f"To operate or function, {subj} fundamentally requires {obj}."

        # 1. Unknown / Insufficient Evidence
        if level == EpistemicVerbalizationLevel.INSUFFICIENT_EVIDENCE:
            if q_target == "location":
                return f"I do not have enough evidence to determine where the {subj} is."
            elif q_target == "svo_object":
                return f"I do not know what {subj} {self._inflect_3s_simple(pred)}."
            elif q_target == "svo_location":
                return f"I do not know where {subj} {self._inflect_3s_simple(pred)}."
            elif q_target == "svo_subject":
                if pred == "has_property":
                    return f"I do not know who is {obj}."
                return f"I do not know who {pred} {obj}."
            elif q_target == "property":
                clean_pred = (pred or "").replace("property_", "").replace("_property", "")
                if clean_pred and clean_pred not in ("property", "state", ""):
                    return f"I do not know the {clean_pred} of the {subj}."
                return f"I do not know what {self._format_noun(subj, title=False)} is."
            elif q_target == "identity":
                return f"I do not know who {subj} is."
            elif q_target == "verification":
                if pred in ("causes", "enables"):
                    return (
                        f"I do not have enough evidence to determine whether {subj} causes {obj}."
                    )
                elif pred == "requires":
                    return (
                        f"I do not have enough evidence to determine whether {subj} requires {obj}."
                    )
                elif pred in ("is_a", "type_of"):
                    obj_str = self._format_noun(obj, title=False)
                    return (
                        f"I do not have enough evidence to determine whether {subj} is {obj_str}."
                    )
                elif pred == "has_property" or pred.startswith("property_"):
                    return f"I do not have enough evidence to determine whether {subj} is {obj}."
                elif pred not in (
                    "located_on",
                    "located_in",
                    "located_at",
                    "below",
                    "above",
                    "near",
                    "located_under",
                    "located_beneath",
                    "located_above",
                    "located_near",
                    "located_beside",
                ):
                    tail = f" {obj}" if obj else ""
                    return f"I do not have enough evidence to determine whether {subj} {self._inflect_3s_simple(pred)}{tail}."
                return f"I do not have enough evidence to determine whether the {subj} is {prep} the {obj}."
            return f"I do not have enough evidence to determine the state of the {subj}."

        # 2. Contradiction
        if level == EpistemicVerbalizationLevel.CONTRADICTED:
            return f"There is conflicting evidence regarding the {subj}."

        # 2.5 SVO Q&A Answers ("What do florbs eat?", "Where do florbs live?", "Who was tired?")
        if q_target == "svo_object" and obj:
            subj_fmt = self._format_svo_subject(subj, original_frame)
            return f"{subj_fmt} {self._inflect_3s_simple(pred)} {obj}."

        if q_target == "svo_location" and obj:
            subj_fmt = self._format_svo_subject(subj, original_frame)
            svo_prep = "in"
            if epistemic_state.provenance.startswith("prep:"):
                svo_prep = epistemic_state.provenance.split(":", 1)[1] or "in"
            return f"{subj_fmt} {self._inflect_3s_simple(pred)} {svo_prep} {obj}."

        if q_target == "svo_subject" and subj:
            subj_fmt = subj.strip()
            if subj_fmt:
                subj_fmt = subj_fmt[0].upper() + subj_fmt[1:]
            if pred == "has_property":
                cop = (
                    "was" if (original_frame and original_frame.metadata.tense == "past") else "is"
                )
                return f"{subj_fmt} {cop} {obj}."
            if pred == "is_a":
                return f"{subj_fmt} is {self._format_noun(obj, title=False)}."
            if pred.startswith("located_") or pred in ("below", "above", "near"):
                return f"{subj_fmt} is {prep} {obj}."
            tense = original_frame.metadata.tense if original_frame else "present"
            if tense == "past":
                v_surf = (
                    pred + "d"
                    if pred.endswith("e")
                    else (
                        pred[:-1] + "ied" if pred.endswith("y") and len(pred) > 2 else pred + "ed"
                    )
                )
            else:
                v_surf = self._inflect_3s_simple(pred)
            return f"{subj_fmt} {v_surf} {obj}." if obj else f"{subj_fmt} {v_surf}."

        # 3. Verification (Yes/No answer)
        if q_target == "verification":
            if epistemic_state.raw_belief_value is True:
                if pred in ("is_a", "type_of"):
                    art_s = self._format_noun(subj, title=False)
                    art_o = self._format_noun(obj, title=False)
                    return f"Yes, {art_s} is {art_o}."
                elif pred == "has_property" or pred.startswith("property_"):
                    art_s = self._format_noun(subj, title=False)
                    return f"Yes, {art_s} is {obj}."
                elif pred == "causes":
                    return f"Yes, {subj} causes {obj}."
                elif pred == "enables":
                    return f"Yes, {subj} enables {obj}."
                elif pred == "requires":
                    art_o = self._format_noun(obj, title=False)
                    return f"Yes, {subj} requires {art_o}."
                elif pred not in (
                    "located_on",
                    "located_in",
                    "located_at",
                    "below",
                    "above",
                    "near",
                    "located_under",
                    "located_beneath",
                    "located_above",
                    "located_near",
                    "located_beside",
                ):
                    art_s = self._format_noun(subj, title=False)
                    tail = f" {obj}" if obj else ""
                    return f"Yes, {art_s} {self._inflect_3s_simple(pred)}{tail}."
                else:
                    if level == EpistemicVerbalizationLevel.CERTAIN:
                        return f"Yes, the {subj} is {prep} the {obj}."
                    elif level == EpistemicVerbalizationLevel.PROBABLE:
                        return f"Yes, the {subj} is probably {prep} the {obj}."
                    else:
                        return f"I think the {subj} may be {prep} the {obj}."
            else:
                if pred in ("is_a", "type_of"):
                    art_s = self._format_noun(subj, title=False)
                    art_o = self._format_noun(obj, title=False)
                    return f"No, {art_s} is not {art_o}."
                elif pred == "has_property" or pred.startswith("property_"):
                    art_s = self._format_noun(subj, title=False)
                    return f"No, {art_s} is not {obj}."
                elif pred == "causes":
                    return f"No, {subj} does not cause {obj}."
                elif pred == "enables":
                    return f"No, {subj} does not enable {obj}."
                elif pred == "requires":
                    art_o = self._format_noun(obj, title=False)
                    return f"No, {subj} does not require {art_o}."
                elif pred not in (
                    "located_on",
                    "located_in",
                    "located_at",
                    "below",
                    "above",
                    "near",
                    "located_under",
                    "located_beneath",
                    "located_above",
                    "located_near",
                    "located_beside",
                ):
                    art_s = self._format_noun(subj, title=False)
                    tail = f" {obj}" if obj else ""
                    return f"No, {art_s} does not {pred}{tail}."
                else:
                    return f"No, the {subj} is not {prep} the {obj}."

        # 4. Definition and taxonomic discourse answers
        if pred in ("definition", "is_a") and obj:
            # If obj is a multi-sentence explanation or paragraph, return it directly
            if ("." in obj or "\n" in obj) and len(obj) > 30:
                return obj
            obj_clean = obj.strip()
            if obj_clean.lower().startswith(
                (f"{subj.lower()} is ", f"a {subj.lower()} is ", f"an {subj.lower()} is ")
            ):
                return obj_clean if obj_clean.endswith(".") else f"{obj_clean}."
            subj_title = self._format_noun(subj, title=True)
            obj_fmt = self._format_noun(obj, title=False)
            if level == EpistemicVerbalizationLevel.CERTAIN:
                return f"{subj_title} is {obj_fmt}."
            elif level == EpistemicVerbalizationLevel.PROBABLE:
                return f"{subj_title} is generally described as {obj_fmt}."
            elif level == EpistemicVerbalizationLevel.PLAUSIBLE:
                return f"I believe {subj_title.lower()} is {obj_fmt}."
            else:
                return f"From what I know, {subj_title.lower()} may be described as {obj_fmt}."

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

    @staticmethod
    def _indefinite_article(word: str) -> str:
        """Return 'A' or 'An' for the given word (title-cased), or empty for proper/mass nouns."""
        if not word:
            return "A"
        clean = word.strip()
        lower = clean.lower()
        if (
            clean[0].isupper()
            or lower in ("gandalf", "darcy", "mr. darcy", "saruman")
            or lower in MASS_NOUNS
        ):
            return ""
        if lower[0] in "aeiou":
            return "An"
        return "A"

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
