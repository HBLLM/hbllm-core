"""
BabyAI Mission Parser — extracts structured semantic goals from instructions.

Supports standard BabyAI grammar (Level-1 GoToObj / PickupObj) and
provides seamless integration hooks for the multilingual bridge (English, Sinhala, Tamil).
"""

from __future__ import annotations

import re

from .types import BabyAIGoal

# Canonical English vocab
_EN_COLORS = {"red", "green", "blue", "purple", "yellow", "grey", "gray"}
_EN_OBJECTS_SPECIFIC = ["ball", "box", "key", "door"]
_EN_OBJECTS_GENERIC = ["object", "item"]
_EN_OBJECTS = set(_EN_OBJECTS_SPECIFIC) | set(_EN_OBJECTS_GENERIC)

# Sinhala lexicon
_SI_COLORS = {
    "රතු": "red",
    "කොළ": "green",
    "නිල්": "blue",
    "දම්": "purple",
    "කහ": "yellow",
    "අළු": "grey",
}
_SI_OBJECTS = {
    "බෝලය": "ball",
    "බෝල": "ball",
    "පෙට්ටිය": "box",
    "පෙට්ටි": "box",
    "යතුර": "key",
    "දොර": "door",
    "වස්තුව": "object",
    "වස්තුවක්": "object",
    "දෙයක්": "object",
}

# Tamil lexicon
_TA_COLORS = {
    "சிவப்பு": "red",
    "பச்சை": "green",
    "நீலம்": "blue",
    "ஊதா": "purple",
    "மஞ்சள்": "yellow",
    "சாம்பல்": "grey",
}
_TA_OBJECTS = {
    "பந்து": "ball",
    "பந்துக்கு": "ball",
    "பெட்டி": "box",
    "பெட்டியை": "box",
    "சாவி": "key",
    "கதவு": "door",
    "கதவை": "door",
    "பொருள்": "object",
    "பொருளை": "object",
}


class BabyAIMissionParser:
    """Parses natural language BabyAI instructions into typed BabyAIGoal specifications."""

    def parse(self, instruction: str) -> BabyAIGoal:
        """Parse instruction string into a BabyAIGoal."""
        cleaned = instruction.strip().lower()

        # Check Sinhala
        if any(c in cleaned for c in _SI_COLORS) or any(o in cleaned for o in _SI_OBJECTS):
            return self._parse_sinhala(cleaned, instruction)

        # Check Tamil
        if any(c in cleaned for c in _TA_COLORS) or any(o in cleaned for o in _TA_OBJECTS):
            return self._parse_tamil(cleaned, instruction)

        # Default: English
        return self._parse_english(cleaned, instruction)

    def _extract_entity_en(self, text: str) -> tuple[str, str | None]:
        found_color = None
        for color in _EN_COLORS:
            if re.search(rf"\b{color}\b", text):
                found_color = "grey" if color == "gray" else color
                break
        found_object = None
        for obj in _EN_OBJECTS_SPECIFIC:
            if re.search(rf"\b{obj}\b", text):
                found_object = obj
                break
        if found_object is None:
            for obj in _EN_OBJECTS_GENERIC:
                if re.search(rf"\b{obj}\b", text):
                    found_object = "object"
                    break
        if found_object is None:
            found_object = "ball"
        return found_object, found_color

    def _extract_entity_si(self, text: str) -> tuple[str, str | None]:
        found_color = None
        for si_col, en_col in _SI_COLORS.items():
            if si_col in text:
                found_color = en_col
                break
        found_object = "ball"
        for si_obj, en_obj in _SI_OBJECTS.items():
            if si_obj in text:
                found_object = en_obj
                break
        return found_object, found_color

    def _extract_entity_ta(self, text: str) -> tuple[str, str | None]:
        found_color = None
        for ta_col, en_col in _TA_COLORS.items():
            if ta_col in text:
                found_color = en_col
                break
        found_object = "ball"
        for ta_obj, en_obj in _TA_OBJECTS.items():
            if ta_obj in text:
                found_object = en_obj
                break
        return found_object, found_color

    def _extract_relative_loc_en(self, text: str) -> tuple[str, str | None]:
        rel = None
        cleaned = text
        for phrase, loc_name in [
            (" in front of you", "front"),
            (" behind you", "behind"),
            (" on your left", "left"),
            (" on your right", "right"),
        ]:
            if phrase in cleaned:
                rel = loc_name
                cleaned = cleaned.replace(phrase, "")
        return cleaned, rel

    def _parse_english(self, cleaned: str, raw: str) -> BabyAIGoal:
        # Check compound sequencing in English
        if " after you " in cleaned:
            parts = cleaned.split(" after you ", 1)
            # Causal prerequisite: parts[1] must precede parts[0]
            sub_prereq = self._parse_english(parts[1], parts[1])
            sub_action = self._parse_english(parts[0], parts[0])
            flat_subgoals = sub_prereq.flatten_subgoals() + sub_action.flatten_subgoals()
            return BabyAIGoal(
                action="sequence",
                subgoals=flat_subgoals,
                sequence_mode="after",
                language="en",
                raw_instruction=raw,
            )

        if ", then " in cleaned or " then " in cleaned:
            delim = ", then " if ", then " in cleaned else " then "
            parts = cleaned.split(delim)
            flat_subgoals: list[BabyAIGoal] = []
            for p in parts:
                flat_subgoals.extend(self._parse_english(p, p).flatten_subgoals())
            return BabyAIGoal(
                action="sequence",
                subgoals=flat_subgoals,
                sequence_mode="sequence",
                language="en",
                raw_instruction=raw,
            )

        if " and " in cleaned:
            parts = cleaned.split(" and ")
            flat_subgoals = []
            for p in parts:
                flat_subgoals.extend(self._parse_english(p, p).flatten_subgoals())
            return BabyAIGoal(
                action="sequence",
                subgoals=flat_subgoals,
                sequence_mode="and",
                language="en",
                raw_instruction=raw,
            )

        # Single clause: extract relative location if present
        cleaned, rel_loc = self._extract_relative_loc_en(cleaned)

        # Check PutNext
        if " next to " in cleaned:
            parts = cleaned.split(" next to ", 1)
            target_obj, target_col = self._extract_entity_en(parts[0])
            fixed_obj, fixed_col = self._extract_entity_en(parts[1])
            return BabyAIGoal(
                action="put_next",
                target_type=target_obj,
                target_color=target_col,
                fixed_type=fixed_obj,
                fixed_color=fixed_col,
                relative_loc=rel_loc,
                language="en",
                raw_instruction=raw,
            )

        # Determine intent / action
        if "pick up" in cleaned or "pickup" in cleaned:
            action = "pickup"
        elif "open" in cleaned or "unlock" in cleaned:
            action = "open"
        else:
            action = "go_to"

        found_object, found_color = self._extract_entity_en(cleaned)

        return BabyAIGoal(
            action=action,
            target_type=found_object,
            target_color=found_color,
            relative_loc=rel_loc,
            language="en",
            raw_instruction=raw,
        )

    def _parse_sinhala(self, cleaned: str, raw: str) -> BabyAIGoal:
        # Check compound sequencing in Sinhala
        if "පසුව" in cleaned:  # then
            parts = cleaned.split("පසුව")
            flat_subgoals: list[BabyAIGoal] = []
            for p in parts:
                flat_subgoals.extend(self._parse_sinhala(p, p).flatten_subgoals())
            return BabyAIGoal(
                action="sequence",
                subgoals=flat_subgoals,
                sequence_mode="sequence",
                language="si",
                raw_instruction=raw,
            )

        if " සහ " in cleaned:  # and
            parts = cleaned.split(" සහ ")
            flat_subgoals = []
            for p in parts:
                flat_subgoals.extend(self._parse_sinhala(p, p).flatten_subgoals())
            return BabyAIGoal(
                action="sequence",
                subgoals=flat_subgoals,
                sequence_mode="and",
                language="si",
                raw_instruction=raw,
            )

        # Check PutNext in Sinhala
        if ("ළඟින්" in cleaned or "අසල" in cleaned or "ළඟ" in cleaned) and (
            "තියන්න" in cleaned or "තබන්න" in cleaned
        ):
            splitter = "ළඟින්" if "ළඟින්" in cleaned else ("අසල" if "අසල" in cleaned else "ළඟ")
            parts = cleaned.split(splitter, 1)
            fixed_obj, fixed_col = self._extract_entity_si(parts[0])
            target_obj, target_col = self._extract_entity_si(parts[1])
            return BabyAIGoal(
                action="put_next",
                target_type=target_obj,
                target_color=target_col,
                fixed_type=fixed_obj,
                fixed_color=fixed_col,
                language="si",
                raw_instruction=raw,
            )

        if "ගන්න" in cleaned or "උස්සන්න" in cleaned:
            action = "pickup"
        elif "අරින්න" in cleaned or "හරින්න" in cleaned or "විවෘත" in cleaned or "අගුළු" in cleaned:
            action = "open"
        else:
            action = "go_to"

        found_object, found_color = self._extract_entity_si(cleaned)

        return BabyAIGoal(
            action=action,
            target_type=found_object,
            target_color=found_color,
            language="si",
            raw_instruction=raw,
        )

    def _parse_tamil(self, cleaned: str, raw: str) -> BabyAIGoal:
        # Check compound sequencing in Tamil
        if "பிறகு" in cleaned:  # then / after
            parts = cleaned.split("பிறகு")
            flat_subgoals: list[BabyAIGoal] = []
            for p in parts:
                flat_subgoals.extend(self._parse_tamil(p, p).flatten_subgoals())
            return BabyAIGoal(
                action="sequence",
                subgoals=flat_subgoals,
                sequence_mode="sequence",
                language="ta",
                raw_instruction=raw,
            )

        if " மற்றும் " in cleaned:  # and
            parts = cleaned.split(" மற்றும் ")
            flat_subgoals = []
            for p in parts:
                flat_subgoals.extend(self._parse_tamil(p, p).flatten_subgoals())
            return BabyAIGoal(
                action="sequence",
                subgoals=flat_subgoals,
                sequence_mode="and",
                language="ta",
                raw_instruction=raw,
            )

        # Check PutNext in Tamil
        if "அருகில்" in cleaned and ("வைக்கவும்" in cleaned or "வை" in cleaned):
            parts = cleaned.split("அருகில்", 1)
            fixed_obj, fixed_col = self._extract_entity_ta(parts[0])
            target_obj, target_col = self._extract_entity_ta(parts[1])
            return BabyAIGoal(
                action="put_next",
                target_type=target_obj,
                target_color=target_col,
                fixed_type=fixed_obj,
                fixed_color=fixed_col,
                language="ta",
                raw_instruction=raw,
            )

        if "எடுக்கவும்" in cleaned or "எடு" in cleaned:
            action = "pickup"
        elif "திறக்கவும்" in cleaned or "திற" in cleaned or "பூட்டு" in cleaned:
            action = "open"
        else:
            action = "go_to"

        found_object, found_color = self._extract_entity_ta(cleaned)

        return BabyAIGoal(
            action=action,
            target_type=found_object,
            target_color=found_color,
            language="ta",
            raw_instruction=raw,
        )
