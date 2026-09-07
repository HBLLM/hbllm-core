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
_EN_OBJECTS = {"ball", "box", "key", "door"}

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

    def _parse_english(self, cleaned: str, raw: str) -> BabyAIGoal:
        # Determine intent / action
        if "pick up" in cleaned or "pickup" in cleaned:
            action = "pickup"
        elif "open" in cleaned or "unlock" in cleaned:
            action = "open"
        else:
            action = "go_to"

        # Extract color
        found_color = None
        for color in _EN_COLORS:
            if re.search(rf"\b{color}\b", cleaned):
                found_color = "grey" if color == "gray" else color
                break

        # Extract target object
        found_object = "ball"  # default
        for obj in _EN_OBJECTS:
            if re.search(rf"\b{obj}\b", cleaned):
                found_object = obj
                break

        return BabyAIGoal(
            action=action,
            target_type=found_object,
            target_color=found_color,
            language="en",
            raw_instruction=raw,
        )

    def _parse_sinhala(self, cleaned: str, raw: str) -> BabyAIGoal:
        if "ගන්න" in cleaned or "උස්සන්න" in cleaned:
            action = "pickup"
        elif "අරින්න" in cleaned or "හරින්න" in cleaned or "විවෘත" in cleaned or "අගුළු" in cleaned:
            action = "open"
        else:
            action = "go_to"

        found_color = None
        for si_col, en_col in _SI_COLORS.items():
            if si_col in cleaned:
                found_color = en_col
                break

        found_object = "ball"
        for si_obj, en_obj in _SI_OBJECTS.items():
            if si_obj in cleaned:
                found_object = en_obj
                break

        return BabyAIGoal(
            action=action,
            target_type=found_object,
            target_color=found_color,
            language="si",
            raw_instruction=raw,
        )

    def _parse_tamil(self, cleaned: str, raw: str) -> BabyAIGoal:
        if "எடுக்கவும்" in cleaned or "எடு" in cleaned:
            action = "pickup"
        elif "திறக்கவும்" in cleaned or "திற" in cleaned or "பூட்டு" in cleaned:
            action = "open"
        else:
            action = "go_to"

        found_color = None
        for ta_col, en_col in _TA_COLORS.items():
            if ta_col in cleaned:
                found_color = en_col
                break

        found_object = "ball"
        for ta_obj, en_obj in _TA_OBJECTS.items():
            if ta_obj in cleaned:
                found_object = en_obj
                break

        return BabyAIGoal(
            action=action,
            target_type=found_object,
            target_color=found_color,
            language="ta",
            raw_instruction=raw,
        )
