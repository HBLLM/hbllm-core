"""
Semantic Contradiction & Antonym Utilities.

Single source of truth for structural, LLM-free contradiction detection across
epistemic and reasoning components:
- ContradictionEngine (epistemic loop proactive scanning)
- ContradictionOperator (classical reasoning operator)
- PredictionTracker (empirical prediction vs observation evaluation)

Provides bidirectional canonical antonym lookup, morphological affix negation
(un-, in-, dis-, non-, etc.), modal copula negations, and proposition-level
structural contradiction analysis.
"""

from __future__ import annotations

import re

# ── 1. Modal & Copular Negation Pairs ─────────────────────────────────────────

MODAL_NEGATION_PAIRS: list[tuple[str, str]] = [
    ("is ", "is not "),
    ("is ", "isn't "),
    ("was ", "was not "),
    ("was ", "wasn't "),
    ("are ", "are not "),
    ("are ", "aren't "),
    ("were ", "were not "),
    ("were ", "weren't "),
    ("can ", "cannot "),
    ("can ", "can't "),
    ("could ", "could not "),
    ("could ", "couldn't "),
    ("will ", "will not "),
    ("will ", "won't "),
    ("would ", "would not "),
    ("would ", "wouldn't "),
    ("should ", "should not "),
    ("should ", "shouldn't "),
    ("has ", "has no "),
    ("has ", "has not "),
    ("has ", "hasn't "),
    ("have ", "have no "),
    ("have ", "have not "),
    ("have ", "haven't "),
    ("does ", "does not "),
    ("does ", "doesn't "),
    ("do ", "do not "),
    ("do ", "don't "),
    ("did ", "did not "),
    ("did ", "didn't "),
    ("must ", "must not "),
    ("must ", "mustn't "),
]

# ── 2. Morphological Negation Affixes ──────────────────────────────────────────

# Prefixes that form direct opposites when prepended to an adjective/adverb/verb.
AFFIX_NEGATION_PREFIXES: tuple[str, ...] = (
    "un",
    "in",
    "im",
    "il",
    "ir",
    "dis",
    "mis",
    "non",
    "a",
)

# ── 3. Canonical Antonym Dictionary ───────────────────────────────────────────

_RAW_ANTONYMS: list[tuple[str, str]] = [
    # State & binary conditions
    ("locked", "unlocked"),
    ("open", "closed"),
    ("open", "shut"),
    ("on", "off"),
    ("true", "false"),
    ("yes", "no"),
    ("present", "absent"),
    ("empty", "full"),
    ("empty", "occupied"),
    ("active", "inactive"),
    ("enabled", "disabled"),
    ("connected", "disconnected"),
    ("online", "offline"),
    ("armed", "disarmed"),
    ("attended", "unattended"),
    ("plugged", "unplugged"),
    ("stable", "unstable"),
    ("safe", "unsafe"),
    ("safe", "dangerous"),
    ("safe", "hazardous"),
    ("clean", "dirty"),
    ("clean", "contaminated"),
    ("clear", "blocked"),
    ("clear", "occluded"),
    ("clear", "unclear"),
    ("visible", "invisible"),
    ("detected", "undetected"),
    ("known", "unknown"),
    ("authorized", "unauthorized"),
    ("allowed", "disallowed"),
    ("allowed", "prohibited"),
    ("permitted", "forbidden"),
    ("granted", "denied"),
    ("valid", "invalid"),
    ("correct", "incorrect"),
    ("correct", "wrong"),
    ("success", "failure"),
    ("success", "fail"),
    ("pass", "fail"),
    ("passed", "failed"),
    ("healthy", "unhealthy"),
    ("alive", "dead"),
    ("normal", "abnormal"),
    ("normal", "anomalous"),
    ("regular", "irregular"),
    ("consistent", "inconsistent"),
    ("aligned", "misaligned"),
    ("symmetric", "asymmetric"),
    ("possible", "impossible"),
    ("legal", "illegal"),
    # Quantitative & Directional polarities
    ("high", "low"),
    ("hot", "cold"),
    ("warm", "cool"),
    ("increase", "decrease"),
    ("increasing", "decreasing"),
    ("increased", "decreased"),
    ("rise", "fall"),
    ("rising", "falling"),
    ("rise", "drop"),
    ("rising", "dropping"),
    ("maximum", "minimum"),
    ("max", "min"),
    ("positive", "negative"),
    ("above", "below"),
    ("up", "down"),
    ("inside", "outside"),
    ("in", "out"),
    ("front", "back"),
    ("front", "rear"),
    ("left", "right"),
    ("start", "stop"),
    ("started", "stopped"),
    ("begin", "end"),
    ("accept", "reject"),
    ("accepted", "rejected"),
    ("light", "dark"),
    ("bright", "dim"),
]

# Build bidirectional lookup table
CANONICAL_ANTONYMS: dict[str, set[str]] = {}
for a, b in _RAW_ANTONYMS:
    CANONICAL_ANTONYMS.setdefault(a, set()).add(b)
    CANONICAL_ANTONYMS.setdefault(b, set()).add(a)


# ── 4. Utility Functions ──────────────────────────────────────────────────────


def _clean_token(token: str) -> str:
    """Normalize a token by stripping trailing punctuation and lowercasing."""
    return re.sub(r"[^\w\-]", "", token.strip().lower())


def are_antonyms(word_a: str, word_b: str) -> bool:
    """Check if two words are canonical or morphological antonyms.

    Supports:
    1. Direct canonical dictionary match (e.g. locked/unlocked, open/closed).
    2. Morphological prefix stripping (e.g. safe vs un-safe, active vs in-active).
    3. Hyphenated prefix forms (e.g. non-empty vs empty).
    """
    w_a = _clean_token(word_a)
    w_b = _clean_token(word_b)

    if not w_a or not w_b or w_a == w_b:
        return False

    # 1. Canonical antonym lookup
    if w_b in CANONICAL_ANTONYMS.get(w_a, set()):
        return True

    # 2. Hyphenated prefix match (e.g. "non-empty" vs "empty")
    if w_a.startswith("non-") and w_a[4:] == w_b:
        return True
    if w_b.startswith("non-") and w_b[4:] == w_a:
        return True

    # 3. Morphological prefix check
    for prefix in AFFIX_NEGATION_PREFIXES:
        if w_b.startswith(prefix) and len(w_b) > len(prefix) + 2:
            root_b = w_b[len(prefix) :]
            if root_b == w_a:
                return True
        if w_a.startswith(prefix) and len(w_a) > len(prefix) + 2:
            root_a = w_a[len(prefix) :]
            if root_a == w_b:
                return True

    return False


def extract_base_and_polar_tokens(claim: str) -> tuple[set[str], list[str]]:
    """Extract context/base tokens and state tokens from a claim string.

    Returns:
        (base_tokens, ordered_tokens)
    """
    raw_tokens = claim.lower().replace("-", " ").split()
    clean_tokens = [_clean_token(t) for t in raw_tokens if _clean_token(t)]
    return set(clean_tokens), clean_tokens


_STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "will",
    "shall",
    "to",
    "of",
    "in",
    "for",
    "on",
    "by",
    "at",
    "this",
    "that",
}


def detect_structural_contradiction(
    claim_a: str,
    claim_b: str,
) -> tuple[bool, str, float]:
    """Detect if claim_a and claim_b contradict each other without an LLM.

    Strategies evaluated in sequence:
    1. Modal / copular negation pattern checking
    2. Shared subject and canonical / affix antonym state conflict
    3. Direct polarity opposition for concise statements

    Returns:
        (is_contradiction, explanation, confidence)
    """
    c_a = claim_a.strip()
    c_b = claim_b.strip()

    if not c_a or not c_b or c_a.lower() == c_b.lower():
        return False, "", 0.0

    lower_a = c_a.lower()
    lower_b = c_b.lower()

    # ── Strategy 1: Modal / Copular Negation Patterns ─────────────────
    for pos, neg in MODAL_NEGATION_PAIRS:
        # Check claim_a has pos and claim_b has neg
        if pos in lower_a and neg in lower_b:
            base_a = lower_a.replace(pos, "", 1).strip()
            base_b = lower_b.replace(neg, "", 1).strip()
            if base_a and base_b and (base_a == base_b or base_a in base_b or base_b in base_a):
                return (
                    True,
                    f"Modal negation conflict: '{claim_a}' contradicts '{claim_b}'",
                    0.85,
                )
        # Check claim_a has neg and claim_b has pos
        if neg in lower_a and pos in lower_b:
            base_a = lower_a.replace(neg, "", 1).strip()
            base_b = lower_b.replace(pos, "", 1).strip()
            if base_a and base_b and (base_a == base_b or base_a in base_b or base_b in base_a):
                return (
                    True,
                    f"Modal negation conflict: '{claim_a}' contradicts '{claim_b}'",
                    0.85,
                )

    # ── Strategy 2: Subject & Antonym State Matching ───────────────────
    words_a = [_clean_token(w) for w in lower_a.split() if _clean_token(w)]
    words_b = [_clean_token(w) for w in lower_b.split() if _clean_token(w)]

    if not words_a or not words_b:
        return False, "", 0.0

    set_a = set(words_a)
    set_b = set(words_b)

    # Words unique to A and unique to B
    diff_a = [w for w in words_a if w not in set_b]
    diff_b = [w for w in words_b if w not in set_a]

    intersection = set_a & set_b
    if intersection:
        overlap_ratio = len(intersection) / max(len(set_a), len(set_b))

        # Check all candidate pairs in diff_a and diff_b
        for d_a in diff_a:
            for d_b in diff_b:
                if are_antonyms(d_a, d_b):
                    # Check whether any OTHER non-stopword tokens differ
                    other_diff_a = [w for w in diff_a if w != d_a and w not in _STOP_WORDS]
                    other_diff_b = [w for w in diff_b if w != d_b and w not in _STOP_WORDS]
                    if not other_diff_a and not other_diff_b:
                        severity = 0.95 if overlap_ratio >= 0.5 else 0.80
                        return (
                            True,
                            f"Antonym state conflict between '{d_a}' and '{d_b}' on shared subject",
                            severity,
                        )

    # ── Strategy 3: Concise Direct Polarity Opposition ────────────────
    # Only applies to very short phrases (e.g. "increase" vs "decrease")
    if len(words_a) <= 2 and len(words_b) <= 2:
        for w_a in words_a:
            for w_b in words_b:
                if are_antonyms(w_a, w_b):
                    return (
                        True,
                        f"Direct polarity opposition: '{w_a}' vs '{w_b}'",
                        0.90,
                    )

    return False, "", 0.0
