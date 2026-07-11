"""
Failure Analyzer — Root Cause Analysis engine.

Sits between raw failure observation and ContradictionDetector.
Prevents false contradictions by classifying failures into categories:

    - WRONG_INPUT:     Wrong path, wrong parameter, typo
    - AUTH_FAILURE:     Authentication/authorization issue
    - TIMEOUT:         Network/resource timeout
    - RESOURCE_MISSING: File/API/service not available
    - LOGIC_ERROR:     Actual logic/reasoning error
    - STALE_KNOWLEDGE: Belief was correct before but world changed
    - TRUE_CONTRADICTION: Core belief was wrong — needs revision

Only TRUE_CONTRADICTION and STALE_KNOWLEDGE flow to ContradictionDetector.
Other categories are recorded as operational failures, not belief failures.

Example:
    Expected: API call success
    Actual:   401 Unauthorized
    Root cause: AUTH_FAILURE (not a belief error)

    Expected: File exists at /etc/config.yaml
    Actual:   File missing
    Root cause: WRONG_INPUT (path error, not a belief about file systems)

    Expected: PostgreSQL accepts connections on port 5432
    Actual:   Connection refused
    Root cause: Could be RESOURCE_MISSING or TRUE_CONTRADICTION
               (belief that PostgreSQL is running)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class FailureCategory(Enum):
    """Classification of failure root causes."""

    WRONG_INPUT = "wrong_input"
    AUTH_FAILURE = "auth_failure"
    TIMEOUT = "timeout"
    RESOURCE_MISSING = "resource_missing"
    LOGIC_ERROR = "logic_error"
    STALE_KNOWLEDGE = "stale_knowledge"
    TRUE_CONTRADICTION = "true_contradiction"
    UNKNOWN = "unknown"


# Categories that indicate a belief should be revised
BELIEF_RELEVANT_CATEGORIES = frozenset(
    {
        FailureCategory.TRUE_CONTRADICTION,
        FailureCategory.STALE_KNOWLEDGE,
        FailureCategory.LOGIC_ERROR,
    }
)


@dataclass
class RootCause:
    """Result of root cause analysis."""

    category: FailureCategory
    description: str
    expected: str
    actual: str
    affected_belief: str | None = None  # Which belief was wrong (if any)
    affected_mechanism_ids: list[str] = field(default_factory=list)
    confidence: float = 0.8  # How confident we are in this classification
    requires_belief_revision: bool = False
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def is_belief_error(self) -> bool:
        """Whether this failure indicates a belief was wrong."""
        return self.category in BELIEF_RELEVANT_CATEGORIES

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "description": self.description,
            "expected": self.expected,
            "actual": self.actual,
            "affected_belief": self.affected_belief,
            "affected_mechanism_ids": self.affected_mechanism_ids,
            "confidence": self.confidence,
            "requires_belief_revision": self.requires_belief_revision,
            "is_belief_error": self.is_belief_error,
        }


# ─── Pattern matchers for heuristic classification ──────────────────

_AUTH_PATTERNS = re.compile(
    r"(?i)(401|403|unauthorized|forbidden|permission denied|access denied"
    r"|authentication failed|invalid.*token|invalid.*key|invalid.*credentials)",
)

_TIMEOUT_PATTERNS = re.compile(
    r"(?i)(timeout|timed out|deadline exceeded|connection timed out"
    r"|read timed out|request timed out|gateway timeout|504)",
)

_MISSING_PATTERNS = re.compile(
    r"(?i)(not found|404|no such file|file not found|does not exist"
    r"|module not found|import error|command not found|connection refused"
    r"|no route to host|name resolution|dns)",
)

_INPUT_PATTERNS = re.compile(
    r"(?i)(invalid argument|type.?error|value.?error|syntax.?error"
    r"|malformed|unexpected token|parse error|invalid format"
    r"|missing required|wrong type|bad request|400)",
)


class FailureAnalyzer:
    """Identifies root cause from raw failure data.

    Uses heuristic pattern matching for classification — no LLM needed.
    This keeps it lightweight enough for the query-time path.

    For ambiguous cases, defaults to UNKNOWN with low confidence,
    allowing ContradictionDetector to investigate during sleep.
    """

    def __init__(self) -> None:
        self._analysis_count = 0
        self._category_counts: dict[str, int] = {}

    def analyze(
        self,
        expected: str,
        actual: str,
        error_message: str = "",
        context: dict[str, Any] | None = None,
        mechanism_ids: list[str] | None = None,
    ) -> RootCause:
        """Analyze a failure and classify its root cause.

        Args:
            expected: What was expected to happen
            actual: What actually happened
            error_message: Raw error message/traceback
            context: Additional context (skill name, domain, etc.)
            mechanism_ids: IDs of mechanisms involved in the failure

        Returns:
            RootCause with category, description, and whether belief revision is needed
        """
        self._analysis_count += 1
        ctx = context or {}
        mechs = mechanism_ids or []

        combined_text = f"{actual} {error_message}"

        # ── Heuristic classification (ordered by specificity) ────────

        # 1. Authentication/Authorization
        if _AUTH_PATTERNS.search(combined_text):
            root = RootCause(
                category=FailureCategory.AUTH_FAILURE,
                description="Authentication or authorization failure — not a belief error",
                expected=expected,
                actual=actual,
                confidence=0.9,
                requires_belief_revision=False,
                context=ctx,
                affected_mechanism_ids=mechs,
            )
            self._record(root)
            return root

        # 2. Timeout
        if _TIMEOUT_PATTERNS.search(combined_text):
            root = RootCause(
                category=FailureCategory.TIMEOUT,
                description="Operation timed out — transient failure, not a belief error",
                expected=expected,
                actual=actual,
                confidence=0.85,
                requires_belief_revision=False,
                context=ctx,
                affected_mechanism_ids=mechs,
            )
            self._record(root)
            return root

        # 3. Resource missing
        if _MISSING_PATTERNS.search(combined_text):
            # Could be stale knowledge (resource was there before)
            # or wrong input (path error). Check context.
            was_previously_successful = ctx.get("previous_success", False)
            if was_previously_successful:
                root = RootCause(
                    category=FailureCategory.STALE_KNOWLEDGE,
                    description=(
                        "Resource previously accessible but now missing — "
                        "world state may have changed"
                    ),
                    expected=expected,
                    actual=actual,
                    affected_belief=f"Resource availability: {expected}",
                    confidence=0.7,
                    requires_belief_revision=True,
                    context=ctx,
                    affected_mechanism_ids=mechs,
                )
            else:
                root = RootCause(
                    category=FailureCategory.RESOURCE_MISSING,
                    description="Resource not found — likely wrong path or missing dependency",
                    expected=expected,
                    actual=actual,
                    confidence=0.8,
                    requires_belief_revision=False,
                    context=ctx,
                    affected_mechanism_ids=mechs,
                )
            self._record(root)
            return root

        # 4. Input/syntax errors
        if _INPUT_PATTERNS.search(combined_text):
            root = RootCause(
                category=FailureCategory.WRONG_INPUT,
                description="Input validation or syntax error — operational issue, not belief error",
                expected=expected,
                actual=actual,
                confidence=0.85,
                requires_belief_revision=False,
                context=ctx,
                affected_mechanism_ids=mechs,
            )
            self._record(root)
            return root

        # 5. Logic error detection (expected vs actual semantic mismatch)
        if self._is_semantic_contradiction(expected, actual):
            root = RootCause(
                category=FailureCategory.TRUE_CONTRADICTION,
                description=(
                    "Expected outcome contradicts actual result — "
                    "underlying belief may be incorrect"
                ),
                expected=expected,
                actual=actual,
                affected_belief=f"Belief: {expected}",
                confidence=0.6,  # Lower confidence — needs investigation
                requires_belief_revision=True,
                context=ctx,
                affected_mechanism_ids=mechs,
            )
            self._record(root)
            return root

        # 6. Unknown — mark for sleep-time investigation
        root = RootCause(
            category=FailureCategory.UNKNOWN,
            description="Could not determine root cause from heuristics — queued for investigation",
            expected=expected,
            actual=actual,
            confidence=0.3,
            requires_belief_revision=False,  # Don't revise on low-confidence
            context=ctx,
            affected_mechanism_ids=mechs,
        )
        self._record(root)
        return root

    def _is_semantic_contradiction(self, expected: str, actual: str) -> bool:
        """Check if expected and actual are semantically contradictory.

        Lightweight heuristic — checks for negation patterns and
        boolean inversions without LLM.
        """
        expected_lower = expected.lower().strip()
        actual_lower = actual.lower().strip()

        # Direct boolean contradiction
        truth_pairs = [
            ("true", "false"),
            ("success", "failure"),
            ("pass", "fail"),
            ("exists", "missing"),
            ("found", "not found"),
            ("valid", "invalid"),
            ("correct", "incorrect"),
            ("enabled", "disabled"),
            ("running", "stopped"),
            ("connected", "disconnected"),
        ]

        for positive, negative in truth_pairs:
            if positive in expected_lower and negative in actual_lower:
                return True
            if negative in expected_lower and positive in actual_lower:
                return True

        # Negation insertion ("X works" vs "X does not work")
        if "not" in actual_lower and "not" not in expected_lower:
            # Check if removing "not" makes them similar
            stripped = actual_lower.replace(" not ", " ").replace("not ", "").replace(" not", "")
            overlap = len(set(stripped.split()) & set(expected_lower.split()))
            if overlap >= 2:
                return True

        return False

    def _record(self, root: RootCause) -> None:
        """Record analysis for internal stats."""
        cat = root.category.value
        self._category_counts[cat] = self._category_counts.get(cat, 0) + 1
        logger.info(
            "FailureAnalyzer: %s — %s (confidence=%.2f, belief_revision=%s)",
            cat,
            root.description[:80],
            root.confidence,
            root.requires_belief_revision,
        )

    def stats(self) -> dict[str, Any]:
        """Get analysis statistics."""
        return {
            "total_analyses": self._analysis_count,
            "by_category": dict(self._category_counts),
        }
