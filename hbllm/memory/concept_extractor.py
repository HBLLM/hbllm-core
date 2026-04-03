"""
Concept Extractor — mines patterns from episodic memory to form abstract knowledge.

Pipeline:
1. Scan episodic memory for recurring patterns
2. Cluster related experiences into concepts
3. Extract generalized rules
4. Store as semantic memory entries

Example:
  20 questions about "Laravel queue" → concept: "Laravel Queue Management"
  → rule: "Queue workers require supervisor for production reliability"
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ExtractedConcept:
    """A concept extracted from recurring patterns."""
    concept_id: str
    name: str
    description: str
    frequency: int  # how many times the pattern appeared
    keywords: list[str]
    rules: list[str]  # generalized rules
    examples: list[str]  # example queries
    confidence: float  # 0-1
    created_at: float = field(default_factory=time.time)


class ConceptExtractor:
    """
    Extracts abstract concepts and rules from experience patterns.

    This is how the system moves from raw memory → structured knowledge.
    """

    def __init__(
        self,
        min_frequency: int = 3,
        min_keyword_count: int = 2,
        max_concepts: int = 100,
    ):
        self.min_frequency = min_frequency
        self.min_keyword_count = min_keyword_count
        self.max_concepts = max_concepts
        self._stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "and", "or", "but",
            "to", "in", "of", "for", "on", "with", "it", "this", "that", "be",
            "how", "what", "why", "when", "can", "do", "does", "did", "will",
            "would", "should", "could", "have", "has", "had", "not", "my", "i",
        }

    def extract_from_queries(self, queries: list[str]) -> list[ExtractedConcept]:
        """
        Extract concepts from a list of user queries.

        Groups queries by topic clusters and generates concepts.
        """
        if not queries:
            return []

        # Step 1: Extract keywords
        keyword_freq = self._count_keywords(queries)

        # Step 2: Find keyword clusters (co-occurring terms)
        clusters = self._cluster_keywords(queries, keyword_freq)

        # Step 3: Generate concepts from clusters
        concepts = []
        for i, (keywords, matched_queries) in enumerate(clusters):
            if len(matched_queries) < self.min_frequency:
                continue

            concept = ExtractedConcept(
                concept_id=f"concept_{int(time.time())}_{i}",
                name=self._generate_concept_name(keywords),
                description=f"Recurring topic with keywords: {', '.join(keywords)}",
                frequency=len(matched_queries),
                keywords=keywords,
                rules=self._extract_rules(matched_queries),
                examples=matched_queries[:5],
                confidence=min(1.0, len(matched_queries) / 20),
            )
            concepts.append(concept)

        concepts.sort(key=lambda c: c.frequency, reverse=True)
        return concepts[:self.max_concepts]

    def extract_from_qa_pairs(
        self, pairs: list[dict[str, str]],
    ) -> list[ExtractedConcept]:
        """
        Extract concepts from Q&A pairs (including answers).

        More informative than queries alone since answers contain knowledge.
        """
        queries = [p.get("question", p.get("query", "")) for p in pairs]
        concepts = self.extract_from_queries(queries)

        # Enrich rules with answer patterns
        for concept in concepts:
            relevant_pairs = [
                p for p in pairs
                if any(k in (p.get("question", "") + p.get("query", "")).lower()
                       for k in concept.keywords)
            ]
            answer_keywords = self._count_keywords([
                p.get("answer", p.get("response", "")) for p in relevant_pairs
            ])
            if answer_keywords:
                top_answer_terms = [k for k, _ in answer_keywords.most_common(5)]
                concept.rules.append(
                    f"Common answer themes: {', '.join(top_answer_terms)}"
                )

        return concepts

    # ─── Internals ───────────────────────────────────────────────────

    def _count_keywords(self, texts: list[str]) -> Counter:
        """Count significant keywords across texts."""
        keywords: Counter = Counter()
        for text in texts:
            words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            filtered = [w for w in words if w not in self._stopwords]
            keywords.update(set(filtered))  # count unique per text
        return keywords

    def _cluster_keywords(
        self, queries: list[str], keyword_freq: Counter,
    ) -> list[tuple[list[str], list[str]]]:
        """Cluster queries by co-occurring keywords."""
        # Get significant keywords
        significant = {
            k for k, v in keyword_freq.items()
            if v >= self.min_frequency
        }

        # Group queries by their significant keyword sets
        clusters: dict[frozenset[str], list[str]] = {}
        for query in queries:
            words = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
            matched = words & significant
            if len(matched) >= self.min_keyword_count:
                key = frozenset(list(matched)[:5])
                clusters.setdefault(key, []).append(query)

        # Merge overlapping clusters
        result = []
        for keywords, matched_queries in clusters.items():
            result.append((sorted(keywords), matched_queries))

        return result

    def _extract_rules(self, queries: list[str]) -> list[str]:
        """Extract generalized rules from a set of similar queries."""
        rules = []

        # If many questions → likely knowledge gap
        question_count = sum(1 for q in queries if "?" in q)
        if question_count > len(queries) * 0.7:
            rules.append("Users frequently ask questions about this topic")

        # If many "how to" queries → procedural knowledge needed
        howto_count = sum(1 for q in queries if "how" in q.lower())
        if howto_count > len(queries) * 0.3:
            rules.append("Topic requires procedural/how-to knowledge")

        # If many "error" or "fix" queries → troubleshooting topic
        error_count = sum(
            1 for q in queries
            if any(w in q.lower() for w in ["error", "fix", "bug", "issue", "problem"])
        )
        if error_count > len(queries) * 0.3:
            rules.append("Topic is frequently related to troubleshooting")

        return rules

    def _generate_concept_name(self, keywords: list[str]) -> str:
        """Generate a human-readable concept name."""
        return " ".join(w.capitalize() for w in keywords[:4])
