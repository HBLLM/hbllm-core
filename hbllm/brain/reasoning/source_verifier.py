"""
Source Verifier — credibility scoring engine for web research findings.

Evaluates the trustworthiness of web sources using a multi-factor scoring
model: domain reputation tiers, multi-source corroboration, and content
recency. Designed for standalone testability, independent of the bus.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class SourceCredibility:
    """Credibility assessment of a single web source."""

    url: str
    domain: str
    trust_score: float
    domain_reputation: float
    corroboration_score: float
    recency_score: float
    is_trusted: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "domain": self.domain,
            "trust_score": round(self.trust_score, 3),
            "domain_reputation": round(self.domain_reputation, 3),
            "corroboration_score": round(self.corroboration_score, 3),
            "recency_score": round(self.recency_score, 3),
            "is_trusted": self.is_trusted,
            "reasons": self.reasons,
        }


class SourceVerifier:
    """Multi-factor credibility scoring for web sources."""

    TIER_1_DOMAINS: set[str] = {
        "wikipedia.org",
        "python.org",
        "docs.python.org",
        "developer.mozilla.org",
        "learn.microsoft.com",
        "docs.aws.amazon.com",
        "cloud.google.com",
        "developer.apple.com",
        "docs.flutter.dev",
        "dart.dev",
        "laravel.com",
        "reactjs.org",
        "react.dev",
        "vuejs.org",
        "angular.io",
        "nodejs.org",
        "docs.rs",
        "go.dev",
        "pytorch.org",
        "tensorflow.org",
        "scikit-learn.org",
        "numpy.org",
        "arxiv.org",
        "nature.com",
        "w3.org",
        "rfc-editor.org",
        "nist.gov",
        "ncbi.nlm.nih.gov",
        "who.int",
        "nasa.gov",
    }

    TIER_2_DOMAINS: set[str] = {
        "stackoverflow.com",
        "github.com",
        "gitlab.com",
        "w3schools.com",
        "geeksforgeeks.org",
        "realpython.com",
        "digitalocean.com",
        "freecodecamp.org",
        "baeldung.com",
        "npmjs.com",
        "pypi.org",
        "pub.dev",
        "crates.io",
        "huggingface.co",
        "reddit.com",
    }

    TIER_3_DOMAINS: set[str] = {
        "medium.com",
        "dev.to",
        "hashnode.dev",
        "towardsdatascience.com",
        "substack.com",
        "wordpress.com",
        "quora.com",
    }

    WEIGHT_DOMAIN = 0.50
    WEIGHT_CORROBORATION = 0.30
    WEIGHT_RECENCY = 0.20

    def __init__(self, min_trust_score: float = 0.6):
        self.min_trust_score = min_trust_score

    def get_domain_reputation(self, url: str) -> tuple[float, str]:
        """Score domain reputation based on tier."""
        domain = self._extract_domain(url)
        for d in self.TIER_1_DOMAINS:
            if domain == d or domain.endswith("." + d):
                return 0.9, "tier_1_authoritative"
        for d in self.TIER_2_DOMAINS:
            if domain == d or domain.endswith("." + d):
                return 0.7, "tier_2_community"
        for d in self.TIER_3_DOMAINS:
            if domain == d or domain.endswith("." + d):
                return 0.4, "tier_3_blog"
        return 0.3, "tier_unknown"

    def compute_corroboration(
        self, claim: str, all_results: list[dict[str, Any]], exclude_url: str = ""
    ) -> float:
        """Score how well a claim is corroborated by other sources."""
        if len(all_results) <= 1:
            return 0.0
        claim_terms = {
            w.lower().strip(".,!?\"'()[]{}:;")
            for w in claim.split()
            if len(w.strip(".,!?\"'()[]{}:;")) > 4
        }
        if not claim_terms:
            return 0.0
        corroborating = 0
        others = 0
        for r in all_results:
            if r.get("url", "") == exclude_url:
                continue
            others += 1
            content = r.get("page_content", r.get("search_snippet", "")).lower()
            matches = sum(1 for t in claim_terms if t in content)
            if matches / len(claim_terms) >= 0.3:
                corroborating += 1
        if others == 0:
            return 0.0
        if corroborating >= 3:
            return 1.0
        elif corroborating >= 2:
            return 0.8
        elif corroborating >= 1:
            return 0.5
        return 0.0

    def compute_recency(self, content: str) -> float:
        """Estimate content recency from date patterns."""
        current_year = time.localtime().tm_year
        years = [int(y) for y in re.findall(r"\b(20[12]\d)\b", content)]
        if not years:
            return 0.5
        age = current_year - max(years)
        if age <= 0:
            return 1.0
        elif age == 1:
            return 0.8
        elif age <= 3:
            return 0.6
        elif age <= 5:
            return 0.4
        return 0.2

    def verify_source(
        self, result: dict[str, Any], all_results: list[dict[str, Any]]
    ) -> SourceCredibility:
        """Compute composite credibility for a single source."""
        url = result.get("url", "")
        content = result.get("page_content", result.get("search_snippet", ""))
        domain = self._extract_domain(url)
        domain_rep, tier = self.get_domain_reputation(url)
        corroboration = self.compute_corroboration(content, all_results, exclude_url=url)
        recency = self.compute_recency(content)
        trust_score = (
            self.WEIGHT_DOMAIN * domain_rep
            + self.WEIGHT_CORROBORATION * corroboration
            + self.WEIGHT_RECENCY * recency
        )
        reasons = [
            f"Domain tier: {tier} ({domain_rep:.1f})",
            f"Corroboration: {corroboration:.1f}",
            f"Recency: {recency:.1f}",
        ]
        return SourceCredibility(
            url=url,
            domain=domain,
            trust_score=trust_score,
            domain_reputation=domain_rep,
            corroboration_score=corroboration,
            recency_score=recency,
            is_trusted=trust_score >= self.min_trust_score,
            reasons=reasons,
        )

    def verify_sources(self, results: list[dict[str, Any]]) -> list[SourceCredibility]:
        """Verify all results, return sorted by trust_score desc."""
        assessments = [self.verify_source(r, results) for r in results]
        assessments.sort(key=lambda c: c.trust_score, reverse=True)
        return assessments

    def _extract_domain(self, url: str) -> str:
        """Extract root domain from URL."""
        try:
            domain = urlparse(url).netloc.lower()
            return domain[4:] if domain.startswith("www.") else domain
        except Exception:
            return ""
