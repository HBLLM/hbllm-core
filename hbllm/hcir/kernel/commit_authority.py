from __future__ import annotations

"""Per-tier commit authority (WP6.4).

Grants are opaque kernel-issued capabilities: ``grant_id`` is a random
``secrets.token_hex(16)`` minted only by :class:`CommitAuthorityRegistry`.
A transaction carries the grant id in ``metadata["commit_grant"]``; the
ConstitutionalVerifier asks the registry whether that id authorizes the
workspace's tier. Self-declared author strings are not consulted.
"""

import secrets
from collections.abc import Iterable
from dataclasses import dataclass, field

from hbllm.hcir.tier_types import WorkspaceTier


@dataclass(frozen=True)
class CommitGrant:
    """Immutable commit capability issued by the authority registry."""

    grant_id: str  # secrets.token_hex(16), issued only by the registry
    holder: str  # stage/subsystem name, for audit
    tiers: frozenset[WorkspaceTier]


@dataclass
class CommitAuthorityRegistry:
    """Issues, revokes, and adjudicates per-tier commit grants."""

    _grants: dict[str, CommitGrant] = field(default_factory=dict)

    def issue(self, holder: str, tiers: Iterable[WorkspaceTier]) -> CommitGrant:
        grant = CommitGrant(
            grant_id=secrets.token_hex(16),
            holder=holder,
            tiers=frozenset(tiers),
        )
        self._grants[grant.grant_id] = grant
        return grant

    def revoke(self, grant_id: str) -> None:
        self._grants.pop(grant_id, None)

    def authorizes(self, grant_id: str | None, tier: WorkspaceTier) -> bool:
        """WORKING is always open; other tiers require a live grant."""
        if tier == WorkspaceTier.WORKING:
            return True
        if grant_id is None:
            return False
        grant = self._grants.get(grant_id)
        return grant is not None and tier in grant.tiers
