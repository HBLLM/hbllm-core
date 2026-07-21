"""
Cognitive Consensus Engine — swarm conflict resolution & belief arbitration.

Resolves conflicting assertions across distributed swarm nodes or multi-source inputs:

    Conflicting Beliefs / Facts
             │
    CognitiveConsensusEngine
             ├── Provenance Reliability Weighting
             ├── UncertaintyVector Confidence Evaluation
             ├── Source Trust Score
             └── Temporal Freshness
             │
      Consensus Decision (Accepted Belief Transaction)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from hbllm.hcir.graph import BeliefNode
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.transactions import HCIRTransaction, TransactionOp, TransactionOperation
from hbllm.hcir.workspace import HCIRWorkspaceState

logger = logging.getLogger(__name__)


@dataclass
class CandidateBelief:
    """A candidate belief claim evaluated during consensus arbitration."""

    source_id: str
    belief_node: BeliefNode
    source_trust: float = 1.0  # 0.0 to 1.0 trust score of author
    provenance_reliability: float = 0.8  # Observed vs Inferred vs Reported
    received_timestamp: float = field(default_factory=time.time)

    def compute_consensus_score(self) -> float:
        """Compute arbitration score for consensus decision making.

        Higher score = higher priority for acceptance.
        """
        conf = self.belief_node.uncertainty.confidence
        rel = self.provenance_reliability
        trust = self.source_trust

        # Temporal decay: fresher knowledge scores slightly higher
        age_seconds = max(0.0, time.time() - self.received_timestamp)
        decay = 1.0 / (1.0 + age_seconds * 0.0001)

        return conf * rel * trust * decay


class CognitiveConsensusEngine:
    """Arbitrates conflicting assertions across swarm nodes or reasoning engines.

    Usage::

        engine = CognitiveConsensusEngine(workspace, tx_manager)
        consensus_belief = engine.arbitrate_beliefs(
            candidates=[c1, c2],
            claim_subject="greenhouse_humidity",
        )
    """

    def __init__(
        self,
        workspace: HCIRWorkspaceState,
        transaction_manager: TransactionManager,
    ) -> None:
        self._workspace = workspace
        self._tx_manager = transaction_manager

    def arbitrate_beliefs(
        self,
        candidates: list[CandidateBelief],
        author: str = "consensus_engine",
    ) -> BeliefNode | None:
        """Select the highest-scoring candidate belief and commit via transaction."""
        if not candidates:
            return None

        # Sort candidates by consensus score descending
        sorted_candidates = sorted(
            candidates, key=lambda c: c.compute_consensus_score(), reverse=True
        )
        winner = sorted_candidates[0]

        logger.info(
            "Consensus decision for claim '%s': selected source '%s' (score=%.3f)",
            winner.belief_node.claim,
            winner.source_id,
            winner.compute_consensus_score(),
        )

        # Commit winning belief to workspace graph via transaction
        tx = HCIRTransaction(
            author=author,
            operations=[
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=winner.belief_node.id,
                    node_data=winner.belief_node.model_dump(),
                )
            ],
        )

        res = self._tx_manager.commit(tx)
        if res.is_committed:
            return winner.belief_node
        return None
