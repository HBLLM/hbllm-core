"""Recursive Value Induction and Causal Credit Propagation — HCIR §6 / AGI Gap 5.

Biologically grounded in dopamine credit assignment (Schultz 1998) and hierarchical
reinforcement learning value composition (Botvinick 2009; Sutton, Precup, Singh 1999):
1. Causal Enablement Graph:
   - Tracks prerequisite and enablement relations between environmental entities
     and state transitions (e.g., key enables opening door; switch enables opening gate).
2. Backward Value Induction:
   - Recursively propagates terminal extrinsic reward backwards through causal
     dependency chains:
         V(u) = max(V_terminal(u), gamma * max_{v in Enables(u)} V(v))
   - Enables the agent to autonomously discover that intermediate subgoals (keys,
     switches, bridges) are intrinsically valuable because they causally unlock
     terminal goals, without hardcoded heuristic recipes.
3. Transactional Graph Export:
   - Serializes induced values into HCIR CausalPredicates and ConceptNode properties
     for use across cortical planning, MCTS rollouts, and goal arbitration.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class CausalEnablementLink:
    """A directed causal link where interaction with source enables progress toward target."""

    source_key: str
    target_key: str
    confidence: float = 1.0
    support_count: int = 1
    timestamp: float = field(default_factory=time.time)

    def to_causal_predicate(self) -> CausalPredicate:
        return CausalPredicate(
            variable=f"causal_enablement:{self.source_key}->{self.target_key}",
            operator="==",
            value=f"conf:{self.confidence:.2f}:supp:{self.support_count}",
            relation="enables",
            target_entity=self.target_key,
        )


class RecursiveValueInducer:
    """Induces intermediate entity values by propagating rewards backwards over causal chains."""

    def __init__(self, discount_factor: float = 0.90) -> None:
        self.discount_factor = max(0.01, min(0.99, float(discount_factor)))
        # Canonical terminal rewards (e.g. goal=1.0, exit=1.0)
        self.terminal_values: dict[str, float] = {}
        # source -> set of target keys it enables
        self.enablement_graph: dict[str, set[str]] = {}
        # Link metadata
        self.links: dict[tuple[str, str], CausalEnablementLink] = {}
        # Calculated induced values
        self.induced_values: dict[str, float] = {}

    def _normalize_key(self, key: Any) -> str:
        if isinstance(key, (tuple, list)):
            return f"pos_{key[0]}_{key[1]}"
        return str(key).lower()

    def register_terminal_value(self, entity_key: Any, value: float = 1.0) -> None:
        """Register a terminal reward attractor."""
        norm_key = self._normalize_key(entity_key)
        self.terminal_values[norm_key] = max(0.0, float(value))
        self.induced_values[norm_key] = max(self.induced_values.get(norm_key, 0.0), float(value))
        self.propagate_values()

    def record_causal_enablement(
        self,
        source_key: Any,
        target_key: Any,
        confidence: float = 1.0,
    ) -> CausalEnablementLink:
        """Record that source causally enables or unlocks target."""
        s = self._normalize_key(source_key)
        t = self._normalize_key(target_key)

        if s not in self.enablement_graph:
            self.enablement_graph[s] = set()
        self.enablement_graph[s].add(t)

        link_key = (s, t)
        if link_key in self.links:
            link = self.links[link_key]
            link.support_count += 1
            link.confidence = max(link.confidence, float(confidence))
            link.timestamp = time.time()
        else:
            link = CausalEnablementLink(
                source_key=s,
                target_key=t,
                confidence=float(confidence),
                support_count=1,
            )
            self.links[link_key] = link

        self.propagate_values()
        return link

    def propagate_values(
        self,
        max_iterations: int = 15,
        convergence_eps: float = 1e-4,
    ) -> dict[str, float]:
        """Perform Bellman-style value iteration over the causal enablement DAG."""
        # Initialize with terminal values
        values: dict[str, float] = dict(self.terminal_values)

        # Collect all active nodes
        all_nodes: set[str] = set(self.terminal_values.keys())
        for s, targets in self.enablement_graph.items():
            all_nodes.add(s)
            all_nodes.update(targets)

        for _ in range(max_iterations):
            max_delta = 0.0
            for u in all_nodes:
                old_val = values.get(u, 0.0)
                term_val = self.terminal_values.get(u, 0.0)

                # Recursive enablement backup
                targets = self.enablement_graph.get(u, set())
                max_target_val = 0.0
                for v in targets:
                    link = self.links.get((u, v))
                    conf = link.confidence if link else 1.0
                    v_val = values.get(v, 0.0) * conf
                    if v_val > max_target_val:
                        max_target_val = v_val

                backed_up = self.discount_factor * max_target_val
                new_val = max(term_val, backed_up)

                delta = abs(new_val - old_val)
                if delta > max_delta:
                    max_delta = delta
                values[u] = new_val

            if max_delta < convergence_eps:
                break

        self.induced_values = values
        return self.induced_values

    def get_induced_value(self, entity_key: Any) -> float:
        """Get the recursive induced value for an entity or coordinate."""
        norm_key = self._normalize_key(entity_key)
        return self.induced_values.get(norm_key, 0.0)

    def to_causal_predicates(self) -> list[CausalPredicate]:
        """Export all induced values as declarative HCIR CausalPredicates."""
        predicates: list[CausalPredicate] = []
        for k, v in self.induced_values.items():
            if v > 0.01:
                predicates.append(
                    CausalPredicate(
                        variable=f"induced_value:{k}",
                        operator="==",
                        value=float(round(v, 4)),
                        relation="recursive_valuation",
                        target_entity=k,
                    )
                )
        return predicates

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Propose atomic transaction operations reflecting induced value state."""
        ops: list[TransactionOperation] = []
        for k, v in self.induced_values.items():
            if v > 0.01:
                ops.append(
                    TransactionOperation(
                        op=TransactionOp.UPSERT_NODE,
                        node_id=f"val_{k}",
                        node_data={
                            "entity_key": k,
                            "induced_value": float(v),
                            "is_terminal": k in self.terminal_values,
                            "timestamp": time.time(),
                        },
                    )
                )
        return ops

    def to_dict(self) -> dict[str, Any]:
        return {
            "discount_factor": self.discount_factor,
            "terminal_values": self.terminal_values,
            "enablement_graph": {k: list(v) for k, v in self.enablement_graph.items()},
            "induced_values": self.induced_values,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecursiveValueInducer:
        inst = cls(discount_factor=float(data.get("discount_factor", 0.90)))
        inst.terminal_values = {k: float(v) for k, v in data.get("terminal_values", {}).items()}
        inst.enablement_graph = {k: set(v) for k, v in data.get("enablement_graph", {}).items()}
        inst.induced_values = {k: float(v) for k, v in data.get("induced_values", {}).items()}
        return inst
