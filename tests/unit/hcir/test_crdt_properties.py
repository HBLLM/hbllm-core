"""Property-Based and Algebraic Convergence Tests for CRDT Merge in HCIRDelta.

Verifies:
1. Commutativity: delta_A.merge(delta_B) == delta_B.merge(delta_A)
2. Associativity: (delta_A.merge(delta_B)).merge(delta_C) == delta_A.merge(delta_B.merge(delta_C))
3. Idempotence: delta_A.merge(delta_A) == delta_A
4. Multi-Replica Eventual Consistency: Permuted delivery of N concurrent deltas converges to identical state
5. Deterministic Conflict Resolution: LWW by timestamp with device_id tiebreaker for concurrent node mutations
"""

from __future__ import annotations

import itertools
import random
from typing import Any

from hbllm.hcir.transactions import HCIRDelta


def _normalize_delta(delta: HCIRDelta) -> dict[str, Any]:
    """Helper to convert delta into canonical comparable form for algebraic equivalence."""
    return {
        "add_nodes": sorted(delta.add_nodes, key=lambda n: n.get("id", "")),
        "modify_nodes": sorted(
            delta.modify_nodes, key=lambda m: m.get("id") or m.get("node_id", "")
        ),
        "remove_node_ids": sorted(delta.remove_node_ids),
        "add_edges": sorted(delta.add_edges, key=lambda e: e.get("id", "")),
        "remove_edge_ids": sorted(delta.remove_edge_ids),
        "vector_clock": dict(sorted(delta.vector_clock.items())),
    }


def _make_sample_delta(
    device_id: str,
    clock: dict[str, int],
    timestamp: float,
    add_nodes: list[dict[str, Any]] | None = None,
    modify_nodes: list[dict[str, Any]] | None = None,
    remove_nodes: list[str] | None = None,
    add_edges: list[dict[str, Any]] | None = None,
    remove_edges: list[str] | None = None,
) -> HCIRDelta:
    return HCIRDelta(
        origin_device=device_id,
        vector_clock=clock,
        timestamp=timestamp,
        add_nodes=add_nodes or [],
        modify_nodes=modify_nodes or [],
        remove_node_ids=remove_nodes or [],
        add_edges=add_edges or [],
        remove_edge_ids=remove_edges or [],
    )


class TestCRDTAlgebraicProperties:
    """Mathematical invariant tests for HCIRDelta.merge()."""

    def test_idempotence(self) -> None:
        """Property: delta * delta == delta."""
        delta = _make_sample_delta(
            device_id="node_a",
            clock={"node_a": 2, "node_b": 1},
            timestamp=100.0,
            add_nodes=[{"id": "n1", "val": 10}, {"id": "n2", "val": 20}],
            modify_nodes=[{"id": "n0", "changes": {"status": "active"}}],
            remove_nodes=["old_node"],
            add_edges=[{"id": "e1", "src": "n1", "dst": "n2"}],
            remove_edges=["old_edge"],
        )

        merged = delta.merge(delta)
        assert _normalize_delta(merged) == _normalize_delta(delta)

    def test_commutativity_concurrent_disjoint(self) -> None:
        """Property: A * B == B * A for concurrent disjoint operations."""
        delta_a = _make_sample_delta(
            device_id="node_a",
            clock={"node_a": 3, "node_b": 1},
            timestamp=100.0,
            add_nodes=[{"id": "n1", "author": "a"}],
            add_edges=[{"id": "e1", "src": "n1", "dst": "n0"}],
            remove_nodes=["del_by_a"],
        )

        delta_b = _make_sample_delta(
            device_id="node_b",
            clock={"node_a": 1, "node_b": 3},
            timestamp=105.0,
            add_nodes=[{"id": "n2", "author": "b"}],
            add_edges=[{"id": "e2", "src": "n2", "dst": "n0"}],
            remove_nodes=["del_by_b"],
        )

        merged_ab = delta_a.merge(delta_b)
        merged_ba = delta_b.merge(delta_a)

        assert _normalize_delta(merged_ab) == _normalize_delta(merged_ba)

    def test_commutativity_with_modification_conflict_lww(self) -> None:
        """Property: A * B == B * A when both modify the same node (LWW wins)."""
        delta_a = _make_sample_delta(
            device_id="node_a",
            clock={"node_a": 2, "node_b": 1},
            timestamp=100.0,  # Earlier timestamp
            modify_nodes=[{"id": "target_node", "val": "from_a"}],
        )

        delta_b = _make_sample_delta(
            device_id="node_b",
            clock={"node_a": 1, "node_b": 2},
            timestamp=110.0,  # Later timestamp -> should win
            modify_nodes=[{"id": "target_node", "val": "from_b"}],
        )

        merged_ab = delta_a.merge(delta_b)
        merged_ba = delta_b.merge(delta_a)

        norm_ab = _normalize_delta(merged_ab)
        norm_ba = _normalize_delta(merged_ba)

        assert norm_ab == norm_ba
        assert norm_ab["modify_nodes"] == [{"id": "target_node", "val": "from_b"}]

    def test_commutativity_equal_timestamp_tiebreak(self) -> None:
        """Property: A * B == B * A when timestamps match exactly (device_id tiebreaker)."""
        delta_a = _make_sample_delta(
            device_id="node_alpha",
            clock={"node_alpha": 2, "node_beta": 1},
            timestamp=100.0,
            modify_nodes=[{"id": "conflicted", "val": "alpha_val"}],
        )

        delta_b = _make_sample_delta(
            device_id="node_beta",
            clock={"node_alpha": 1, "node_beta": 2},
            timestamp=100.0,  # Identical timestamp
            modify_nodes=[{"id": "conflicted", "val": "beta_val"}],
        )

        merged_ab = delta_a.merge(delta_b)
        merged_ba = delta_b.merge(delta_a)

        assert _normalize_delta(merged_ab) == _normalize_delta(merged_ba)

    def test_associativity(self) -> None:
        """Property: (A * B) * C == A * (B * C) for three concurrent deltas."""
        delta_a = _make_sample_delta(
            device_id="dev_1",
            clock={"dev_1": 2, "dev_2": 0, "dev_3": 0},
            timestamp=10.0,
            add_nodes=[{"id": "n1"}],
            modify_nodes=[{"id": "common", "step": 1}],
        )
        delta_b = _make_sample_delta(
            device_id="dev_2",
            clock={"dev_1": 0, "dev_2": 2, "dev_3": 0},
            timestamp=20.0,
            add_nodes=[{"id": "n2"}],
            modify_nodes=[{"id": "common", "step": 2}],
        )
        delta_c = _make_sample_delta(
            device_id="dev_3",
            clock={"dev_1": 0, "dev_2": 0, "dev_3": 2},
            timestamp=30.0,
            add_nodes=[{"id": "n3"}],
            remove_nodes=["old_x"],
        )

        ab_c = delta_a.merge(delta_b).merge(delta_c)
        a_bc = delta_a.merge(delta_b.merge(delta_c))

        assert _normalize_delta(ab_c) == _normalize_delta(a_bc)

    def test_multi_replica_interleaving_convergence(self) -> None:
        """Generates random permutations of concurrent deltas across 4 nodes and verifies convergence."""
        rng = random.Random(42)

        deltas = [
            _make_sample_delta(
                f"dev_{i}",
                {f"dev_{i}": 2},
                timestamp=100.0 + rng.uniform(0, 50),
                add_nodes=[{"id": f"node_{i}_{k}", "data": k} for k in range(3)],
                modify_nodes=[{"id": "shared_resource", "version": i}],
                remove_nodes=[f"purge_{i}"],
                add_edges=[{"id": f"edge_{i}"}],
            )
            for i in range(4)
        ]

        all_perms = list(itertools.permutations(deltas))
        canonical_state = None

        for perm in all_perms:
            acc = perm[0]
            for d in perm[1:]:
                acc = acc.merge(d)
            norm = _normalize_delta(acc)
            if canonical_state is None:
                canonical_state = norm
            else:
                assert norm == canonical_state, "Divergence found under permutation interleaving!"
