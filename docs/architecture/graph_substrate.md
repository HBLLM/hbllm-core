# Graph Substrate Architecture (WP4)

## Decision

`CognitiveGraph` keeps the **Python dict-of-nodes as the single source of truth**.
The Rust chunked structural-sharing graph (`hbllm_hcir_graph`, via
`NativeGraphAdapter`) is a **verified accelerator** only: every mutation is
mirrored into the adapter, and adapter state never feeds back into Python state
(algorithm results are routed through it when present).

## Divergence semantics

Native mirroring is fail-loud, never silent. Any exception while syncing a
mutation increments `CognitiveGraph._native_divergence_count`, stores the error
in `_native_last_error`, and raises `SubstrateDivergenceError` — results stay
correct because Python state is authoritative. `CognitiveGraph.native_health()`
returns `{"enabled", "available", "divergences", "last_error"}` and is surfaced
through `KernelTelemetry.health_probe` → `snapshot()["graph_native_health"]`.
Adapter init failure logs WARNING once per process when native was explicitly
requested, `logger.debug` otherwise.

## Fork / diff design

`CognitiveGraph.fork()` is structural-sharing copy-on-write:

- `_nodes`/`_edges` get cheap `dict.copy()` snapshots.
- The seven secondary indexes + digest tables are *shared* between parent and
  fork; both sides set `_cow_indexes`, and the first mutator call on either side
  (`_materialize_indexes()`) lazily deep-copies them.
- The fork records `_dirty_nodes`/`_dirty_edges` (ids added/upserted/removed
  since fork), `_fork_parent`, and `_fork_base_version` (stamped by
  `HCIRWorkspaceState.fork`).

`HCIRWorkspaceState.diff_branch()` uses the branch's dirty sets when the branch
graph was forked from this workspace (`_fork_parent is main_g`), restricting the
comparison to dirty ids — identical output to the full `model_dump()` compare at
O(dirty) cost. Non-fork branches fall back to the full comparison.

## Parity tests

- `tests/unit/hcir/test_branch_diff_parity.py`: seeded random op sequences
  (≥200 ops, adds/upserts/removes of nodes & edges incl. add-then-remove)
  asserting the dirty-set diff equals the full-comparison diff, plus a
  `merge_branch` unchanged-result test.
- `tests/unit/hcir/test_native_parity_matrix.py`: 5 seeds × native
  enabled/disabled variants (enabled variant skips when the native module is
  unavailable).
