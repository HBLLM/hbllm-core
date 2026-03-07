# HBLLM Core — Code Review & Improvement Opportunities

A comprehensive audit of the existing codebase, organized by severity and module.

---

## 🔴 Critical Issues (Bugs / Memory Leaks)

### 1. Unbounded Blackboard Memory Leak — `workspace_node.py`
The `self.blackboards` dict is only cleaned up in `_commit_to_decision` and `_send_error_fallback`. If a simulation result (`handle_thought` with `simulation_result`) returns `SUCCESS` and `_commit_to_decision` throws before `finally`, or if no thoughts arrive AND the consensus watcher crashes unexpectedly, the board entry leaks forever.

**Fix:** Add a periodic TTL sweeper or use `WeakValueDictionary`. At minimum, add cleanup in `_finalize_board` after the simulation dispatch path (line 301 currently just `return`s without cleanup).

### 2. Unbounded `_query_cache` in `critic_node.py`
`_cache_query` appends every correlation_id → query text but never evicts old entries. In a long-running production system, this dict will grow indefinitely.

**Fix:** Use an LRU cache (e.g., `functools.lru_cache` or a simple `OrderedDict` with max size) or clean up entries after the board resolves.

### 3. Fire-and-Forget Task Without Error Handling — `memory_node.py:139`
```python
asyncio.create_task(
    asyncio.to_thread(self.semantic_db.store, content, {...})
)
```
This fire-and-forget task has no exception handler. If `semantic_db.store` fails, the exception is silently swallowed (Python warns about unhandled task exceptions but doesn't crash). This could cause silent data loss.

**Fix:** Wrap in a named async function with try/except, or use `task.add_done_callback()`.

---

## 🟠 Performance Issues

### 4. TF-IDF Re-encodes ALL Documents on Every Store — `semantic.py:169`
```python
all_texts = [d["content"] for d in self.documents] + [content]
all_vectors = self._tfidf.encode(all_texts)
```
Every call to `store()` re-encodes the entire document corpus. With 1000 documents, this is O(n²) over time.

**Fix:** Only re-encode when the vocabulary changes (new tokens detected). Cache the old vectors and only recompute the new document's vector, plus invalidate only affected dimensions.

### 5. Repeated `import asyncio` Inside Methods — `memory_node.py`
`import asyncio` appears at lines 71, 93, 138, 192 inside method bodies. While Python caches module imports, this is a code smell and adds unnecessary overhead on first call.

**Fix:** Move `import asyncio` to the top of the file with other imports.

### 6. Inline `__import__('time')` in Bus Dispatch — `bus.py:194,198`
```python
_start = __import__('time').monotonic()
latency = (__import__('time').monotonic() - _start) * 1000
```
Using `__import__` inline is an anti-pattern. It's slower than a top-level import and harder to read.

**Fix:** Import `time` at the top of the file (it's already imported in other modules).

### 7. `np.vstack` on Every Dense Store — `semantic.py:183`
```python
self.vectors = np.vstack((self.vectors, embedding))
```
`np.vstack` creates a full copy of the array every time. With 10,000 embeddings of dim 384, this copies ~15MB per insert.

**Fix:** Pre-allocate a larger array and track a fill pointer, or use a list of arrays and stack lazily before search.

---

## 🟡 Code Quality & Architecture

### 8. No Persistent Experience Log — `experience_node.py:53`
The comment says "In a real system, we would write to a persistent log/db here." Currently experiences are only published to the bus and lost if nothing is listening.

**Fix:** Write experiences to the episodic memory database or a dedicated experience log table.

### 9. Missing `try/except` on LLM Calls — `critic_node.py:71`, `router_node.py:71`
The CriticNode's `evaluate_thought` calls `self.llm.generate_json()` without any error handling. If the LLM provider times out or returns invalid JSON, the entire critic pipeline crashes.

**Fix:** Wrap all LLM calls in try/except and default to `PASS` on failure (fail-open for the critic, fail-safe for the decision node).

### 10. Hardcoded Domain List in Router — `router_node.py:74`
```python
f"Available domains: general, coding, math, planner, api_synth, fuzzy\n"
```
The domain list is hardcoded in the prompt string. When `SpawnerNode` creates new modules, the router doesn't know about them.

**Fix:** Build the domain list dynamically from the `ServiceRegistry` capabilities, or maintain a mutable `self.known_domains` set that spawner updates.

### 11. Decision Node Code Extraction is Fragile — `decision_node.py:102`
```python
code = content.split("```python")[1].split("```")[0].strip()
```
This will crash with `IndexError` if the content doesn't contain exactly the expected markdown code fence format.

**Fix:** Use a regex with a fallback, or wrap in try/except.

### 12. Workspace `board["resolved"] = True` Race Condition — `workspace_node.py:258`
Between checking `board["resolved"]` and setting it on line 258, another concurrent handler could also enter `_finalize_board`. While unlikely with asyncio's single-threaded nature, the pattern is fragile if ever moved to a threaded environment.

**Fix:** Use `asyncio.Lock` per board, or use an atomic compare-and-swap pattern.

### 13. `MemoryNode.__init__` Assumes Writable Parent Directory — `memory_node.py:33`
```python
self.procedural_db = ProceduralMemory(Path(db_path).parent / "procedural_memory.db")
```
If `db_path` is just `"working_memory.db"` (no directory), `Path(db_path).parent` resolves to `.` which works but is implicit. If run from a read-only directory, this silently fails.

**Fix:** Use a configurable memory directory with explicit `os.makedirs(exist_ok=True)`.

---

## 🔵 Missing Functionality / Enhancements

### 14. No Message Priority Handling in Bus
The `Message` model has a `priority` field and a `Priority` enum, but `InProcessBus` uses a FIFO `asyncio.Queue` that ignores priority entirely.

**Fix:** Replace `asyncio.Queue` with `asyncio.PriorityQueue` and sort by `message.priority`.

### 15. No TTL Enforcement on Messages
The `Message` model has `ttl_seconds` but it's never checked. Expired messages are delivered normally.

**Fix:** Check TTL in `_dispatch_to_subscribers` and drop expired messages.

### 16. No Graceful Shutdown for WorkspaceNode
When `Brain.shutdown()` is called, active blackboard sessions are abandoned without sending any response to the user.

**Fix:** In `WorkspaceNode.on_stop()`, iterate over active blackboards and send error fallback messages.

### 17. Factory Doesn't Wire Optional Nodes
`BrainFactory.create()` conditionally checks `cfg.inject_memory`, `cfg.inject_identity`, `cfg.inject_curiosity` but these flags are only used in the `PipelineConfig`, not in node creation. The `MemoryNode`, `IdentityNode`, etc. are never actually created in the factory.

**Fix:** Conditionally create and wire `MemoryNode`, `IdentityNode`, and `CuriosityNode` based on config flags.

### 18. No Backpressure on Bus Queue
If producers flood the bus faster than consumers can process, `asyncio.Queue(maxsize=1000)` will block the publisher on `await self._queue.put()`. There's no monitoring or alerting for queue saturation.

**Fix:** Add a queue fullness metric and optional overflow policy (drop oldest, raise, etc.).

### 19. ExperienceNode Only Listens to `sensory.output`
It only records final outputs, missing the entire reasoning process (workspace thoughts, critic evaluations, planning steps). This limits the learning signal.

**Fix:** Subscribe to additional topics like `workspace.thought` and `decision.evaluate` to build a richer experience trace.

### 20. No Deduplication in Semantic Memory
If the same content is stored multiple times (e.g., repeated queries), it creates duplicate entries that dilute search results.

**Fix:** Add content hashing and skip storage if hash already exists, or merge metadata.

---

## Summary Table

| #   | Severity   | Module          | Issue                      |
| --- | ---------- | --------------- | -------------------------- |
| 1   | 🔴 Critical | workspace_node  | Blackboard memory leak     |
| 2   | 🔴 Critical | critic_node     | Unbounded query cache      |
| 3   | 🔴 Critical | memory_node     | Silent task failure        |
| 4   | 🟠 Perf     | semantic        | O(n²) TF-IDF re-encoding   |
| 5   | 🟠 Perf     | memory_node     | Repeated inline imports    |
| 6   | 🟠 Perf     | bus             | Inline `__import__` calls  |
| 7   | 🟠 Perf     | semantic        | `np.vstack` copying        |
| 8   | 🟡 Quality  | experience_node | No persistent log          |
| 9   | 🟡 Quality  | critic/router   | Missing LLM error handling |
| 10  | 🟡 Quality  | router_node     | Hardcoded domain list      |
| 11  | 🟡 Quality  | decision_node   | Fragile code extraction    |
| 12  | 🟡 Quality  | workspace_node  | Potential race condition   |
| 13  | 🟡 Quality  | memory_node     | Implicit path resolution   |
| 14  | 🔵 Feature  | bus             | Priority queue unused      |
| 15  | 🔵 Feature  | bus             | TTL not enforced           |
| 16  | 🔵 Feature  | workspace_node  | No graceful shutdown       |
| 17  | 🔵 Feature  | factory         | Optional nodes not wired   |
| 18  | 🔵 Feature  | bus             | No backpressure monitoring |
| 19  | 🔵 Feature  | experience_node | Limited observation scope  |
| 20  | 🔵 Feature  | semantic        | No deduplication           |
