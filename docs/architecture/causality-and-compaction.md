# Causal Cognition and Memory Compaction

To operate continuously over long periods, the agent must be able to reason about cause and effect while ruthlessly compressing historical data. Without compaction, cognitive systems become sluggish as their memory graphs expand infinitely.

## Causal Cognition Graph (`causal_graph.py`)

Rather than merely recording a chronological `EventLog`, the system infers *why* things happened.
- The **CausalGraph** evaluates relationships between events using a probabilistic scoring function: `f(temporal_distance, source_trust, event_match, state_alignment, intervention_signal_strength)`.
- It connects events via `CausalLink` edges that denote the probability of causality.
- **Hallucination Thresholding**: Discards weak or false links below a calculated confidence score, ensuring the agent doesn't establish superstitious beliefs about its actions.

## Single-Query Trace API (`tracer.py`)

A fundamental requirement for autonomous agents is auditability.
- The `DecisionTraceLedger` maintains a full history of cognitive state changes.
- The `explain_decision(trace_id)` API answers "Why did you do this?" by merging planner intent logic and causal relationships, returning an answer in under 5 seconds.

## Cognitive Compaction Engine (`engine.py`)

Long-term memory must be heavily compressed to maintain low latency.
- **GraphDelta & UtilityDelta**: Replaces full-state snapshots with event-sourced deltas, massively reducing storage overhead.
- **CognitiveEntropyEngine**: Tracks graph density, causal drift, and stale nodes. It serves as a metric for systemic "mental health" to trigger compaction cycles.
- **MemoryImportanceScorer**: Ranks the importance of memory nodes based on emotion, network centrality, user relevance, and novelty.
- **Attention-Based Forgetting**: Memories that aren't activated gradually decay and are folded.
- **Multi-Stage Semantic Folding**: Compresses old data by clustering heuristically related nodes and then using an LLM to assign them a high-level semantic label, preserving the essence while deleting the raw events.
