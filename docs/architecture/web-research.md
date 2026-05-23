# Web Research Architecture

## Overview

The HBLLM Web Research system enables autonomous knowledge acquisition from
the internet. When the system encounters a knowledge gap during reasoning,
it can search the web, verify source credibility, and ingest validated
findings — all without human intervention.

## Information vs Knowledge

The core design principle is that **not all web findings are equal**. The
system classifies every finding into one of three tiers:

| Tier | Name | Lifetime | Storage | Example |
|---|---|---|---|---|
| 🔵 T1 | Information | Single response | Workspace only | Weather, stock prices |
| 🟡 T2 | Task Knowledge | Session-scoped | Episodic memory | API docs, error fixes |
| 🟢 T3 | Core Knowledge | Long-term (30d TTL) | KnowledgeBase + KnowledgeGraph | How OAuth2 works |

### Classification Signals

- **T1 (Information)**: Time-sensitive, changes hourly, one-off queries
- **T2 (Task Knowledge)**: Useful for current task but may not be needed later
- **T3 (Core Knowledge)**: Foundational facts with cross-project reuse value

## Architecture

```
User Query
    │
    ▼
WorkspaceNode ──── low confidence ───► WebResearchNode
    │                                      │
    │                                      ├── BrowserNode (DuckDuckGo search)
    │                                      ├── SourceVerifier (credibility scoring)
    │                                      ├── Tier Classifier (T1/T2/T3)
    │                                      │
    │                           ┌──────────┼──────────┐
    │                           │          │          │
    │                        T1: Info   T2: Task   T3: Core
    │                           │          │          │
    │                      workspace   episodic    KnowledgeBase
    │                       thought    memory     + KnowledgeGraph
    │                           │          │          │
    ▼                           └──────────┴──────────┘
Response                                │
                                        ▼
                              SleepCycleNode
                                   │
                              ┌────┴────┐
                              │         │
                        Staleness    T2→T3
                         Audit     Promotion
```

## Components

### WebResearchNode (`hbllm/brain/web_research_node.py`)

Meta-cognitive node that orchestrates the full research pipeline:

1. **Gap Detection**: Subscribes to `workspace.thought` (low confidence <0.4)
   and `workspace.fallback` (errors)
2. **Search**: Sends structured queries to BrowserNode via `task.execute.search`
3. **Verify**: SourceVerifier scores credibility using domain reputation,
   multi-source corroboration, and content recency
4. **Classify**: Determines T1/T2/T3 tier using LLM or keyword heuristics
5. **Ingest**: Routes to correct storage based on tier

Bus topics:
- Subscribes: `workspace.thought`, `workspace.fallback`, `system.research.request`
- Publishes: `task.execute.search`, `memory.store`, `knowledge.ingest`, `workspace.thought`

### SourceVerifier (`hbllm/brain/source_verifier.py`)

Standalone credibility scoring engine with three factors:

- **Domain Reputation** (50% weight): Three tiers of trusted domains
  - Tier 1 (0.9): Official docs (python.org, MDN, Wikipedia)
  - Tier 2 (0.7): Community (StackOverflow, GitHub)
  - Tier 3 (0.4): Blogs (Medium, dev.to)
  - Unknown (0.3): Everything else
- **Corroboration** (30% weight): Same fact from 2+ independent sources
- **Recency** (20% weight): Content mentioning recent years scores higher

### Sleep Cycle Integration

Two new NREM sub-stages in SleepCycleNode:

- **Stage 1.8: Staleness Audit**: Scans web-sourced T3 entries past their
  30-day TTL and marks them obsolete
- **Stage 1.9: T2→T3 Promotion**: Promotes frequently-accessed Task Knowledge
  (3+ accesses across sessions) to permanent Core Knowledge

## Rate Limiting

- Per-tenant: 10 searches/hour (configurable)
- Per-topic: 5-minute cooldown to avoid re-searching the same query
- Budget-aware: SleepNode's staleness audit respects sleep budget
