---
title: "Sleep Cycle & Continuous Learning"
description: "How Human Brain LLM (HBLLM) uses biologically inspired sleep cycles to consolidate memory, prune noise, and strengthen neural patterns across sessions."
---

# Sleep Cycle & Continuous Learning

In the **Human Brain LLM (HBLLM)** architecture, the **Sleep Cycle** is not just a background task — it is the fundamental mechanism for **Long-term Potentiation** and model refinement. 

Unlike stateless LLMs that forget every interaction after the session ends, HBLLM uses designated "idle periods" to consolidate information, refine its reasoning, and grow its intelligence.

---

## Biological Inspiration: The 3-Phase Consolidation

Human brains use sleep to move information from short-term hippocampus storage to long-term cortical networks. HBLLM mirrors this with a **3-Phase Sleep Cycle** orchestrated by the `SleepCycleNode`.

### Phase 1: Memory Replay & Consolidation
**Code:** `_consolidate_memory()`

During the day, the **Episodic Memory** (System 2) records raw interaction logs. During sleep:
1.  **Selective Replay:** The system re-reads the most salient dialogue turns from the session.
2.  **Semantic Synthesis:** The 1.5B backbone synthesizes these into high-level **Semantic Summaries**.
3.  **GraphRAG Clustering:** Entities and relations are extracted and clustered into thematic **Communities** in the Knowledge Graph.
4.  **Synaptic Pruning:** Low-salience or redundant logs are archived, keeping the live vector space efficient.

### Phase 2: Artificial Neuroplasticity (Continuous DPO)
**Code:** `_run_self_improvement()`

This is where the brain's weights actually change.
- **DPO Queue:** Feedback from the `CriticNode` and `ValueMemory` is collected into an atomic JSON queue throughout the day.
- **Contrastive Learning:** The `LearnerNode` triggers a **Direct Preference Optimization (DPO)** loop, penalizing "rejected" (criticized) paths and strengthening "chosen" (validated) paths.
- **Read-Only Preservation (Zero Catastrophic Forgetting):** To prevent breaking the core model, the base model and *all downloaded domain adapters* are kept strictly **read-only**.
- **Isolation:** DPO training happens locally and exclusively on a dynamically created `personalization` adapter, ensuring that private learning never leaks to other users or corrupts established domains.

### Phase 3: Curiosity-Driven Exploration
**Code:** `_replay_curiosity_goals()`

If the `CuriosityNode` identified knowledge gaps or "exploratory goals" during the day, they are replayed here.
- The system generates research queries based on these gaps.
- It "imagines" potential scenarios or searches its internal Knowledge Graph to bridge conceptual distances.
- Resulting insights are stored back in **Semantic Memory**, ready for the next active session.

---

## Execution & Triggers

The Sleep Cycle can be triggered in two ways:

### 1. Auto-Trigger (Idle Timeout)
The `SleepCycleNode` monitors the message bus. If no `router.query` is detected for a configurable duration (default: 6 hours), the system automatically enters deep sleep.
- **Wake-on-Activity:** If a user query arrives during sleep, the system immediately aborts the cycle and wakes up to provide sub-millisecond response time.

### 2. Manual Trigger (API)
For SaaS platforms, you can manually trigger a consolidation cycle for a specific tenant:

```bash
# Example REST API Trigger
curl -X POST "https://api.hbllm.ai/v1/system/sleep" \
     -H "Authorization: Bearer <TOKEN>" \
     -d '{"tenant_id": "tenant-001", "mode": "deep"}'
```

---

## Architectural Advantages

| Feature | Monolithic LLM (OpenAI) | HBLLM (Human Brain) |
|---|---|---|
| **Memory** | Stateless / External RAG | **Integrated Consolidation** |
| **Learning** | Manual Fine-Tuning | **Autonomous Continuous DPO** |
| **Hardware** | Constant 80GB+ VRAM | **Low-VRAM (Inference) / Batch (Sleep)** |
| **Privacy** | Shared Global Weights | **Per-Tenant Neural Isolation** |
| **Evolution** | Static until next release | **Grows smarter every sleep cycle** |

---

## Performance Considerations

- **VRAM/CPU Offset:** Consolidation is a batch process. It is configured to run at low priority (nice level) to ensure it doesn't starve the host OS of resources.
- **Incremental:** The system only processes *new* memories since the last sleep cycle, keeping each cycle duration predictable.
- **Safety:** The `SentinelNode` monitors sleep-cycle training to ensure the model doesn't "drift" into unstable or unethical states during autonomous refinement.
