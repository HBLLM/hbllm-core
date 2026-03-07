# RuVector Integration Implementation Plan

This plan aims to integrate core agentic features and concepts from the `RuVector` neural database into the HBLLM Cognitive Architecture. 

## Goal Description
Enhance HBLLM's memory and reasoning capabilities by making `SemanticMemory` self-learning and context-aware (like RuVector's GNN re-ranking) and giving nodes the ability to hot-swap specialized LoRA skills dynamically.

## User Review Required
> [!IMPORTANT]
> The following features have been identified from `RuVector`. Please review and let me know which one you'd like to prioritize first, or if you approve the entire plan:
> 1. **Self-Learning GNN Re-Ranking:** Making semantic memory rank results based on past experience rewards, not just static cosine distance.
> 2. **Hybrid Search Integration:** A robust fallback that combines keyword matches (TF-IDF) with semantic embeddings into a single ranked list.
> 3. **Dynamic LoRA Hot-Swapping:** A new node trait allowing `SpawnerNode` and `DecisionNode` to hot-swap adapter weights at runtime.
> 4. **Distributed Swarm (Raft Consensus):** Upgrading the HBLLM Message Bus to a multi-master distributed network, allowing nodes to run on different physical machines with automatic failover.
> 5. **Neuromorphic / EXO-AI Metrics:** Adding Spiking Neural Network logic to the Meta Reasoning node, and calculating thermodynamic compute costs for all cognitive paths.
> 6. **Self-Learning DAG Optimizer:** An adaptive planner that tracks execution latencies and actively rewires complex queries (or multi-turn LLM reasoning chains) for speed.
> 7. **Federated Learning:** Allowing separate HBLLM instances to merge learned priority weights securely using Differential Privacy.
> 8. **Edge-Net Collective Compute:** Allowing web/browser clients to donate idle compute to your central HBLLM core via pure WASM clusters.
> 9. **Agentic-Jujutsu Locks:** Implementing lock-free concurrent commit strategies allowing 100+ sub-agents to collaborate on a single task instantly.
> 10. **SciPix OCR Perception:** A specialized engine for reading and routing mathematical and scientific PDF/Images.
> 11. **Cognitive Robotics (Swarm Coordination):** Injecting potential field logic and fast A* pathfinding into the Action nodes for robotic control.
> 12. **rvDNA (Genomics Decoding):** Enabling the Perception node to natively read DNA sequences and calculate pharmacogenomics.
> 13. **RVF Cognitive Containers (.rvf):** Compressing the entire HBLLM brain, vectors, and weights into a single bootable `.rvf` container file.
> 14. **Native 7sense (Bird Call Analysis):** Integrating pure audio embedding pipelines for nature/acoustic anomaly detection.
> 15. **Synthetic Data Generation Engine:** Hooking the Memory Reflection loop into a pipeline that outputs mass QA pairs for fine-tuning offline models.
> 16. **Temporal Tensor Store (Block-Based Vector Files):** Writing memories to highly optimized, separate block files on disk (ADR-018) rather than using a single monolithic SQLite/Vector DB, drastically speeding up I/O.

## Proposed Changes

### Semantic Memory Upgrade (Hybrid + Self-Learning)
#### [MODIFY] `hbllm/memory/semantic.py`
- Build a dual-index structure: Dense (sentence-transformers) + Sparse (TF-IDF).
- Add an `alpha` parameter for hybrid scoring (e.g., `score = alpha * dense + (1 - alpha) * sparse`).
- Implement an **Attention/Re-ranker matrix**: adjust raw vector scores based on historical interactions recorded by the new `ExperienceNode`.

### LoRA Adapter Hot-Swapping
#### [NEW] `hbllm/serving/lora_manager.py`
- A utility to mock or load specialized adapters for local LLM routing (analogous to RuVector's `RuvLtraAdapters`).
#### [MODIFY] `hbllm/brain/spawner_node.py`
- Update the spawner so it doesn't just spawn prompt-based agents, but applies specific LoRA weights contextually.

### REFRAG Simulation (Tensor Caching)
#### [MODIFY] `hbllm/brain/workspace_node.py`
- Introduce a mechanism to cache exact reasoning tensors (or mock them) for common repetitive logic loops, bypassing the heavy generator LLM calls.

### Distributed Swarm Networking (Advanced)
#### [NEW] `hbllm/network/raft_bus.py`
- Create a new implementation of `MessageBus` that leverages a Raft consensus algorithm, allowing a cluster of HBLLM brains to map topics and share a distributed memory store.

### EXO-AI Thermodynamic Critic (Advanced)
#### [MODIFY] `hbllm/brain/critic_node.py`
- Inject logic to calculate the computational cost of reasoning steps (simulating Landauer's principle/EXO-AIs substrate) to penalize overly verbose or circular reasoning by the LLM Planner.

### Federated Learning Network (Advanced)
#### [NEW] `hbllm/network/federation.py`
- Add a secure parameter merge pipeline where extracted patterns from multiple isolated brain nodes encrypt and share their associative weight matrices.

---

## Verification Plan

### Automated Tests
- `pytest tests/test_hybrid_search.py`: Verify that given a mismatch in dense context, the TF-IDF pushes exact keywords higher up the list.
- `pytest tests/test_memory_reranking.py`: Verify that consecutive queries re-prioritize outputs if a `reward` signal was passed between them.
- `pytest tests/test_lora_manager.py`: Ensure hot-swapping profiles yields deterministic shifts in behavior configurations. 

### Manual Verification
1. Run local queries against the memory matrix, supply feedback, and observe the re-ranking in real time without retraining the embedding model.
