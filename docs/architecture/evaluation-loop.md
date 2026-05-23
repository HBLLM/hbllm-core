# Evaluation & Micro-Learning Loop

HBLLM uses a continuous, closed-loop evaluation system designed to monitor the quality of every cognitive interaction and automatically correct mistakes over time.

## 1. The Evaluation Node

The `EvaluationNode` sits on the event bus and listens for `system.evaluation` events, which are generated immediately after a response is delivered to the user.

### Scoring Dimensions
Every interaction is scored (0.0 to 1.0) across five dimensions using an LLM-as-a-judge heuristic (or fast-path checks for simple queries):
- **Task Success** (`task_success`): Did the system achieve the user's intent?
- **Plan Validity** (`plan_validity`): If steps were generated, were they logically sound and well-structured?
- **Tool Accuracy** (`tool_accuracy`): Did the selected plugins/tools execute successfully?
- **Memory Usage** (`memory_usage`): Was the retrieved context relevant and useful for addressing the query?
- **Confidence Error** (`confidence_error`): The calibration quality, measured as the difference between the predicted confidence and the actual success level.

*(Note: Safety, toxicity, and policy compliance are handled proactively by the `SentinelNode` and `GovernanceGuard` layers rather than rolling post-decision evaluation scores.)*

## 2. Micro-Learning

The evaluation pipeline feeds directly into the `LearnerNode` to form the **Micro-Learning Loop**:

1. **Queueing Failures**: If an interaction scores below the `micro_learn_threshold` (e.g., `< 0.4`), the `LearnerNode` queues the `bad_response` associated with that query.
2. **Success Matching**: When the user retries a prompt and the system generates a highly-scored response (a `good_response`), the `LearnerNode` detects that the query exists in its queue.
3. **Triggering Micro-Learn**: It pairs the `bad_response` with the new `good_response` and triggers a micro-learning task.
4. **DPO Generation**: This pair is formatted into a Direct Preference Optimization (DPO) tuple (prompt, chosen, rejected) and stored in the distillation bank.

## 3. Feedback Processing

User feedback explicitly overrides heuristic evaluation:
- **Negative Feedback**: Automatically calculates a confidence error and injects the interaction into the micro-learning pipeline.
- **Corrections**: If the user provides a preferred response alongside negative feedback, the system immediately triggers the micro-learning correction, bypassing the need to wait for a successful retry.

## 4. Distillation and Sleep

High-scoring interactions (> 0.8) and corrected DPO pairs are held in the `distillation_bank`. During the sleep cycle, the `SleepNode` orchestrates continuous learning, flushing these banks to permanently update the model weights or RAG priority tables.
