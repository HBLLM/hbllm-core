# Self-Model Architecture

The Self-Model is the internal awareness component of HBLLM. It tracks what the system is good at, where it is weak, and uses this self-knowledge to make better routing and delegation decisions.

## Purpose

Traditional LLMs do not know their own failure rates or competence boundaries. The HBLLM Self-Model solves this by maintaining a rolling database of performance metrics across different cognitive domains (e.g., coding, medical, math).

## Core Responsibilities

1. **Performance Tracking**: Records the success rate, confidence calibration, and latency of every interaction via the `EvaluationNode`.
2. **Domain Expertise Evaluation**: Identifies strengths (score > 0.8) and weaknesses (score < 0.5).
3. **Trend Analysis**: Detects if a domain's performance is `improving`, `declining`, or `stable` over the last 20 interactions.
4. **Delegation Decisions**: Informs the `DecisionNode` whether a query should be handled locally, passed to a larger model, or delegated to a specialized agent.

## Integration with Sleep Cycle

The Self-Model tightly integrates with the `SleepCycleNode` to direct the autonomous Continuous DPO (Direct Preference Optimization) training.

During the overnight stage, `SleepNode._run_self_improvement()` queries `SelfModel.get_weaknesses()` and `SelfModel.get_metrics()['declining']`. This ensures that the system focuses its neural plasticity and training cycles exclusively on the domains where it is currently struggling, leading to targeted self-improvement rather than random sampling.

## Competence Tracking

To assess granular tool and action capabilities, the Self-Model tracks structural competence records:

### 1. Capability Profile
The `CapabilityProfile` stores running metrics for specific capability strings (e.g., `execute_python`, `web_search`):
- **capability**: The unique capability identifier.
- **confidence**: Rolling confidence score representing predicted success rate.
- **success_rate**: Running success rate of executions.
- **avg_cost**: Average token usage and latency (in milliseconds).
- **last_validated**: Timestamp of the most recent evaluation.

### 2. Experience Record
The `ExperienceRecord` preserves raw execution history and validation details for offline consolidation:
- **capability**: The targeted capability.
- **executions_count**: Cumulative number of execution runs.
- **validation_runs**: JSON list containing individual execution metadata and validation outputs.

### Key Methods
* **`get_capability_profile(capability: str)`**: Returns the capability's rolling performance profiles as a `CapabilityProfile` dataclass.
* **`get_experience_record(capability: str)`**: Returns the raw history of execution and validation runs as an `ExperienceRecord` dataclass.
* **`record_experience(capability: str, success: bool, tokens_used: int, latency_ms: float, validation_info: dict[str, Any] | None = None)`**: Appends a new execution outcome, increments execution counts, stores verification data, and updates the capability profile via a rolling average.

---

## Database Schema

The Self-Model stores its data locally in an SQLite database (`self_model.db`) with four main tables:
- `capabilities`: Aggregated metrics per domain.
- `performance_log`: Event-level logs for trend analysis.
- `capability_profiles`: Running profiles for granular action capabilities.
- `experience_records`: Full execution counts and validation metadata.

