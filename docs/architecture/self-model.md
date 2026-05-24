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

## Database Schema

The Self-Model stores its data locally in an SQLite database (`self_model.db`) with two main tables:
- `capabilities`: Aggregated metrics per domain.
- `performance_log`: Event-level logs for trend analysis.
