"""Experiment Engine — verify learned knowledge through experimentation.

Generates hypotheses from causal models and runs experiments at different
reality levels. Each experiment method has a different confidence weight:

    SIMULATED           weight=0.2  (pure logic/heuristic)
    LLM_PREDICTED       weight=0.3  (LLM-based reasoning)
    WORLD_MODEL         weight=0.5  (AST simulation/command simulation)
    REAL_EXECUTION      weight=0.8  (sandboxed code execution)
    REAL_OBSERVATION     weight=1.0  (real-world observed outcome)

This prevents simulated knowledge from being overtrusted.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Reality Levels ───────────────────────────────────────────────────────────


class RealityLevel(str, Enum):
    """Ranked experiment fidelity — higher = more trusted."""

    SIMULATED = "simulated"
    LLM_PREDICTED = "llm_predicted"
    WORLD_MODEL = "world_model"
    REAL_EXECUTION = "real_execution"
    REAL_OBSERVATION = "real_observation"


REALITY_CONFIDENCE_WEIGHTS: dict[RealityLevel, float] = {
    RealityLevel.SIMULATED: 0.2,
    RealityLevel.LLM_PREDICTED: 0.3,
    RealityLevel.WORLD_MODEL: 0.5,
    RealityLevel.REAL_EXECUTION: 0.8,
    RealityLevel.REAL_OBSERVATION: 1.0,
}


# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass
class Hypothesis:
    """A testable hypothesis derived from a causal model."""

    hypothesis_id: str = field(default_factory=lambda: f"hyp_{uuid.uuid4().hex[:10]}")
    statement: str = ""
    concept: str = ""
    causal_model_id: str | None = None
    confidence_before: float = 0.5
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "concept": self.concept,
            "causal_model_id": self.causal_model_id,
            "confidence_before": self.confidence_before,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Hypothesis:
        return cls(
            hypothesis_id=d.get("hypothesis_id", f"hyp_{uuid.uuid4().hex[:10]}"),
            statement=d.get("statement", ""),
            concept=d.get("concept", ""),
            causal_model_id=d.get("causal_model_id"),
            confidence_before=d.get("confidence_before", 0.5),
            created_at=d.get("created_at", time.time()),
        )


@dataclass
class Experiment:
    """A designed experiment to verify a hypothesis."""

    experiment_id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:10]}")
    hypothesis: Hypothesis = field(default_factory=Hypothesis)
    reality_level: RealityLevel = RealityLevel.LLM_PREDICTED
    setup: str = ""  # Code, scenario, or simulation description
    expected_outcome: str = ""
    actual_outcome: str | None = None
    success: bool | None = None
    confidence_after: float = 0.0
    resource_cost: float = 1.0  # For MetaLearner cost-efficiency
    executed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "hypothesis": self.hypothesis.to_dict(),
            "reality_level": self.reality_level.value,
            "setup": self.setup,
            "expected_outcome": self.expected_outcome,
            "actual_outcome": self.actual_outcome,
            "success": self.success,
            "confidence_after": self.confidence_after,
            "resource_cost": self.resource_cost,
            "executed_at": self.executed_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Experiment:
        return cls(
            experiment_id=d.get("experiment_id", f"exp_{uuid.uuid4().hex[:10]}"),
            hypothesis=Hypothesis.from_dict(d.get("hypothesis", {})),
            reality_level=RealityLevel(d.get("reality_level", "llm_predicted")),
            setup=d.get("setup", ""),
            expected_outcome=d.get("expected_outcome", ""),
            actual_outcome=d.get("actual_outcome"),
            success=d.get("success"),
            confidence_after=d.get("confidence_after", 0.0),
            resource_cost=d.get("resource_cost", 1.0),
            executed_at=d.get("executed_at"),
        )


@dataclass
class ExperimentResult:
    """The outcome of running an experiment."""

    experiment: Experiment = field(default_factory=Experiment)
    confidence_delta: float = 0.0
    reality_weight: float = 0.3
    new_knowledge: list[str] = field(default_factory=list)
    causal_updates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment.to_dict(),
            "confidence_delta": self.confidence_delta,
            "reality_weight": self.reality_weight,
            "new_knowledge": self.new_knowledge,
            "causal_updates": self.causal_updates,
        }


# ── LLM Prompts ──────────────────────────────────────────────────────────────

_HYPOTHESIS_PROMPT = """\
Given the concept "{concept}" and this causal model:
{model_summary}

Generate a testable hypothesis that could verify or falsify one of the
causal relationships in this model.

Return a JSON object:
{{
  "statement": "If [condition], then [prediction] because [mechanism]",
  "target_edge": "which causal edge this tests",
  "expected_outcome": "what we expect to observe"
}}

Return ONLY valid JSON, no markdown."""

_EXPERIMENT_DESIGN_PROMPT = """\
Design an experiment to test this hypothesis:
"{statement}"

Expected outcome: {expected}

The experiment should be one of:
1. A logical reasoning scenario (for LLM-level verification)
2. A code snippet that demonstrates the concept (for simulation)
3. A thought experiment with clear success/failure criteria

Return a JSON object:
{{
  "setup": "description or code for the experiment",
  "method": "reasoning|code_simulation|thought_experiment",
  "success_criteria": "how to determine if hypothesis is confirmed",
  "failure_criteria": "how to determine if hypothesis is falsified"
}}

Return ONLY valid JSON, no markdown."""

_EXPERIMENT_EVALUATION_PROMPT = """\
Evaluate this experiment result:

Hypothesis: {statement}
Expected: {expected}
Setup: {setup}
Actual Outcome: {outcome}

Questions:
1. Does the outcome confirm or falsify the hypothesis?
2. What confidence level (0.0-1.0) should we assign?
3. Are there any unexpected findings or new knowledge?

Return a JSON object:
{{
  "confirmed": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "why",
  "new_knowledge": ["any unexpected findings"],
  "causal_updates": [
    {{"source": "...", "target": "...", "confidence_delta": 0.0-1.0}}
  ]
}}

Return ONLY valid JSON, no markdown."""


# ── Experiment Engine ────────────────────────────────────────────────────────


class ExperimentEngine:
    """Generates and runs experiments to verify learned knowledge.

    Uses LLM for hypothesis generation and experiment design,
    WorldModelNode for sandboxed execution, and confidence-weighted
    RealityLevels to avoid overtrusting simulated results.
    """

    def __init__(
        self,
        llm: Any,
        world_model: Any | None = None,
        data_dir: str | Path = "data",
        sandbox_enabled: bool = True,
    ) -> None:
        self.llm = llm
        self.world_model = world_model
        self.sandbox_enabled = sandbox_enabled
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # SQLite storage
        self._db_path = self.data_dir / "experiments.db"
        self._init_db()

        # Telemetry
        self._total_experiments = 0
        self._confirmed = 0
        self._falsified = 0
        self._total_cost = 0.0

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    concept TEXT NOT NULL,
                    data TEXT NOT NULL,
                    reality_level TEXT NOT NULL,
                    success INTEGER,
                    confidence_after REAL,
                    resource_cost REAL DEFAULT 1.0,
                    executed_at REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_exp_concept ON experiments(concept)")

    def _persist(self, experiment: Experiment) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO experiments
                       (experiment_id, concept, data, reality_level,
                        success, confidence_after, resource_cost, executed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        experiment.experiment_id,
                        experiment.hypothesis.concept,
                        json.dumps(experiment.to_dict()),
                        experiment.reality_level.value,
                        1 if experiment.success else (0 if experiment.success is False else None),
                        experiment.confidence_after,
                        experiment.resource_cost,
                        experiment.executed_at,
                    ),
                )
        except Exception as e:
            logger.warning("Failed to persist experiment: %s", e)

    # ── Core API ─────────────────────────────────────────────────────────

    async def generate_hypothesis(
        self,
        concept: str,
        causal_model: Any,
    ) -> Hypothesis:
        """Use LLM to generate a testable hypothesis from a causal model."""
        model_summary = (
            json.dumps(causal_model.to_dict(), indent=2)
            if hasattr(causal_model, "to_dict")
            else str(causal_model)
        )

        prompt = _HYPOTHESIS_PROMPT.format(
            concept=concept,
            model_summary=model_summary[:2000],
        )

        try:
            response = await self.llm.generate(prompt)
            content = response if isinstance(response, str) else str(response)
            parsed = self._parse_json(content)

            return Hypothesis(
                statement=parsed.get("statement", f"Hypothesis about {concept}"),
                concept=concept,
                causal_model_id=getattr(causal_model, "model_id", None),
                confidence_before=getattr(causal_model, "confidence", 0.5),
            )
        except Exception as e:
            logger.warning("Failed to generate hypothesis for '%s': %s", concept, e)
            return Hypothesis(
                statement=f"The causal model for {concept} is accurate",
                concept=concept,
                confidence_before=0.5,
            )

    async def design_experiment(
        self,
        hypothesis: Hypothesis,
    ) -> Experiment:
        """Design an experiment to test a hypothesis."""
        prompt = _EXPERIMENT_DESIGN_PROMPT.format(
            statement=hypothesis.statement,
            expected=hypothesis.statement,
        )

        try:
            response = await self.llm.generate(prompt)
            content = response if isinstance(response, str) else str(response)
            parsed = self._parse_json(content)

            method = parsed.get("method", "reasoning")
            reality_level = {
                "reasoning": RealityLevel.LLM_PREDICTED,
                "code_simulation": RealityLevel.WORLD_MODEL,
                "thought_experiment": RealityLevel.SIMULATED,
            }.get(method, RealityLevel.LLM_PREDICTED)

            # Cost varies by reality level
            cost_map = {
                RealityLevel.SIMULATED: 0.5,
                RealityLevel.LLM_PREDICTED: 1.0,
                RealityLevel.WORLD_MODEL: 2.0,
                RealityLevel.REAL_EXECUTION: 5.0,
                RealityLevel.REAL_OBSERVATION: 10.0,
            }

            return Experiment(
                hypothesis=hypothesis,
                reality_level=reality_level,
                setup=parsed.get("setup", ""),
                expected_outcome=parsed.get(
                    "success_criteria",
                    hypothesis.statement,
                ),
                resource_cost=cost_map.get(reality_level, 1.0),
            )
        except Exception as e:
            logger.warning("Failed to design experiment: %s", e)
            return Experiment(
                hypothesis=hypothesis,
                reality_level=RealityLevel.LLM_PREDICTED,
                setup=f"Evaluate: {hypothesis.statement}",
                expected_outcome="Hypothesis is confirmed",
                resource_cost=1.0,
            )

    async def run_experiment(
        self,
        experiment: Experiment,
    ) -> ExperimentResult:
        """Execute an experiment and return weighted results."""
        reality_weight = REALITY_CONFIDENCE_WEIGHTS.get(experiment.reality_level, 0.3)

        # Execute based on reality level
        if experiment.reality_level == RealityLevel.WORLD_MODEL and self.world_model:
            outcome = await self._run_world_model(experiment)
        elif experiment.reality_level in (
            RealityLevel.REAL_EXECUTION,
            RealityLevel.REAL_OBSERVATION,
        ):
            # Safety: downgrade to LLM_PREDICTED if sandbox disabled
            if not self.sandbox_enabled:
                logger.info(
                    "Sandbox disabled — downgrading %s to LLM_PREDICTED",
                    experiment.reality_level.value,
                )
                experiment.reality_level = RealityLevel.LLM_PREDICTED
                reality_weight = REALITY_CONFIDENCE_WEIGHTS[RealityLevel.LLM_PREDICTED]
            outcome = await self._run_llm_evaluation(experiment)
        else:
            outcome = await self._run_llm_evaluation(experiment)

        # Evaluate the outcome
        result = await self._evaluate_outcome(experiment, outcome, reality_weight)

        # Persist
        experiment.actual_outcome = outcome
        experiment.executed_at = time.time()
        self._persist(experiment)

        # Telemetry
        self._total_experiments += 1
        self._total_cost += experiment.resource_cost
        if experiment.success:
            self._confirmed += 1
        elif experiment.success is False:
            self._falsified += 1

        logger.info(
            "Experiment %s: concept='%s' reality=%s confirmed=%s confidence_delta=%.2f weight=%.1f",
            experiment.experiment_id,
            experiment.hypothesis.concept,
            experiment.reality_level.value,
            experiment.success,
            result.confidence_delta,
            result.reality_weight,
        )
        return result

    async def experiential_learn(
        self,
        concept: str,
        causal_model: Any,
        max_experiments: int = 3,
    ) -> list[ExperimentResult]:
        """Full experiential learning loop for a concept.

        1. Generate hypotheses from causal model
        2. Design experiments for each
        3. Run experiments
        4. Collect results + update causal model confidence
        """
        results: list[ExperimentResult] = []

        for i in range(max_experiments):
            try:
                # 1. Generate hypothesis
                hypothesis = await self.generate_hypothesis(concept, causal_model)

                # 2. Design experiment
                experiment = await self.design_experiment(hypothesis)

                # 3. Run experiment
                result = await self.run_experiment(experiment)
                results.append(result)

                # Update model confidence for next iteration
                if hasattr(causal_model, "confidence"):
                    causal_model.confidence = max(
                        0.0,
                        min(1.0, causal_model.confidence + result.confidence_delta),
                    )
            except Exception as e:
                logger.warning(
                    "Experiential learning iteration %d failed for '%s': %s",
                    i,
                    concept,
                    e,
                )

        logger.info(
            "Experiential learning for '%s': %d experiments, %d confirmed, %d falsified",
            concept,
            len(results),
            sum(1 for r in results if r.experiment.success),
            sum(1 for r in results if r.experiment.success is False),
        )
        return results

    def stats(self) -> dict[str, Any]:
        """Return experiment engine statistics."""
        return {
            "total_experiments": self._total_experiments,
            "confirmed": self._confirmed,
            "falsified": self._falsified,
            "total_cost": self._total_cost,
            "confirmation_rate": (
                self._confirmed / self._total_experiments if self._total_experiments > 0 else 0.0
            ),
        }

    # ── Internal Helpers ─────────────────────────────────────────────────

    async def _run_world_model(self, experiment: Experiment) -> str:
        """Run experiment via WorldModelNode simulation."""
        try:
            from hbllm.network.messages import Message, MessageType

            msg = Message(
                type=MessageType.QUERY,
                source_node_id="experiment_engine",
                topic="world_model.simulate",
                payload={
                    "action_type": "thought_experiment",
                    "content": experiment.setup,
                },
            )
            result = await self.world_model.simulate_action(msg)
            if result and hasattr(result, "payload"):
                return json.dumps(result.payload)
            return str(result)
        except Exception as e:
            logger.warning("WorldModel simulation failed: %s", e)
            return await self._run_llm_evaluation(experiment)

    async def _run_llm_evaluation(self, experiment: Experiment) -> str:
        """Run experiment via LLM reasoning."""
        prompt = (
            f"Evaluate this scenario:\n\n"
            f"Setup: {experiment.setup}\n\n"
            f"Question: Does the following hold true?\n"
            f"{experiment.hypothesis.statement}\n\n"
            f"Think step by step and provide your conclusion."
        )
        try:
            response = await self.llm.generate(prompt)
            return response if isinstance(response, str) else str(response)
        except Exception as e:
            return f"LLM evaluation failed: {e}"

    async def _evaluate_outcome(
        self,
        experiment: Experiment,
        outcome: str,
        reality_weight: float,
    ) -> ExperimentResult:
        """Use LLM to evaluate experiment outcome and compute confidence delta."""
        prompt = _EXPERIMENT_EVALUATION_PROMPT.format(
            statement=experiment.hypothesis.statement,
            expected=experiment.expected_outcome,
            setup=experiment.setup[:500],
            outcome=outcome[:500],
        )

        try:
            response = await self.llm.generate(prompt)
            content = response if isinstance(response, str) else str(response)
            parsed = self._parse_json(content)

            confirmed = parsed.get("confirmed", False)
            raw_confidence = parsed.get("confidence", 0.5)

            # Weight confidence by reality level
            weighted_confidence = raw_confidence * reality_weight

            # Compute delta
            if confirmed:
                confidence_delta = weighted_confidence * 0.1  # Positive reinforcement
                experiment.success = True
            else:
                confidence_delta = -weighted_confidence * 0.15  # Negative is stronger
                experiment.success = False

            experiment.confidence_after = max(
                0.0,
                min(
                    1.0,
                    experiment.hypothesis.confidence_before + confidence_delta,
                ),
            )

            return ExperimentResult(
                experiment=experiment,
                confidence_delta=confidence_delta,
                reality_weight=reality_weight,
                new_knowledge=parsed.get("new_knowledge", []),
                causal_updates=parsed.get("causal_updates", []),
            )
        except Exception as e:
            logger.warning("Failed to evaluate experiment outcome: %s", e)
            experiment.success = None
            experiment.confidence_after = experiment.hypothesis.confidence_before
            return ExperimentResult(
                experiment=experiment,
                confidence_delta=0.0,
                reality_weight=reality_weight,
            )

    def _parse_json(self, content: str) -> dict[str, Any]:
        """Extract JSON from LLM response."""
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(text[start:end])
                except json.JSONDecodeError:
                    pass
            return {}
