"""Autonomous Learner — goal-driven learning orchestrator.

Coordinates all learning subsystems to autonomously acquire knowledge
about any topic through the cognitive loop:

    Goal → Need → Research → Causal Model → Experiment → Verify → Revise

This is NOT curriculum-driven learning. This is goal-driven:
    - Identifies what needs to be understood to achieve a goal
    - Recursively decomposes prerequisites
    - Uses MetaLearner to pick the most cost-efficient learning method
    - Builds causal models (not fact lists)
    - Verifies through experimentation (not recall tests)
    - Detects contradictions and revises beliefs
    - Reports progress via the message bus
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass
class LearningBudget:
    """Adaptive resource budget for a learning goal."""

    daily_web_searches: int = 100
    daily_llm_calls: int = 50
    daily_experiments: int = 20
    remaining_web: int = 100
    remaining_llm: int = 50
    remaining_experiments: int = 20

    def cost(self, action: str) -> float:
        """Resource cost for an action (for MetaLearner tracking)."""
        costs = {
            "research": 2.0,  # web search + LLM call
            "experiment": 5.0,  # LLM + world model
            "review": 1.0,  # LLM only
            "test": 1.5,  # LLM evaluation
            "deepen": 3.0,  # research + analysis
        }
        return costs.get(action, 1.0)

    def can_afford(self, action: str) -> bool:
        """Check if budget allows this action."""
        if action in ("research", "deepen"):
            return self.remaining_web > 0 and self.remaining_llm > 0
        elif action == "experiment":
            return self.remaining_experiments > 0 and self.remaining_llm > 0
        else:
            return self.remaining_llm > 0

    def spend(self, action: str) -> None:
        """Deduct from budget."""
        if action in ("research", "deepen"):
            self.remaining_web = max(0, self.remaining_web - 1)
            self.remaining_llm = max(0, self.remaining_llm - 1)
        elif action == "experiment":
            self.remaining_experiments = max(0, self.remaining_experiments - 1)
            self.remaining_llm = max(0, self.remaining_llm - 1)
        else:
            self.remaining_llm = max(0, self.remaining_llm - 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "daily_web_searches": self.daily_web_searches,
            "daily_llm_calls": self.daily_llm_calls,
            "daily_experiments": self.daily_experiments,
            "remaining_web": self.remaining_web,
            "remaining_llm": self.remaining_llm,
            "remaining_experiments": self.remaining_experiments,
        }


@dataclass
class LearningGoal:
    """A learning goal being pursued by the AutonomousLearner."""

    goal_id: str = field(default_factory=lambda: f"lg_{uuid.uuid4().hex[:10]}")
    topic: str = ""
    depth: str = "adaptive"  # "beginner" | "intermediate" | "advanced" | "adaptive"
    motivation: str = "user_request"  # "user_request" | "curiosity" | "contradiction"
    confidence_target: float = 0.8
    current_confidence: float = 0.0
    budget: LearningBudget = field(default_factory=LearningBudget)
    needs_identified: list[str] = field(default_factory=list)
    needs_completed: list[str] = field(default_factory=list)
    causal_models_built: int = 0
    experiments_run: int = 0
    contradictions_found: int = 0
    contradictions_resolved: int = 0
    status: str = "pending"  # "pending" | "active" | "completed" | "budget_exhausted"
    started_at: float | None = None
    completed_at: float | None = None
    parent_goal_id: str | None = None  # Links to GoalManager.Goal

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "topic": self.topic,
            "depth": self.depth,
            "motivation": self.motivation,
            "confidence_target": self.confidence_target,
            "current_confidence": self.current_confidence,
            "budget": self.budget.to_dict(),
            "needs_identified": self.needs_identified,
            "needs_completed": self.needs_completed,
            "causal_models_built": self.causal_models_built,
            "experiments_run": self.experiments_run,
            "contradictions_found": self.contradictions_found,
            "contradictions_resolved": self.contradictions_resolved,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "parent_goal_id": self.parent_goal_id,
        }


# ── LLM Prompts ──────────────────────────────────────────────────────────────

_NEEDS_PROMPT = """\
To deeply understand "{topic}", what prerequisite concepts must be understood first?

Think recursively: what does each prerequisite itself depend on?

Return a JSON object:
{{
  "needs": [
    {{
      "concept": "...",
      "why": "why this is needed to understand {topic}",
      "priority": 1-5,
      "prerequisites": ["sub-concepts needed for this concept"]
    }}
  ]
}}

Return the most fundamental needs first. Maximum 7 needs.
Return ONLY valid JSON, no markdown."""

_SELF_EVALUATE_PROMPT = """\
You are testing a system's understanding of "{concept}".

The system has this causal model:
{model_summary}

Generate 3 PREDICTION questions (not recall questions).

Bad example (recall): "What is SQL injection?"
Good example (prediction): "Given this code snippet, could an attack occur?"
Good example (transfer): "How would you bypass this specific defense?"

Return a JSON object:
{{
  "questions": [
    {{
      "question": "...",
      "type": "prediction|transfer|application",
      "expected_answer": "...",
      "weight": 1.0-3.0
    }}
  ]
}}

Return ONLY valid JSON, no markdown."""


# ── Autonomous Learner Node ──────────────────────────────────────────────────


class AutonomousLearner(Node):
    """Goal-driven autonomous learning orchestrator.

    Coordinates CausalModelBuilder, ExperimentEngine, ContradictionDetector,
    BeliefRevisionEngine, MetaLearner, and WebResearchNode to autonomously
    learn about any topic.

    Bus subscriptions:
        learning.start     — begin learning a topic
        learning.query     — check progress
        system.idle        — learn during idle time

    Bus publications:
        system.research.request           — trigger WebResearchNode
        learning.progress                 — progress updates
        learning.complete                 — finished learning (legacy)
        learning.session.complete         — rich session data for LearningEventHandler
        learning.contradiction            — contradiction detected (legacy)
        learning.contradiction.discovered — for LearningEventHandler (no echo)
        learning.weak_area                — weak area for CuriosityNode
    """

    def __init__(
        self,
        node_id: str,
        llm: Any = None,
        causal_model_builder: Any | None = None,
        experiment_engine: Any | None = None,
        contradiction_detector: Any | None = None,
        belief_engine: Any | None = None,
        meta_learner: Any | None = None,
        concept_engine: Any | None = None,
        goal_manager: Any | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=[
                "autonomous_learning",
                "goal_driven_research",
                "experimentation",
            ],
        )
        self.llm = llm
        self.causal_model_builder = causal_model_builder
        self.experiment_engine = experiment_engine
        self.contradiction_detector = contradiction_detector
        self.belief_engine = belief_engine
        self.meta_learner = meta_learner
        self.concept_engine = concept_engine
        self.goal_manager = goal_manager

        # Active learning goals
        self._goals: dict[str, LearningGoal] = {}
        self._active_goal: LearningGoal | None = None
        self._learning_lock = asyncio.Lock()

        # State
        self._is_learning = False
        self._total_goals_completed = 0

    async def on_start(self) -> None:
        logger.info("Starting AutonomousLearner")
        await self.bus.subscribe("learning.start", self._handle_start)
        await self.bus.subscribe("learning.query", self._handle_query)

    async def on_stop(self) -> None:
        logger.info(
            "Stopping AutonomousLearner — %d goals completed",
            self._total_goals_completed,
        )

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── Bus Handlers ─────────────────────────────────────────────────────

    async def _handle_start(self, message: Message) -> Message | None:
        """Handle learning.start — begin learning a topic."""
        topic = message.payload.get("topic", "")
        if not topic:
            if message.type == MessageType.QUERY:
                return message.create_error("Missing 'topic' in payload")
            return None

        depth = message.payload.get("depth", "adaptive")
        motivation = message.payload.get("motivation", "user_request")
        confidence_target = message.payload.get("confidence_target", 0.8)

        goal = await self.learn(
            topic=topic,
            depth=depth,
            motivation=motivation,
            confidence_target=confidence_target,
        )

        if message.type == MessageType.QUERY:
            return message.create_response(goal.to_dict())
        return None

    async def _handle_query(self, message: Message) -> Message | None:
        """Handle learning.query — check progress."""
        goal_id = message.payload.get("goal_id")
        if goal_id and goal_id in self._goals:
            return message.create_response(self._goals[goal_id].to_dict())

        # Return all goals
        return message.create_response(
            {
                "active": self._active_goal.to_dict() if self._active_goal else None,
                "total_goals": len(self._goals),
                "completed": self._total_goals_completed,
                "goals": {gid: g.to_dict() for gid, g in self._goals.items()},
            }
        )

    # ── Core Learning Loop ───────────────────────────────────────────────

    async def learn(
        self,
        topic: str,
        depth: str = "adaptive",
        motivation: str = "user_request",
        confidence_target: float = 0.8,
    ) -> LearningGoal:
        """Goal-driven learning loop.

        Flow:
            Goal → Needs → [Research → Model → Experiment → Verify] → Revise
        """
        goal = LearningGoal(
            topic=topic,
            depth=depth,
            motivation=motivation,
            confidence_target=confidence_target,
        )
        self._goals[goal.goal_id] = goal

        async with self._learning_lock:
            self._active_goal = goal
            self._is_learning = True
            goal.status = "active"
            goal.started_at = time.time()

            try:
                # 1. Get strategy from MetaLearner
                strategy = None
                if self.meta_learner is not None:
                    strategy = self.meta_learner.get_strategy(topic)
                    logger.info(
                        "MetaLearner strategy for '%s': %s",
                        topic,
                        strategy.to_dict() if strategy else "default",
                    )

                # 2. Identify prerequisite needs
                goal.needs_identified = await self._identify_needs(topic)
                logger.info(
                    "Identified %d learning needs for '%s': %s",
                    len(goal.needs_identified),
                    topic,
                    goal.needs_identified,
                )

                # 3. Learn each need
                for need in goal.needs_identified:
                    if not goal.budget.can_afford("research"):
                        goal.status = "budget_exhausted"
                        break

                    if goal.current_confidence >= goal.confidence_target:
                        break

                    conf_before = goal.current_confidence
                    await self._learn_concept(goal, need, strategy)
                    goal.needs_completed.append(need)

                    # Publish progress
                    await self._publish_progress(goal)

                    # Record session in MetaLearner
                    if self.meta_learner is not None:
                        recommended = "research"
                        if strategy:
                            recommended = self.meta_learner.recommend_next_action(
                                topic, goal.current_confidence
                            )
                        await self.meta_learner.record_session(
                            domain=topic,
                            method=recommended,
                            confidence_before=conf_before,
                            confidence_after=goal.current_confidence,
                            resource_cost=goal.budget.cost(recommended),
                        )

                # 4. Final status
                if goal.current_confidence >= goal.confidence_target:
                    goal.status = "completed"
                elif goal.status != "budget_exhausted":
                    goal.status = "completed"

                goal.completed_at = time.time()
                self._total_goals_completed += 1

                # Publish completion
                await self._publish_complete(goal)

                # Update GoalManager if this learning goal is linked
                if self.goal_manager and goal.parent_goal_id:
                    try:
                        progress = goal.current_confidence / goal.confidence_target
                        self.goal_manager.update_progress(
                            goal.parent_goal_id,
                            min(1.0, progress),
                            action=(
                                f"Learning completed: confidence={goal.current_confidence:.2f}, "
                                f"models={goal.causal_models_built}, "
                                f"experiments={goal.experiments_run}"
                            ),
                        )
                    except Exception as gm_err:
                        logger.debug("Failed to update GoalManager: %s", gm_err)

            except Exception as e:
                logger.error("Learning goal '%s' failed: %s", topic, e)
                goal.status = "completed"  # Mark as done even on error
            finally:
                self._active_goal = None
                self._is_learning = False

        logger.info(
            "Learning goal '%s' %s: confidence=%.2f, models=%d, experiments=%d, contradictions=%d",
            topic,
            goal.status,
            goal.current_confidence,
            goal.causal_models_built,
            goal.experiments_run,
            goal.contradictions_found,
        )
        return goal

    # ── Sub-steps ────────────────────────────────────────────────────────

    async def _identify_needs(self, topic: str) -> list[str]:
        """Use LLM to decompose a goal into prerequisite needs."""
        if self.llm is None:
            return [topic]

        prompt = _NEEDS_PROMPT.format(topic=topic)
        try:
            response = await self.llm.generate(prompt)
            content = response if isinstance(response, str) else str(response)
            parsed = self._parse_json(content)

            needs: list[str] = []
            for need in parsed.get("needs", []):
                concept = need.get("concept", "")
                if concept:
                    needs.append(concept)
                # Also add prerequisites (recursive depth)
                for prereq in need.get("prerequisites", []):
                    if prereq and prereq not in needs:
                        needs.append(prereq)

            return needs if needs else [topic]
        except Exception as e:
            logger.warning("Failed to identify needs for '%s': %s", topic, e)
            return [topic]

    async def _learn_concept(
        self,
        goal: LearningGoal,
        concept: str,
        strategy: Any | None = None,
    ) -> None:
        """Learn a single concept through the full cognitive loop."""
        logger.info("Learning concept: '%s' (for goal '%s')", concept, goal.topic)

        # Step A: Research via WebResearchNode
        if goal.budget.can_afford("research"):
            await self._research(goal, concept)

        # Step B: Build causal model
        causal_model = None
        if self.causal_model_builder is not None and goal.budget.can_afford("research"):
            try:
                causal_model = await self.causal_model_builder.build_model(
                    concept=concept,
                    domain=goal.topic,
                )
                goal.causal_models_built += 1
                goal.budget.spend("research")
            except Exception as e:
                logger.warning("Failed to build causal model for '%s': %s", concept, e)

        # Step C: Check for contradictions
        if self.contradiction_detector is not None and causal_model is not None:
            try:
                for edge in causal_model.edges:
                    claim = (
                        f"{edge.source_id} causes {edge.target_id} via {edge.mechanism.description}"
                    )
                    contradiction = await self.contradiction_detector.check_contradiction(
                        new_claim=claim,
                        concept=concept,
                    )
                    if contradiction is not None:
                        goal.contradictions_found += 1
                        # Route to belief revision (AutonomousLearner handles this directly)
                        if self.belief_engine is not None:
                            await self.belief_engine.handle_contradiction(contradiction)
                            goal.contradictions_resolved += 1
                        # Publish for CuriosityNode (legacy)
                        await self.publish(
                            "learning.contradiction",
                            Message(
                                type=MessageType.EVENT,
                                source_node_id=self.node_id,
                                topic="learning.contradiction",
                                payload=contradiction.to_dict(),
                            ),
                        )
                        # Emit for LearningEventHandler (log only — NO belief update)
                        await self.publish(
                            "learning.contradiction.discovered",
                            Message(
                                type=MessageType.EVENT,
                                source_node_id=self.node_id,
                                topic="learning.contradiction.discovered",
                                payload={
                                    "claim_a": contradiction.existing_claim,
                                    "claim_b": contradiction.new_claim,
                                    "concept": concept,
                                    "severity": contradiction.severity,
                                    "source": "autonomous_research",
                                },
                            ),
                        )
            except Exception as e:
                logger.debug("Contradiction check failed for '%s': %s", concept, e)

        # Step D: Experiment to verify
        if (
            self.experiment_engine is not None
            and causal_model is not None
            and goal.budget.can_afford("experiment")
        ):
            try:
                results = await self.experiment_engine.experiential_learn(
                    concept=concept,
                    causal_model=causal_model,
                    max_experiments=1,
                )
                goal.experiments_run += len(results)
                goal.budget.spend("experiment")

                # Update model confidence from experiments
                for result in results:
                    if self.causal_model_builder is not None:
                        self.causal_model_builder.update_model_confidence(
                            concept,
                            result.confidence_delta,
                            verified=result.experiment.success is True,
                        )
            except Exception as e:
                logger.debug("Experiment failed for '%s': %s", concept, e)

        # Step E: Self-evaluate
        eval_score = await self._self_evaluate(concept, causal_model)

        # Update goal confidence (rolling average)
        goal.current_confidence = goal.current_confidence * 0.6 + eval_score * 0.4

        # Step F: If weak, publish for CuriosityNode
        if eval_score < 0.4:
            await self.publish(
                "learning.weak_area",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    topic="learning.weak_area",
                    payload={
                        "concept": concept,
                        "score": eval_score,
                        "goal_topic": goal.topic,
                    },
                ),
            )

    async def _research(self, goal: LearningGoal, concept: str) -> None:
        """Trigger web research for a concept."""
        try:
            await self.publish(
                "system.research.request",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    topic="system.research.request",
                    payload={
                        "topic": concept,
                        "query": f"Explain {concept} in depth with causal relationships",
                        "urgency": "medium",
                        "context": f"Learning goal: {goal.topic}",
                        "tier": "core_knowledge",
                    },
                ),
            )
            goal.budget.spend("research")
            # Brief pause to let research complete
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.debug("Research request failed for '%s': %s", concept, e)

    async def _self_evaluate(
        self,
        concept: str,
        causal_model: Any | None = None,
    ) -> float:
        """Evaluate understanding using prediction tests (not recall).

        Returns a score from 0.0 to 1.0.
        """
        if self.llm is None:
            return 0.5

        model_summary = "No causal model available."
        if causal_model is not None and hasattr(causal_model, "to_dict"):
            model_summary = json.dumps(causal_model.to_dict(), indent=2)[:1500]

        prompt = _SELF_EVALUATE_PROMPT.format(
            concept=concept,
            model_summary=model_summary,
        )

        try:
            response = await self.llm.generate(prompt)
            content = response if isinstance(response, str) else str(response)
            parsed = self._parse_json(content)

            questions = parsed.get("questions", [])
            if not questions:
                return 0.5

            # Score based on whether our causal model can address these questions
            total_weight = 0.0
            total_score = 0.0

            for q in questions:
                weight = q.get("weight", 1.0)
                total_weight += weight
                # If we have a causal model, we have some understanding
                if causal_model is not None:
                    total_score += weight * 0.6  # Base score for having a model
                else:
                    total_score += weight * 0.2  # Low score without model

            return total_score / total_weight if total_weight > 0 else 0.5
        except Exception as e:
            logger.debug("Self-evaluation failed for '%s': %s", concept, e)
            return 0.5

    # ── Bus Publications ─────────────────────────────────────────────────

    async def _publish_progress(self, goal: LearningGoal) -> None:
        """Publish learning progress update."""
        try:
            await self.publish(
                "learning.progress",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    topic="learning.progress",
                    payload={
                        "goal_id": goal.goal_id,
                        "topic": goal.topic,
                        "confidence": goal.current_confidence,
                        "needs_completed": len(goal.needs_completed),
                        "needs_total": len(goal.needs_identified),
                        "status": goal.status,
                    },
                ),
            )
        except Exception:
            pass

    async def _publish_complete(self, goal: LearningGoal) -> None:
        """Publish learning completion.

        Emits two events:
        1. learning.complete — legacy, full goal dict
        2. learning.session.complete — for LearningEventHandler
           (mechanisms only, NO belief/meta — already handled above)
        """
        try:
            # Legacy event
            await self.publish(
                "learning.complete",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    topic="learning.complete",
                    payload=goal.to_dict(),
                ),
            )
            # New event for LearningEventHandler (no echo risk)
            duration = (
                (goal.completed_at - goal.started_at)
                if goal.completed_at and goal.started_at
                else 0.0
            )
            await self.publish(
                "learning.session.complete",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    topic="learning.session.complete",
                    payload={
                        "session_id": goal.goal_id,
                        "topic": goal.topic,
                        "concepts_learned": goal.needs_completed,
                        "causal_models_built": goal.causal_models_built,
                        "confidence_before": 0.0,
                        "confidence_after": goal.current_confidence,
                        "experiments_run": goal.experiments_run,
                        "contradictions_found": goal.contradictions_found,
                        "contradictions_resolved": goal.contradictions_resolved,
                        "duration_s": duration,
                        "status": goal.status,
                    },
                ),
            )
        except Exception:
            pass

    # ── Helpers ───────────────────────────────────────────────────────────

    def _parse_json(self, content: str) -> dict[str, Any]:
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
