"""
ExpressionStream — the full expression-side pipeline.

Orchestrates:
    1. ThoughtPlanner:    UnderstandingState → ThoughtGoal[]
    2. ThoughtController: SNN gating per goal
    3. LLM generation:    Constrained per-goal text
    4. RewardEvaluator:   Score + optional revision
    5. Assembly:          Fragments → final response

The ExpressionStream sits between the WorkspaceNode consensus and the
DecisionNode's final output.  It converts a raw "best thought" into a
structured, thought-by-thought generation that is:
    - Anchored to comprehension concepts (no drift)
    - Gated by SNN neurons (no premature generation)
    - Evaluated per fragment (quality control before assembly)
    - Memory-aware (reuses comprehension memories, no duplicate retrieval)

Usage::

    stream = ExpressionStream(
        planner=ThoughtPlanner(),
        controller=ThoughtController(),
        evaluator=RewardEvaluator(encoder=onnx_encode),
        llm_generate=my_llm.generate,
    )

    result = await stream.express(
        understanding=understanding_state,
        base_thought="The LLM's initial response text...",
        original_query="User's original question",
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

from hbllm.brain.snn.expression.models import (
    ExpressionResult,
    ThoughtFragment,
    ThoughtGoal,
)
from hbllm.brain.snn.expression.reward_evaluator import RewardEvaluator
from hbllm.brain.snn.expression.thought_controller import ThoughtController
from hbllm.brain.snn.expression.thought_planner import ThoughtPlanner

logger = logging.getLogger(__name__)

# Type alias for async LLM generate function
LLMGenerateFn = Callable[[str], Coroutine[Any, Any, str]]

# Callback invoked each time a fragment is generated and evaluated
FragmentCallback = Callable[[ThoughtFragment], Coroutine[Any, Any, None]]


class ExpressionStream:
    """Full expression-side Cognitive Stream pipeline.

    Takes comprehension output and a base thought, then generates a
    structured, per-concept response with SNN gating and reward scoring.

    Args:
        planner: ThoughtPlanner for outline generation.
        controller: ThoughtController for SNN gating.
        evaluator: RewardEvaluator for fragment scoring.
        llm_generate: Async function that takes a prompt and returns text.
        max_revisions: Maximum revision attempts per fragment.
        enable_gating: Whether to use SNN gating (False = direct generation).
    """

    def __init__(
        self,
        planner: ThoughtPlanner,
        controller: ThoughtController,
        evaluator: RewardEvaluator,
        llm_generate: LLMGenerateFn | None = None,
        max_revisions: int = 1,
        enable_gating: bool = True,
        trained_prm: Any | None = None,
        shallow_renderer: Any | None = None,
        shallow_mode: bool = False,
        content_planner: Any | None = None,
        broca_encoder: Any | None = None,
        broca_mode: bool = False,
        on_fragment: FragmentCallback | None = None,
        dual_router: Any | None = None,
    ) -> None:
        self.planner = planner
        self.controller = controller
        self.evaluator = evaluator
        self.llm_generate = llm_generate
        self.max_revisions = max_revisions
        self.enable_gating = enable_gating
        self.trained_prm = trained_prm
        self.shallow_renderer = shallow_renderer
        self.shallow_mode = shallow_mode
        self.content_planner = content_planner
        self.broca_encoder = broca_encoder
        self.broca_mode = broca_mode
        self.on_fragment = on_fragment
        self.dual_router = dual_router

    async def express(
        self,
        understanding: Any,  # UnderstandingState
        base_thought: str,
        original_query: str,
        style_hints: Any | None = None,  # StyleHints
        cancel_event: asyncio.Event | None = None,
    ) -> ExpressionResult:
        """Generate a structured response from comprehension output.

        Args:
            understanding: The UnderstandingState from ComprehensionStream.
            base_thought: The raw thought content from workspace consensus.
            original_query: The user's original query text.
            style_hints: Optional StyleHints from StyleAdapter for emotion-aware rendering.
            cancel_event: Optional asyncio.Event — when set, generation is
                interrupted and partial results are returned.

        Returns:
            ExpressionResult with assembled text and per-fragment scores.
        """
        start_time = time.monotonic()

        # Apply style adaptation if provided
        style_prefix = ""
        if style_hints is not None and hasattr(style_hints, "prompt_prefix"):
            style_prefix = style_hints.prompt_prefix or ""
            if style_prefix:
                original_query = style_prefix + original_query
                logger.debug(
                    "[ExpressionStream] Style adaptation active: tone=%s, verbosity=%s",
                    getattr(style_hints, "tone", "?"),
                    getattr(style_hints, "verbosity", "?"),
                )

        # Step 1: Plan thought outline
        goals = self.planner.plan(understanding)

        if not goals:
            # No concepts → fall through to base thought
            return ExpressionResult(
                text=base_thought,
                fragments=[],
                mean_reward=0.5,
                total_tokens=0,
                thought_count=0,
                revision_count=0,
            )

        # Step 2: Try Broca mode (v4) first
        if (
            self.broca_mode
            and self.content_planner is not None
            and self.broca_encoder is not None
            and self.llm_generate is not None
        ):
            try:
                broca_result = await self._express_broca(
                    understanding, goals, base_thought, original_query
                )
                if broca_result is not None:
                    elapsed = time.monotonic() - start_time
                    broca_result.metadata["mode"] = "broca"
                    broca_result.metadata["elapsed_ms"] = elapsed * 1000
                    return broca_result
            except Exception:
                logger.debug("Broca mode failed (non-fatal), falling back")

        # Step 3: Generate fragments per goal (shallow or deep)
        fragments: list[ThoughtFragment] = []
        prev_fragment_text: str | None = None
        total_tokens = 0
        total_revisions = 0

        self.controller.reset()

        for goal in goals:
            # ── Check for interruption ────────────────────────────────────
            if cancel_event is not None and cancel_event.is_set():
                logger.info(
                    "[ExpressionStream] Interrupted after %d/%d fragments",
                    len(fragments),
                    len(goals),
                )
                break
            # Step 2a: SNN gating — the controller's built-in bypass
            # fires automatically after max_wait_steps
            if self.enable_gating:
                self.controller.gate(goal, prev_fragment_text)

            # Step 2b: Generate text for this goal
            # If shallow mode is active, use shallow rendering prompts
            use_shallow = False
            if (
                self.shallow_mode
                and self.shallow_renderer is not None
                and self.llm_generate is not None
            ):
                try:
                    render_ctx = self.shallow_renderer.build_context(
                        understanding, goals, original_query, base_thought
                    )
                    if self.shallow_renderer.should_use_shallow(render_ctx):
                        shallow_prompt = self.shallow_renderer.render_prompt(
                            render_ctx, goal, prev_fragment_text
                        )
                        text = await self.llm_generate(shallow_prompt)
                        estimated_tokens = max(1, len(text) // 4)
                        fragment = ThoughtFragment(
                            goal_id=goal.id,
                            text=text,
                            metadata={
                                "tokens": estimated_tokens,
                                "source": "shallow",
                                "render_confidence": render_ctx.confidence,
                            },
                        )
                        use_shallow = True
                except Exception:
                    logger.debug("ShallowRenderer failed (non-fatal), using deep path")

            if not use_shallow:
                fragment = await self._generate_for_goal(
                    goal=goal,
                    base_thought=base_thought,
                    original_query=original_query,
                    prev_fragment_text=prev_fragment_text,
                )

            # Step 2c: Evaluate the fragment
            gen_metadata = fragment.metadata.copy()
            if self.trained_prm is not None:
                evaluated = self.trained_prm.evaluate(
                    fragment_text=fragment.text,
                    goal=goal,
                    prev_fragment_text=prev_fragment_text,
                )
            else:
                evaluated = self.evaluator.evaluate(
                    fragment_text=fragment.text,
                    goal=goal,
                    prev_fragment_text=prev_fragment_text,
                )
            gen_metadata.update(evaluated.metadata)
            evaluated.metadata = gen_metadata

            # Step 2d: Revise if needed
            revision_count = 0
            _should_revise = (
                self.trained_prm.should_revise(evaluated)
                if self.trained_prm is not None
                else self.evaluator.should_revise(evaluated)
            )
            while (
                _should_revise
                and revision_count < self.max_revisions
                and self.llm_generate is not None
            ):
                revision_count += 1
                total_revisions += 1

                # Generate a revision with feedback
                revised = await self._revise_fragment(
                    goal=goal,
                    original_fragment=evaluated,
                    original_query=original_query,
                    prev_fragment_text=prev_fragment_text,
                )

                rev_metadata = revised.metadata.copy()
                if self.trained_prm is not None:
                    evaluated = self.trained_prm.evaluate(
                        fragment_text=revised.text,
                        goal=goal,
                        prev_fragment_text=prev_fragment_text,
                    )
                else:
                    evaluated = self.evaluator.evaluate(
                        fragment_text=revised.text,
                        goal=goal,
                        prev_fragment_text=prev_fragment_text,
                    )
                rev_metadata.update(evaluated.metadata)
                evaluated.metadata = rev_metadata
                evaluated.revision_count = revision_count

                _should_revise = (
                    self.trained_prm.should_revise(evaluated)
                    if self.trained_prm is not None
                    else self.evaluator.should_revise(evaluated)
                )

            # Record outcome for PRM training
            if self.trained_prm is not None:
                was_accepted = revision_count == 0
                self.trained_prm.record_outcome(evaluated, accepted=was_accepted)

            # Record for coherence tracking
            self.controller.record_generation(evaluated.text)
            prev_fragment_text = evaluated.text
            total_tokens += evaluated.metadata.get("tokens", 0)
            fragments.append(evaluated)

            # Emit fragment for streaming
            if self.on_fragment is not None:
                try:
                    await self.on_fragment(evaluated)
                except Exception:
                    logger.debug("on_fragment callback failed (non-fatal)")

            # Check for interruption after emitting fragment
            if cancel_event is not None and cancel_event.is_set():
                logger.info(
                    "[ExpressionStream] Interrupted after fragment %d/%d",
                    len(fragments),
                    len(goals),
                )
                break

        # Step 3: Assemble final response
        assembled_text = self._assemble(fragments, base_thought)

        # Compute mean reward
        mean_reward = sum(f.reward_score for f in fragments) / len(fragments) if fragments else 0.0

        latency_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "ExpressionStream: %d goals → %d fragments, mean_reward=%.2f, revisions=%d, %.1fms",
            len(goals),
            len(fragments),
            mean_reward,
            total_revisions,
            latency_ms,
        )

        result_metadata: dict[str, Any] = {}
        if style_hints is not None and hasattr(style_hints, "to_dict"):
            result_metadata["style_hints"] = style_hints.to_dict()
        if cancel_event is not None and cancel_event.is_set():
            result_metadata["interrupted"] = True
            result_metadata["fragments_completed"] = len(fragments)
            result_metadata["fragments_total"] = len(goals)

        return ExpressionResult(
            text=assembled_text,
            fragments=fragments,
            mean_reward=mean_reward,
            total_tokens=total_tokens,
            thought_count=len(goals),
            revision_count=total_revisions,
            metadata=result_metadata,
        )

    async def _express_broca(
        self,
        understanding: Any,
        goals: list[ThoughtGoal],
        base_thought: str,
        original_query: str,
    ) -> ExpressionResult | None:
        """v4 Broca's area pipeline.

        ContentPlanner → BrocaEncoder → LLM (text only) → PRM.

        Returns:
            ExpressionResult if successful, None to fall back.
        """
        # Build rendering context for the content planner
        render_ctx = None
        if self.shallow_renderer is not None:
            render_ctx = self.shallow_renderer.build_context(
                understanding, goals, original_query, base_thought
            )
        else:
            # Minimal context for ContentPlanner
            from hbllm.brain.snn.expression.shallow_renderer import RenderingContext

            concepts = []
            if hasattr(understanding, "concepts"):
                concepts = [c.text for c in understanding.concepts]
            render_ctx = RenderingContext(
                concepts=concepts,
                goals=goals,
                original_query=original_query,
                base_thought=base_thought,
                confidence=0.5,
            )

        # Plan content nodes
        content_nodes = self.content_planner.plan(render_ctx)

        if not content_nodes:
            return None  # Fall back

        # Render each node through BrocaEncoder → LLM
        fragments: list[ThoughtFragment] = []
        rendered_pairs: list[tuple[Any, str]] = []

        for node in content_nodes:
            broca_prompt = self.broca_encoder.encode(node)
            if self.llm_generate is None:
                return None  # Can't render without LLM
            text = await self.llm_generate(broca_prompt.prompt_text)

            fragment = ThoughtFragment(
                goal_id=node.goal_id,
                text=text.strip(),
                metadata={
                    "tokens": max(1, len(text) // 4),
                    "source": "broca",
                    "content_type": node.content_type,
                    "tone": node.tone,
                    "broca_prompt_tokens": broca_prompt.estimated_tokens,
                    "node_confidence": node.confidence,
                },
            )

            # Evaluate with PRM if available
            if self.trained_prm is not None:
                goal_match = None
                for g in goals:
                    if g.id == node.goal_id:
                        goal_match = g
                        break
                if goal_match is None and goals:
                    goal_match = goals[0]

                if goal_match is not None:
                    evaluated = self.trained_prm.evaluate(
                        fragment_text=text.strip(),
                        goal=goal_match,
                    )
                    fragment.reward_score = evaluated.reward_score
                    fragment.metadata.update(evaluated.metadata)
                    self.trained_prm.record_outcome(evaluated, accepted=True)

            fragments.append(fragment)
            rendered_pairs.append((node, text.strip()))

            # Emit fragment for streaming
            if self.on_fragment is not None:
                try:
                    await self.on_fragment(fragment)
                except Exception:
                    logger.debug("on_fragment callback failed (non-fatal)")

        # Assemble final text
        assembled = self.broca_encoder.assemble(rendered_pairs)

        if not assembled.strip():
            return None

        mean_reward = sum(f.reward_score for f in fragments) / len(fragments) if fragments else 0.0

        total_tokens = sum(f.metadata.get("tokens", 0) for f in fragments)

        return ExpressionResult(
            text=assembled,
            fragments=fragments,
            mean_reward=mean_reward,
            total_tokens=total_tokens,
            thought_count=len(content_nodes),
            revision_count=0,
        )

    async def _generate_for_goal(
        self,
        goal: ThoughtGoal,
        base_thought: str,
        original_query: str,
        prev_fragment_text: str | None,
    ) -> ThoughtFragment:
        """Generate text for a single thought goal.

        If no LLM is available, extracts a relevant portion from the
        base_thought using lexical matching.
        """
        if self.llm_generate is not None or self.dual_router is not None:
            prompt = self._build_prompt(goal, base_thought, original_query, prev_fragment_text)
            try:
                if self.dual_router is not None:
                    # Use DualLLMRouter with tier-aware routing:
                    # - Broca/Shallow modes use LOCAL tier (fast, on-device)
                    # - Deep generation uses AUTO tier (complexity-based)
                    from hbllm.brain.control.dual_llm_router import TaskTier

                    if self.broca_mode or self.shallow_mode:
                        tier = TaskTier.LOCAL
                    else:
                        tier = TaskTier.AUTO
                    text = await self.dual_router.generate(prompt, tier=tier)
                else:
                    if self.llm_generate is None:
                        raise RuntimeError("No LLM backend configured (llm_generate is None)")
                    text = await self.llm_generate(prompt)
                estimated_tokens = max(1, len(text) // 4)
                source = "dual_router" if self.dual_router is not None else "llm"
                return ThoughtFragment(
                    goal_id=goal.id,
                    text=text,
                    metadata={"tokens": estimated_tokens, "source": source},
                )
            except Exception as e:
                logger.warning("LLM generation failed for goal %s: %s", goal.id, e)

        # Fallback: extract from base_thought
        text = self._extract_from_base(goal, base_thought)
        return ThoughtFragment(
            goal_id=goal.id,
            text=text,
            metadata={"tokens": max(1, len(text) // 4), "source": "extract"},
        )

    async def _revise_fragment(
        self,
        goal: ThoughtGoal,
        original_fragment: ThoughtFragment,
        original_query: str,
        prev_fragment_text: str | None,
    ) -> ThoughtFragment:
        """Revise a low-scoring fragment with feedback."""
        assert self.llm_generate is not None

        feedback_parts = []
        if original_fragment.relevance_score < 0.5:
            feedback_parts.append(
                f"- Your response was not relevant enough to: '{goal.source_concept_text}'"
            )
        if original_fragment.coherence_score < 0.5:
            feedback_parts.append("- Your response did not flow well from the previous section")
        completeness = original_fragment.metadata.get("completeness", 1.0)
        if completeness < 0.5:
            feedback_parts.append(f"- You missed key terms from: '{goal.source_concept_text}'")

        feedback = "\n".join(feedback_parts) if feedback_parts else "Improve overall quality."

        prompt = (
            f"REVISION REQUEST\n"
            f"Original query: {original_query}\n"
            f"Goal: {goal.text}\n"
            f"Your previous attempt:\n{original_fragment.text}\n\n"
            f"Issues:\n{feedback}\n\n"
            f"Please provide an improved response addressing these issues. "
            f"Keep your response under {goal.max_tokens} tokens."
        )

        try:
            text = await self.llm_generate(prompt)
            return ThoughtFragment(
                goal_id=goal.id,
                text=text,
                metadata={
                    "tokens": max(1, len(text) // 4),
                    "source": "revision",
                },
            )
        except Exception as e:
            logger.warning("Revision failed for goal %s: %s", goal.id, e)
            return original_fragment

    def _build_prompt(
        self,
        goal: ThoughtGoal,
        base_thought: str,
        original_query: str,
        prev_fragment_text: str | None,
    ) -> str:
        """Build a constrained LLM prompt for a thought goal."""
        parts = [
            "You are generating one section of a structured response.",
            f"Original user query: {original_query}",
            "",
            f"CURRENT GOAL: {goal.text}",
            f"Source concept: {goal.source_concept_text}",
        ]

        if goal.memory_hints:
            parts.append("Relevant context from memory:")
            for hint in goal.memory_hints[:3]:
                parts.append(f"  - {hint}")

        if goal.constraints:
            parts.append(f"CONSTRAINTS to respect: {goal.constraints}")

        if prev_fragment_text:
            # Show last 200 chars of previous fragment for continuity
            parts.append("")
            parts.append("Previous section ended with:")
            parts.append(f"...{prev_fragment_text[-200:]}")

        parts.append("")
        parts.append("Reference material (full thought):")
        parts.append(base_thought[:1000])
        parts.append("")
        parts.append(
            f"Generate ONLY the section addressing the current goal. "
            f"Keep it under {goal.max_tokens} tokens. "
            f"Do not repeat previous sections."
        )

        return "\n".join(parts)

    def _extract_from_base(self, goal: ThoughtGoal, base_thought: str) -> str:
        """Extract the most relevant portion of base_thought for a goal.

        Used as a fallback when no LLM is available.
        """
        # Split base_thought into sentences
        sentences = []
        current = []
        for char in base_thought:
            current.append(char)
            if char in ".!?\n" and len(current) > 10:
                sentences.append("".join(current).strip())
                current = []
        if current:
            sentences.append("".join(current).strip())

        if not sentences:
            return base_thought

        # Score each sentence by word overlap with goal
        goal_words = set((goal.source_concept_text or goal.text).lower().split())

        scored = []
        for s in sentences:
            s_words = set(s.lower().split())
            overlap = len(s_words & goal_words)
            scored.append((overlap, s))

        # Take top sentences up to token budget
        scored.sort(key=lambda x: x[0], reverse=True)
        result_parts = []
        total_chars = 0
        budget_chars = goal.max_tokens * 4  # rough token → char

        for _, sentence in scored:
            if total_chars + len(sentence) > budget_chars:
                break
            result_parts.append(sentence)
            total_chars += len(sentence)

        return " ".join(result_parts) if result_parts else sentences[0]

    def _assemble(
        self,
        fragments: list[ThoughtFragment],
        fallback: str,
    ) -> str:
        """Assemble fragments into the final response text.

        Joins fragments with paragraph breaks, filtering out empty ones.
        """
        parts = [f.text.strip() for f in fragments if f.text.strip()]

        if not parts:
            return fallback

        return "\n\n".join(parts)
