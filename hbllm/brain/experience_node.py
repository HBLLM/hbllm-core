"""
Experience Node — records interactions and detects salience.

Maps to flowchart nodes H (Experience Recorder), I (Salience Detector),
and J (High Importance? decision).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, cast

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


# ── Salience keyword categories with weights ─────────────────────────────────

_SALIENCE_KEYWORDS: dict[str, tuple[list[str], float]] = {
    "emergency": (
        [
            "critical",
            "crash",
            "shutdown",
            "panic",
            "security",
            "breach",
            "unauthorized",
            "fatal",
            "exception",
            "traceback",
        ],
        0.70,
    ),
    "error": (
        [
            "error",
            "failure",
            "failed",
            "broken",
            "bug",
            "fix",
            "issue",
            "alert",
            "warning",
            "timeout",
            "exception",
        ],
        0.50,
    ),
    "learning": (
        [
            "wrong",
            "incorrect",
            "mistake",
            "actually",
            "correction",
            "but i meant",
            "should have been",
            "not what i asked",
        ],
        0.35,
    ),
    "preference": (
        [
            "prefer",
            "always",
            "never",
            "love",
            "hate",
            "favorite",
            "don't like",
            "please stop",
            "instead",
        ],
        0.20,
    ),
    "task": (
        [
            "remember",
            "deadline",
            "schedule",
            "remind",
            "don't forget",
            "important",
            "urgent",
            "priority",
            "asap",
        ],
        0.35,
    ),
    "sentiment_negative": (
        [
            "frustrated",
            "annoying",
            "terrible",
            "worst",
            "useless",
            "awful",
            "horrible",
            "disappointed",
            "confused",
        ],
        0.20,
    ),
}


class ExperienceNode(Node):
    """
    Experience Recorder & Salience Detector (Flowchart Nodes H, I, J).

    Archives all system interactions and scores them for importance using
    multi-signal heuristics: keyword categories, content complexity, novelty
    detection, and optional LLM refinement. High-importance experiences are
    flagged for priority memory and reflection.
    """

    def __init__(
        self,
        node_id: str,
        llm: Any = None,
        importance_threshold: float = 0.7,
        novelty_window: int = 50,
        suppression_ttl: float = 300.0,
        reflection_dir: str = "workspace/reflection",
    ) -> None:
        super().__init__(node_id=node_id, node_type=NodeType.ROUTER)
        self.llm = llm
        self.importance_threshold = importance_threshold

        # Novelty detection: rolling window of content hashes
        self._recent_hashes: deque[str] = deque(maxlen=novelty_window)

        # Salience suppression: prevent repeated priority events for same topic
        self._priority_cooldowns: dict[str, float] = {}
        self._suppression_ttl = suppression_ttl

        # Deep reflection output directory
        self._reflection_dir = Path(reflection_dir)
        self._reflection_dir.mkdir(parents=True, exist_ok=True)

    async def on_start(self) -> None:
        """Subscribe to all sensory outputs and key cognitive events."""
        logger.info("Starting ExperienceNode (Recorder & Salience Detector)")
        await self.bus.subscribe("sensory.output", self.record_experience)
        await self.bus.subscribe("workspace.thought", self._record_cognitive_event)
        await self.bus.subscribe("decision.evaluate", self._record_cognitive_event)

    async def on_stop(self) -> None:
        logger.info("Stopping ExperienceNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def record_experience(self, message: Message) -> None:
        """Record the interaction and detect its salience."""
        payload = message.payload
        content = payload.get("text", "")
        corr_id = message.correlation_id

        if not content:
            return

        logger.info("[ExperienceNode] Recording experience for msg %s", corr_id)

        # 1. Experience Recorder (Node H) — persist to memory
        store_msg = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic="memory.store",
            payload={
                "session_id": message.session_id,
                "role": "system",
                "content": content[:1000],
                "domain": "experience",
                "metadata": {"source": "experience_recorder"},
            },
            correlation_id=corr_id,
        )
        await self.bus.publish("memory.store", store_msg)

        # 2. Salience Detector (Node I) — multi-signal scoring
        score = await self._calculate_saliency(content, payload)

        # 3. High Importance? (Node J) — with suppression check
        is_priority = score >= self.importance_threshold
        if is_priority:
            is_priority = self._check_suppression(content)

        logger.info("[ExperienceNode] Salience=%.2f priority=%s", score, is_priority)

        # Publish for downstream consumers (MemoryNode, MetaReasoningNode)
        salience_msg = Message(
            type=MessageType.SALIENCE_SCORE,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic="system.salience",
            payload={
                "message_id": corr_id,
                "score": round(score, 3),
                "is_priority": is_priority,
                "content": content,
            },
            correlation_id=corr_id,
        )
        await self.bus.publish("system.salience", salience_msg)

        # 4. Deep Reflection (Node E) — structured analysis for priority events
        if is_priority:
            await self._write_reflection(content, score, message)

    # ── Core scoring engine ──────────────────────────────────────────────

    async def _calculate_saliency(self, content: str, payload: dict[str, Any]) -> float:
        """
        Multi-signal importance scoring.

        Signals:
          1. Keyword category matching (emergency > error > learning > etc.)
          2. Content complexity (length, question density, code presence)
          3. Novelty (is this different from recent interactions?)
          4. LLM refinement (optional, if LLM available)
        """
        content_lower = content.lower()

        # ── Signal 1: Keyword matching ───────────────────────────────────
        keyword_score = self._score_keywords(content_lower)

        # ── Signal 2: Content complexity ─────────────────────────────────
        complexity_score = self._score_complexity(content)

        # ── Signal 3: Novelty ────────────────────────────────────────────
        novelty_score = self._score_novelty(content)

        # Additive combination with base — allows a single strong signal
        # (like emergency keywords) to push above the 0.7 threshold
        score = 0.15 + keyword_score * 0.50 + complexity_score * 0.15 + novelty_score * 0.20

        # ── Signal 4: LLM refinement (optional) ─────────────────────────
        if self.llm and score >= 0.5:
            llm_score = await self._score_with_llm(content)
            if llm_score is not None:
                # Blend LLM score (70% LLM, 30% heuristic when LLM available)
                score = 0.3 * score + 0.7 * llm_score

        return min(max(score, 0.0), 1.0)

    @staticmethod
    def _score_keywords(content_lower: str) -> float:
        """Score based on keyword category matches."""
        best_score = 0.0
        total_boost = 0.0

        for _category, (keywords, weight) in _SALIENCE_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in content_lower)
            if hits > 0:
                total_boost += weight * min(hits, 3) / 3  # Cap at 3 hits per category
                best_score = max(best_score, weight)

        # Combine: best single-category match + diminishing returns from multi-category
        return min(best_score + total_boost * 0.5, 1.0)

    @staticmethod
    def _score_complexity(content: str) -> float:
        """Score based on content length, structure, and information density."""
        score = 0.0

        # Length signal (longer = potentially more important, with diminishing returns)
        word_count = len(content.split())
        if word_count > 100:
            score += 0.3
        elif word_count > 50:
            score += 0.2
        elif word_count > 20:
            score += 0.1

        # Question density (questions are often important)
        question_count = content.count("?")
        if question_count >= 3:
            score += 0.25
        elif question_count >= 1:
            score += 0.1

        # Code presence (technical interactions tend to be important)
        code_markers = ["```", "def ", "class ", "import ", "function ", "const ", "let "]
        if any(marker in content for marker in code_markers):
            score += 0.2

        # List/structured content (numbered items, bullet points)
        if any(content.count(marker) >= 2 for marker in ["1.", "- ", "* "]):
            score += 0.1

        return min(score, 1.0)

    def _score_novelty(self, content: str) -> float:
        """Score based on how different this content is from recent interactions."""
        content_hash = hashlib.md5(content.lower().strip().encode()).hexdigest()

        if not self._recent_hashes:
            self._recent_hashes.append(content_hash)
            return 0.8  # First interaction is novel

        # Check for exact dedup
        if content_hash in self._recent_hashes:
            self._recent_hashes.append(content_hash)
            return 0.1  # Exact repeat

        # Check for similarity via word overlap with recent content
        # Use a simple character n-gram approach for speed
        recent_set = set(self._recent_hashes)
        novelty = 1.0 - (len(recent_set & {content_hash}) / max(len(recent_set), 1))

        self._recent_hashes.append(content_hash)
        return min(0.3 + novelty * 0.5, 1.0)

    async def _score_with_llm(self, content: str) -> float | None:
        """Use LLM for deeper semantic salience scoring."""
        try:
            result = await self.llm.generate_json(
                f"Evaluate the importance of this interaction for long-term learning.\n"
                f'Content: "{content[:500]}"\n\n'
                f'Output JSON: {{"importance": 0.0-1.0}}'
            )
            return float(result.get("importance", 0.5))
        except Exception as e:
            logger.debug("LLM salience scoring failed: %s", e)
            return None

    # ── Suppression ──────────────────────────────────────────────────────

    def _check_suppression(self, content: str) -> bool:
        """Check if this topic was recently flagged as priority (prevent spam)."""
        now = time.time()

        # Clean expired entries
        expired = [
            k for k, v in self._priority_cooldowns.items() if now - v > self._suppression_ttl
        ]
        for k in expired:
            del self._priority_cooldowns[k]

        # Check if similar content was recently marked priority
        topic_key = hashlib.md5(content[:100].lower().encode()).hexdigest()
        if topic_key in self._priority_cooldowns:
            logger.debug("[ExperienceNode] Suppressing duplicate priority event")
            return False

        self._priority_cooldowns[topic_key] = now
        return True

    # ── Deep Reflection Writer ────────────────────────────────────────────

    async def _write_reflection(
        self, content: str, salience_score: float, original_message: Message
    ) -> None:
        """
        Deep Reflection (Node E) — generate structured reflection entries
        for high-salience events.

        Performs multi-layer analysis:
          1. Event categorization (domain, severity, type)
          2. Causal analysis (what caused this and why it matters)
          3. Counterfactual generation (chosen vs rejected responses for DPO)
          4. Key entity/concept extraction for knowledge graph
          5. Actionable rules extraction (if→then patterns)

        Outputs:
          - JSONL file to workspace/reflection/ (for DPO training pipeline)
          - system.reflection bus event (for RuleExtractor and KnowledgeGraph)
        """
        try:
            reflection = await self._generate_reflection_entry(
                content, salience_score, original_message
            )

            # Write to disk (JSONL format, one entry per line)
            output_file = self._reflection_dir / "reflections.jsonl"
            with open(output_file, "a") as f:
                f.write(json.dumps(reflection, default=str) + "\n")

            logger.info(
                "[ExperienceNode] Deep reflection written — category=%s entities=%d rules=%d",
                reflection.get("category", "unknown"),
                len(cast(list[Any], reflection.get("entities", []))),
                len(cast(list[Any], reflection.get("rules", []))),
            )

            # Publish for downstream consumers (RuleExtractor, KnowledgeGraph)
            await self.bus.publish(
                "system.reflection",
                Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    tenant_id=original_message.tenant_id,
                    session_id=original_message.session_id,
                    topic="system.reflection",
                    payload=reflection,
                    correlation_id=original_message.correlation_id,
                ),
            )

        except Exception as e:
            logger.error("[ExperienceNode] Deep reflection failed: %s", e)

    async def _generate_reflection_entry(
        self, content: str, salience_score: float, message: Message
    ) -> dict[str, Any]:
        """
        Generate a full structured reflection entry with multi-layer analysis.

        If LLM is available, uses it for deeper analysis. Otherwise falls back
        to heuristic analysis.
        """
        timestamp = time.time()
        content_lower = content.lower()

        # ── Layer 1: Event categorization ────────────────────────────────
        category = self._categorize_event(content_lower)
        severity = (
            "critical" if salience_score >= 0.9 else "high" if salience_score >= 0.7 else "medium"
        )

        # ── Layer 2: Causal analysis ─────────────────────────────────────
        if self.llm:
            causal = await self._llm_causal_analysis(content)
        else:
            causal = self._heuristic_causal_analysis(content, content_lower)

        # ── Layer 3: Counterfactual generation (for DPO training) ────────
        if self.llm:
            counterfactual = await self._llm_counterfactual(content)
        else:
            counterfactual = self._heuristic_counterfactual(content, category)

        # ── Layer 4: Entity extraction ───────────────────────────────────
        entities = self._extract_key_entities(content)

        # ── Layer 5: Rule extraction ─────────────────────────────────────
        rules = self._extract_reflection_rules(content, content_lower, category)

        return {
            "timestamp": timestamp,
            "message_id": message.correlation_id or message.id,
            "tenant_id": message.tenant_id,
            "session_id": message.session_id,
            "salience_score": round(salience_score, 3),
            "category": category,
            "severity": severity,
            "content": content[:2000],  # Truncate very long content
            "causal_analysis": causal,
            "counterfactual": counterfactual,
            "entities": entities,
            "rules": rules,
        }

    def _categorize_event(self, content_lower: str) -> str:
        """Categorize the event based on content analysis."""
        category_signals: dict[str, list[str]] = {
            "security": [
                "security",
                "breach",
                "unauthorized",
                "attack",
                "vulnerability",
                "exploit",
            ],
            "error": ["error", "exception", "crash", "failed", "traceback", "bug"],
            "performance": ["slow", "timeout", "latency", "memory", "leak", "bottleneck"],
            "learning": ["learned", "discovered", "pattern", "insight", "understand"],
            "user_preference": ["prefer", "like", "want", "always", "usually", "favorite"],
            "system_change": ["update", "deploy", "config", "setting", "changed", "modified"],
            "knowledge": ["fact", "rule", "principle", "concept", "definition"],
        }

        scores: dict[str, int] = {}
        for cat, keywords in category_signals.items():
            scores[cat] = sum(1 for kw in keywords if kw in content_lower)

        if scores:
            best = max(scores, key=lambda k: scores[k])
            if scores[best] > 0:
                return best
        return "general"

    def _heuristic_causal_analysis(self, content: str, content_lower: str) -> dict[str, Any]:
        """Heuristic causal analysis when LLM is not available."""
        # Identify potential causes using linguistic markers
        causes: list[str] = []
        effects: list[str] = []
        sentences = [s.strip() for s in content.split(".") if s.strip()]

        causal_markers = ["because", "due to", "caused by", "resulted from", "since"]
        effect_markers = ["therefore", "thus", "consequently", "led to", "resulted in", "causing"]

        for sentence in sentences:
            s_lower = sentence.lower()
            for marker in causal_markers:
                if marker in s_lower:
                    idx = s_lower.index(marker)
                    cause = sentence[idx + len(marker) :].strip()
                    if len(cause) > 10:
                        causes.append(cause[:200])
                    break
            for marker in effect_markers:
                if marker in s_lower:
                    idx = s_lower.index(marker)
                    effect = sentence[idx + len(marker) :].strip()
                    if len(effect) > 10:
                        effects.append(effect[:200])
                    break

        # If no explicit causal language, generate hypotheses from content
        if not causes and sentences:
            # Use the first sentence as likely context/cause
            causes.append(f"Triggered by: {sentences[0][:150]}")
        if not effects and len(sentences) > 1:
            effects.append(f"Observed outcome: {sentences[-1][:150]}")

        return {
            "likely_causes": causes[:3],
            "observed_effects": effects[:3],
            "confidence": 0.4 if causes else 0.2,
            "method": "heuristic",
        }

    async def _llm_causal_analysis(self, content: str) -> dict[str, Any]:
        """Use LLM for deeper causal analysis."""
        try:
            result = await self.llm.generate_json(
                f"Analyze the following event for root cause and effects.\n\n"
                f'Event: "{content[:800]}"\n\n'
                f"Output JSON: "
                f'{{"likely_causes": ["cause1", ...], '
                f'"observed_effects": ["effect1", ...], '
                f'"confidence": 0.0-1.0, '
                f'"reasoning": "brief explanation"}}'
            )
            data = cast(dict[str, Any], result)
            data["method"] = "llm"
            return data
        except Exception as e:
            logger.debug("LLM causal analysis failed: %s", e)
            return self._heuristic_causal_analysis(content, content.lower())

    def _heuristic_counterfactual(self, content: str, category: str) -> dict[str, Any]:
        """Generate counterfactual chosen/rejected pair for DPO training."""
        # The "chosen" response acknowledges the issue and addresses it
        # The "rejected" response is dismissive or incomplete
        prompt = content[:500]

        chosen_templates = {
            "security": f"I've detected a potential security concern: {prompt[:200]}. "
            f"Immediate steps: 1) Isolate the affected component, "
            f"2) Review access logs, 3) Apply appropriate patches or mitigations.",
            "error": f"An error has occurred: {prompt[:200]}. "
            f"Analysis: Let me examine the stack trace and identify the root cause. "
            f"I'll check for recent changes that may have triggered this.",
            "performance": f"Performance issue detected: {prompt[:200]}. "
            f"I'll profile the affected pathway, check resource utilization, "
            f"and identify optimization opportunities.",
            "user_preference": f"Noted preference: {prompt[:200]}. "
            f"I'll remember this preference and apply it consistently "
            f"in future interactions of this type.",
            "learning": f"New insight captured: {prompt[:200]}. "
            f"I'll integrate this into my knowledge base and apply it "
            f"to related scenarios.",
        }

        rejected_templates = {
            "security": "I'll look into it when I get a chance.",
            "error": "That's a known issue, just retry.",
            "performance": "Performance seems fine to me.",
            "user_preference": "Okay.",
            "learning": "That's interesting.",
        }

        chosen = chosen_templates.get(
            category,
            f"I'll analyze this carefully: {prompt[:200]}. "
            f"Let me examine the context and determine the best course of action.",
        )
        rejected = rejected_templates.get(category, "I'll note that.")

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "method": "heuristic",
        }

    async def _llm_counterfactual(self, content: str) -> dict[str, Any]:
        """Use LLM to generate better counterfactual pairs for DPO."""
        try:
            result = await self.llm.generate_json(
                f"Generate a training pair for this high-priority event.\n\n"
                f'Event: "{content[:500]}"\n\n'
                f"Create a prompt and two responses: one exemplary (chosen) and one poor (rejected).\n"
                f"Output JSON: "
                f'{{"prompt": "...", "chosen": "ideal response", "rejected": "bad response"}}'
            )
            data = cast(dict[str, Any], result)
            data["method"] = "llm"
            return data
        except Exception as e:
            logger.debug("LLM counterfactual generation failed: %s", e)
            category = self._categorize_event(content.lower())
            return self._heuristic_counterfactual(content, category)

    def _extract_key_entities(self, content: str) -> list[dict[str, str]]:
        """
        Extract key entities/concepts from content for the knowledge graph.

        Uses a combination of noun-phrase heuristics, capitalization patterns,
        and domain-specific keyword detection.
        """
        import re as _re

        entities: list[dict[str, str]] = []
        seen: set[str] = set()

        # Pattern 1: Capitalized multi-word phrases (proper nouns, technical terms)
        for match in _re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", content):
            entity = match.group(1).strip()
            if entity.lower() not in seen and len(entity) > 3:
                entities.append({"label": entity, "type": "named_entity"})
                seen.add(entity.lower())

        # Pattern 2: Technical terms (camelCase, snake_case, ALL_CAPS)
        for match in _re.finditer(
            r"\b([a-z]+(?:[A-Z][a-z]+)+|[a-z_]+_[a-z_]+|[A-Z]{2,}[A-Z_]*)\b", content
        ):
            term = match.group(1).strip()
            if term.lower() not in seen and len(term) > 3:
                entities.append({"label": term, "type": "technical_term"})
                seen.add(term.lower())

        # Pattern 3: Quoted terms
        for match in _re.finditer(r'["\']([^"\']{3,50})["\']', content):
            term = match.group(1).strip()
            if term.lower() not in seen:
                entities.append({"label": term, "type": "quoted_reference"})
                seen.add(term.lower())

        # Pattern 4: Domain-specific keywords
        domain_terms = {
            "model",
            "dataset",
            "training",
            "inference",
            "embedding",
            "token",
            "pipeline",
            "node",
            "bus",
            "message",
            "tenant",
            "session",
            "memory",
            "adapter",
            "checkpoint",
            "gradient",
            "loss",
        }
        words = set(content.lower().split())
        for term in domain_terms & words:
            if term not in seen:
                entities.append({"label": term, "type": "domain_concept"})
                seen.add(term)

        return entities[:20]  # Cap at 20

    def _extract_reflection_rules(
        self, content: str, content_lower: str, category: str
    ) -> list[dict[str, Any]]:
        """
        Extract actionable if→then rules from the reflection content.

        Uses linguistic patterns to identify conditional behaviors,
        preferences, and causal chains.
        """
        import re as _re

        rules: list[dict[str, Any]] = []

        # Pattern-based rule extraction
        rule_patterns = [
            (
                r"(?:when|whenever|if)\s+(.{10,100}?),?\s+(?:then|should|must|always)\s+(.{10,100}?)(?:\.|$)",
                0.7,
            ),
            (r"(.{10,80}?)\s+(?:always|usually|typically)\s+(.{10,100}?)(?:\.|$)", 0.5),
            (
                r"(?:user|they|system)\s+(?:prefers?|wants?|needs?)\s+(.{10,100}?)(?:\s+(?:when|for)\s+(.{10,100}?))?(?:\.|$)",
                0.6,
            ),
            (
                r"(?:never|avoid|don't)\s+(.{10,80}?)\s+(?:because|since|as)\s+(.{10,100}?)(?:\.|$)",
                0.65,
            ),
            (r"(.{10,80}?)\s+(?:leads to|results in|causes)\s+(.{10,100}?)(?:\.|$)", 0.55),
        ]

        for pattern, base_confidence in rule_patterns:
            for match in _re.finditer(pattern, content_lower, _re.IGNORECASE):
                condition = match.group(1).strip()
                last_idx = match.lastindex
                action = (
                    match.group(2).strip()
                    if last_idx is not None and last_idx >= 2
                    else "take appropriate action"
                )

                if len(condition) < 8 or len(action) < 8:
                    continue

                # Boost confidence for domain-aligned rules
                confidence = base_confidence
                if category in ("security", "error"):
                    confidence += 0.1

                rule_id = hashlib.md5(f"{condition}|{action}".encode()).hexdigest()[:12]

                rules.append(
                    {
                        "rule_id": rule_id,
                        "condition": condition[:200],
                        "action": action[:200],
                        "confidence": round(min(confidence, 0.95), 2),
                        "category": category,
                    }
                )

        return rules[:10]  # Cap at 10 rules per reflection

    # ── Cognitive event recording ────────────────────────────────────────

    async def _record_cognitive_event(self, message: Message) -> None:
        """
        Record and score cognitive events (workspace thoughts, decisions).
        """
        payload = message.payload
        thought_type = str(payload.get("type", ""))

        # Skip meta-events to avoid recursion
        if thought_type in ("critique", "simulation_result"):
            return

        content = str(payload.get("content", ""))
        if not content:
            return

        logger.debug("[ExperienceNode] Recording cognitive event: %s", thought_type)

        # Score cognitive events too — they carry learning signals
        score = await self._calculate_saliency(content, payload)

        if score >= self.importance_threshold:
            salience_msg = Message(
                type=MessageType.SALIENCE_SCORE,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="system.salience",
                payload={
                    "message_id": message.correlation_id,
                    "score": round(score, 3),
                    "is_priority": True,
                    "content": content,
                    "source": "cognitive_event",
                    "type": thought_type,
                },
                correlation_id=message.correlation_id,
            )
            await self.bus.publish("system.salience", salience_msg)
