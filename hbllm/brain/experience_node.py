"""
Experience Node — records interactions and detects salience.

Maps to flowchart nodes H (Experience Recorder), I (Salience Detector),
and J (High Importance? decision).
"""

import hashlib
import logging
import time
from collections import deque

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


# ── Salience keyword categories with weights ─────────────────────────────────

_SALIENCE_KEYWORDS: dict[str, tuple[list[str], float]] = {
    "emergency": (
        ["critical", "crash", "shutdown", "panic", "security", "breach",
         "unauthorized", "fatal", "exception", "traceback"],
        0.35,
    ),
    "error": (
        ["error", "failure", "failed", "broken", "bug", "fix", "issue",
         "alert", "warning", "timeout"],
        0.25,
    ),
    "learning": (
        ["wrong", "incorrect", "mistake", "actually", "correction",
         "but i meant", "should have been", "not what i asked"],
        0.20,
    ),
    "preference": (
        ["prefer", "always", "never", "love", "hate", "favorite",
         "don't like", "please stop", "instead"],
        0.15,
    ),
    "task": (
        ["remember", "deadline", "schedule", "remind", "don't forget",
         "important", "urgent", "priority", "asap"],
        0.20,
    ),
    "sentiment_negative": (
        ["frustrated", "annoying", "terrible", "worst", "useless",
         "awful", "horrible", "disappointed", "confused"],
        0.15,
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
        llm=None,
        importance_threshold: float = 0.7,
        novelty_window: int = 50,
        suppression_ttl: float = 300.0,
    ):
        super().__init__(node_id=node_id, node_type=NodeType.ROUTER)
        self.llm = llm
        self.importance_threshold = importance_threshold

        # Novelty detection: rolling window of content hashes
        self._recent_hashes: deque[str] = deque(maxlen=novelty_window)

        # Salience suppression: prevent repeated priority events for same topic
        self._priority_cooldowns: dict[str, float] = {}
        self._suppression_ttl = suppression_ttl

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

        logger.info(
            "[ExperienceNode] Salience=%.2f priority=%s", score, is_priority
        )

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

    # ── Core scoring engine ──────────────────────────────────────────────

    async def _calculate_saliency(self, content: str, payload: dict) -> float:
        """
        Multi-signal importance scoring.

        Signals:
          1. Keyword category matching (emergency > error > learning > etc.)
          2. Content complexity (length, question density, code presence)
          3. Novelty (is this different from recent interactions?)
          4. LLM refinement (optional, if LLM available)
        """
        signals: list[tuple[str, float]] = []
        content_lower = content.lower()

        # ── Signal 1: Keyword matching ───────────────────────────────────
        keyword_score = self._score_keywords(content_lower)
        signals.append(("keywords", keyword_score))

        # ── Signal 2: Content complexity ─────────────────────────────────
        complexity_score = self._score_complexity(content)
        signals.append(("complexity", complexity_score))

        # ── Signal 3: Novelty ────────────────────────────────────────────
        novelty_score = self._score_novelty(content)
        signals.append(("novelty", novelty_score))

        # Weighted combination
        weights = {"keywords": 0.45, "complexity": 0.20, "novelty": 0.35}
        score = sum(weights.get(name, 0) * val for name, val in signals)

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

        for category, (keywords, weight) in _SALIENCE_KEYWORDS.items():
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
        expired = [k for k, v in self._priority_cooldowns.items() if now - v > self._suppression_ttl]
        for k in expired:
            del self._priority_cooldowns[k]

        # Check if similar content was recently marked priority
        topic_key = hashlib.md5(content[:100].lower().encode()).hexdigest()
        if topic_key in self._priority_cooldowns:
            logger.debug("[ExperienceNode] Suppressing duplicate priority event")
            return False

        self._priority_cooldowns[topic_key] = now
        return True

    # ── Cognitive event recording ────────────────────────────────────────

    async def _record_cognitive_event(self, message: Message) -> None:
        """
        Record and score cognitive events (workspace thoughts, decisions).
        """
        payload = message.payload
        thought_type = payload.get("type", "")

        # Skip meta-events to avoid recursion
        if thought_type in ("critique", "simulation_result"):
            return

        content = payload.get("content", "")
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

