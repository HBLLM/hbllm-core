"""
Experience Node — records interactions and detects salience.

Maps to flowchart nodes H (Experience Recorder), I (Salience Detector),
and J (High Importance? decision).
"""

import logging

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)

class ExperienceNode(Node):
    """
    Experience Recorder & Salience Detector (Flowchart Nodes H, I, J).
    
    Archives all system interactions and scores them for importance.
    High-importance experiences are flagged for priority memory and reflection.
    """

    def __init__(self, node_id: str, llm=None):
        super().__init__(node_id=node_id, node_type=NodeType.ROUTER)
        self.llm = llm  # LLMInterface for complex saliency scoring
        self.importance_threshold = 0.7

    async def on_start(self) -> None:
        """Subscribe to all sensory outputs and key cognitive events to record experiences."""
        logger.info("Starting ExperienceNode (Recorder & Salience Detector)")
        await self.bus.subscribe("sensory.output", self.record_experience)
        # Expanded observation scope: also monitor workspace thoughts and decision evaluations
        await self.bus.subscribe("workspace.thought", self._record_cognitive_event)
        await self.bus.subscribe("decision.evaluate", self._record_cognitive_event)

    async def on_stop(self) -> None:
        logger.info("Stopping ExperienceNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def record_experience(self, message: Message) -> None:
        """
        Record the interaction and detect its salience (importance).
        """
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
                "content": content[:1000],  # Trim to prevent huge logs
                "domain": "experience",
                "metadata": {"source": "experience_recorder"}
            },
            correlation_id=corr_id
        )
        await self.bus.publish("memory.store", store_msg)
        
        # 2. Salience Detector (Node I)
        score = await self._calculate_saliency(content, payload)
        
        # 3. High Importance? (Node J)
        is_priority = score >= self.importance_threshold
        
        logger.info("[ExperienceNode] Salience score: %.2f (Priority: %s)", score, is_priority)

        # Publish salience score for other nodes (Memory, Meta)
        salience_msg = Message(
            type=MessageType.SALIENCE_SCORE,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic="system.salience",
            payload={
                "message_id": corr_id,
                "score": score,
                "is_priority": is_priority,
                "content": content
            },
            correlation_id=corr_id
        )
        await self.bus.publish("system.salience", salience_msg)

    async def _calculate_saliency(self, content: str, payload: dict) -> float:
        """
        Heuristic + LLM based importance scoring.
        """
        # Basic heuristics
        score = 0.5
        
        # Priority markers
        if any(word in content.lower() for word in ["critical", "error", "failure", "important", "alert"]):
            score += 0.3
            
        # If LLM is available, use it for deeper semantic saliency
        if self.llm:
            try:
                result = await self.llm.generate_json(
                    f"Evaluate the importance of this interaction for long-term learning.\n"
                    f"Content: \"{content[:500]}\"\n\n"
                    f"Output JSON: {{\"importance\": 0.0-1.0}}"
                )
                score = float(result.get("importance", score))
            except Exception as e:
                logger.warning("LLM salience scoring failed: %s", e)
                
        return min(max(score, 0.0), 1.0)

    async def _record_cognitive_event(self, message: Message) -> None:
        """
        Record cognitive events (workspace thoughts, decisions) for richer learning signal.
        Skip meta-events like critiques to avoid loops.
        """
        payload = message.payload
        thought_type = payload.get("type", "")
        
        # Skip critiques and simulation results to avoid recursion
        if thought_type in ("critique", "simulation_result"):
            return
        
        content = payload.get("content", "")
        if not content:
            return
        
        logger.debug("[ExperienceNode] Recording cognitive event: %s", thought_type)
