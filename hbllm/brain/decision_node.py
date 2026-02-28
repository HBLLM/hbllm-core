"""
System Decision Node (The Gatekeeper).

Subscribes to `decision.evaluate`. 
In the Global Workspace model, the LLM and Symbolic Solvers only pose "Thoughts."
This Node operates after the Workspace Blackboard has formed a consensus.
It uses the LLM to evaluate the incoming thought for safety and alignment,
and if approved, dispatches the final command down to the Agent 
Execution Layer (Browser, Code Execution, Audio TTS, or directly to the User).
"""

from __future__ import annotations

import logging

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class DecisionNode(Node):
    """
    Service node that separates generation from execution.
    Uses LLM-based safety classification instead of keyword checks.
    """

    def __init__(self, node_id: str, llm=None):
        super().__init__(node_id=node_id, node_type=NodeType.ROUTER)
        self.llm = llm  # LLMInterface instance

    async def on_start(self) -> None:
        """Subscribe to the decision evaluations from the Workspace."""
        logger.info("Starting DecisionNode (The Gatekeeper)")
        await self.bus.subscribe("decision.evaluate", self.evaluate_workspace_decision)

    async def on_stop(self) -> None:
        logger.info("Stopping DecisionNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def evaluate_workspace_decision(self, message: Message) -> Message | None:
        """
        Triggered when the Workspace has reached a consensus.
        """
        payload = message.payload
        original_query = payload.get("original_query", {})
        thought = payload.get("selected_thought", {})
        
        user_intent = original_query.get("intent", "answer")
        thought_type = thought.get("type", "intuition")
        confidence = thought.get("confidence", 0.0)
        content = thought.get("content", "")
        
        logger.info("[DecisionNode] Evaluating %s thought (Confidence: %s)...", thought_type, confidence)
        
        # 1. LLM-Based Safety Classification
        if self.llm:
            safety = await self.llm.generate_json(
                f"You are a safety classifier for an AI system. Evaluate whether the following "
                f"response content is safe to present to a user. Check for:\n"
                f"- Harmful or dangerous instructions\n"
                f"- Personal information exposure\n"
                f"- Illegal activity guidance\n"
                f"- Explicit or violent content\n\n"
                f"Content: \"{content[:500]}\"\n\n"
                f"Output JSON: {{\"safe\": true/false, \"reason\": \"brief explanation\"}}"
            )
            
            if not safety.get("safe", True):
                reason = safety.get("reason", "Content flagged by safety classifier")
                logger.warning("[DecisionNode] Thought rejected: %s", reason)
                err_msg = Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    tenant_id=message.tenant_id,
                    session_id=message.session_id,
                    topic="sensory.output", 
                    payload={"text": f"I cannot fulfill this request due to safety constraints: {reason}"},
                    correlation_id=message.correlation_id
                )
                await self.bus.publish("sensory.output", err_msg)
                return None
            
        # 2. Agent Execution Layer Routing
        if user_intent == "speak" or original_query.get("force_audio", False):
            logger.info("[DecisionNode] Dispatching Thought to AudioOutputNode.")
            audio_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="sensory.audio.out",
                payload={"text": content},
                correlation_id=message.correlation_id
            )
            await self.bus.publish("sensory.audio.out", audio_msg)
            
        elif "```python" in content:
            logger.info("[DecisionNode] Dispatching Thought to ExecutionNode Sandbox.")
            code = content.split("```python")[1].split("```")[0].strip()
            exec_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="task.execute.python",
                payload={"code": code},
                correlation_id=message.correlation_id
            )
            await self.bus.publish("task.execute.python", exec_msg)
            
        else:
            logger.info("[DecisionNode] Dispatching Thought directly to User Interface.")
            ui_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="sensory.output",
                payload={"text": content, "source": thought_type},
                correlation_id=message.correlation_id
            )
            await self.bus.publish("sensory.output", ui_msg)
            
        return None
