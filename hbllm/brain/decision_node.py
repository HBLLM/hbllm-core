"""
System Decision Node (The Gatekeeper).

Subscribes to `decision.evaluate`. 
In the Global Workspace model, the LLM and Symbolic Solvers only pose "Thoughts."
This Node operates after the Workspace Blackboard has formed a consensus.
It uses the LLM to evaluate the incoming thought for safety and alignment,
and if approved, dispatches the final command down to the Agent 
Execution Layer (Browser, Code Execution, Audio TTS, IoT, MCP, or directly to the User).
"""

from __future__ import annotations

import logging
import re

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class DecisionNode(Node):
    """
    Service node that separates generation from execution.
    Uses LLM-based safety classification and PolicyEngine governance
    before dispatching to the appropriate execution channel.
    """

    def __init__(self, node_id: str, llm=None, policy_engine=None):
        super().__init__(node_id=node_id, node_type=NodeType.ROUTER)
        self.llm = llm  # LLMInterface instance
        self.policy_engine = policy_engine  # PolicyEngine instance

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
        
        # ── 1. PolicyEngine Governance ──
        if self.policy_engine:
            try:
                result = self.policy_engine.evaluate(
                    text=content,
                    tenant_id=message.tenant_id or "default",
                    domain=original_query.get("domain_hint", ""),
                )
                if not result.passed:
                    violations = "; ".join(result.violations)
                    logger.warning("[DecisionNode] Policy violation: %s", violations)
                    err_msg = Message(
                        type=MessageType.EVENT,
                        source_node_id=self.node_id,
                        tenant_id=message.tenant_id,
                        session_id=message.session_id,
                        topic="sensory.output",
                        payload={"text": f"I cannot fulfill this request — policy violation: {violations}"},
                        correlation_id=message.correlation_id,
                    )
                    await self.bus.publish("sensory.output", err_msg)
                    return None
                # Apply any policy transformations (e.g., appended disclaimers)
                content = result.modified_text
                if result.warnings:
                    logger.info("[DecisionNode] Policy warnings: %s", result.warnings)
            except Exception as e:
                logger.warning("[DecisionNode] PolicyEngine evaluation failed: %s", e)
        
        # ── 2. LLM-Based Safety Classification ──
        if self.llm:
            try:
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
            except Exception as e:
                logger.warning("[DecisionNode] Safety classification failed, proceeding cautiously: %s", e)
            
        # ── 3. Agent Execution Layer Routing ──
        await self._route_to_execution(message, user_intent, content, thought_type, original_query)
            
        return None

    async def _route_to_execution(
        self, message: Message, intent: str, content: str, thought_type: str, original_query: dict
    ) -> None:
        """Route the approved thought to the appropriate execution channel."""
        tenant_id = message.tenant_id
        session_id = message.session_id
        correlation_id = message.correlation_id
        
        def _make_msg(topic: str, payload: dict, msg_type: MessageType = MessageType.EVENT) -> Message:
            return Message(
                type=msg_type,
                source_node_id=self.node_id,
                tenant_id=tenant_id,
                session_id=session_id,
                topic=topic,
                payload=payload,
                correlation_id=correlation_id,
            )
        
        # Audio output
        if intent == "speak" or original_query.get("force_audio", False):
            logger.info("[DecisionNode] Dispatching to AudioOutputNode.")
            await self.bus.publish("sensory.audio.out", _make_msg(
                "sensory.audio.out", {"text": content}
            ))
        
        # Python code execution
        elif "```python" in content:
            logger.info("[DecisionNode] Dispatching to ExecutionNode Sandbox.")
            code = ""
            match = re.search(r'```python\s*\n(.*?)\n```', content, re.DOTALL)
            if match:
                code = match.group(1).strip()
            else:
                try:
                    code = content.split("```python")[1].split("```")[0].strip()
                except (IndexError, ValueError):
                    code = content
            await self.bus.publish("task.execute.python", _make_msg(
                "task.execute.python", {"code": code}, MessageType.QUERY
            ))
        
        # Web search
        elif intent == "web_search" or original_query.get("force_search", False):
            logger.info("[DecisionNode] Dispatching to BrowserNode.")
            query = original_query.get("text", content[:200])
            await self.bus.publish("task.execute.search", _make_msg(
                "task.execute.search", {"query": query, "max_results": 3}, MessageType.QUERY
            ))
        
        # API synthesis / execution
        elif thought_type == "api_synthesis" or intent == "tool_synthesis":
            logger.info("[DecisionNode] Dispatching to ApiNode.")
            await self.bus.publish("task.execute.api", _make_msg(
                "task.execute.api", {"schema": content, "intent": intent}, MessageType.QUERY
            ))
        
        # IoT / MQTT commands
        elif intent == "iot_command" or original_query.get("iot_topic"):
            logger.info("[DecisionNode] Dispatching to IoT MQTT Node.")
            await self.bus.publish("iot.publish", _make_msg(
                "iot.publish", {
                    "topic": original_query.get("iot_topic", "hbllm/command"),
                    "payload": content,
                }
            ))
        
        # MCP tool calls
        elif intent == "mcp_tool" or original_query.get("mcp_tool_name"):
            logger.info("[DecisionNode] Dispatching to MCP Client Node.")
            await self.bus.publish("mcp.tool_call", _make_msg(
                "mcp.tool_call", {
                    "tool_name": original_query.get("mcp_tool_name", ""),
                    "arguments": original_query.get("mcp_arguments", {}),
                    "content": content,
                }, MessageType.QUERY
            ))
        
        # Default: text output to user
        else:
            logger.info("[DecisionNode] Dispatching to User Interface.")
            await self.bus.publish("sensory.output", _make_msg(
                "sensory.output", {"text": content, "source": thought_type}
            ))
        
        # Record experience for salience detection
        await self.bus.publish("system.experience", _make_msg(
            "system.experience", {
                "text": content[:500],
                "intent": intent,
                "thought_type": thought_type,
            }
        ))

