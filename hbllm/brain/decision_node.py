"""
System Decision Node (The Gatekeeper).

Subscribes to ``decision.evaluate``.
In the Global Workspace model, the LLM and Symbolic Solvers only pose "Thoughts."
This Node operates after the Workspace Blackboard has formed a consensus.
It uses the ActionPlanner to produce a structured ActionPlan, enforces tiered
safety checks, and dispatches the final command to the Agent Execution Layer
(Browser, Code Execution, Audio TTS, IoT, MCP, or directly to the User).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hbllm.brain.action_planner import ActionPlanner
from hbllm.brain.action_schema import ActionPlan, ActionType, RiskLevel
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

if TYPE_CHECKING:
    from hbllm.brain.policy_engine import PolicyEngine
    from hbllm.brain.provider_adapter import ProviderLLM

logger = logging.getLogger(__name__)


class DecisionNode(Node):
    """
    Service node that separates generation from execution.
    Uses an ActionPlanner for structured intent→action mapping,
    tiered safety classification, and PolicyEngine governance
    before dispatching to the appropriate execution channel.
    """

    def __init__(
        self,
        node_id: str,
        llm: ProviderLLM | None = None,
        policy_engine: PolicyEngine | None = None,
    ) -> None:
        super().__init__(node_id=node_id, node_type=NodeType.CORE)
        self.llm = llm  # LLMInterface instance
        self.policy_engine = policy_engine  # PolicyEngine instance
        self._planner = ActionPlanner()

    async def on_start(self) -> None:
        """Subscribe to the decision evaluations from the Workspace."""
        logger.info("Starting DecisionNode (The Gatekeeper)")
        await self.bus.subscribe("decision.evaluate", self.evaluate_workspace_decision)

    async def on_stop(self) -> None:
        logger.info("Stopping DecisionNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── Main Entry Point ──────────────────────────────────────────────────

    async def evaluate_workspace_decision(self, message: Message) -> Message | None:
        """
        Triggered when the Workspace has reached a consensus.
        Flow: plan → enforce safety → execute.
        """
        payload = message.payload
        original_query = payload.get("original_query", {})
        thought = payload.get("selected_thought", {})

        user_intent = original_query.get("intent", "answer")
        thought_type = thought.get("type", "intuition")
        confidence = thought.get("confidence", 0.0)
        content = thought.get("content") or ""

        logger.info(
            "[DecisionNode] Evaluating %s thought (Confidence: %s)...", thought_type, confidence
        )

        # ── 1. Plan the action ─────────────────────────────────────────────
        plan = self._planner.plan(
            intent=user_intent,
            thought_type=thought_type,
            content=content,
            confidence=confidence,
            original_query=original_query,
        )
        logger.info(
            "[DecisionNode] ActionPlan: %s (risk=%s)", plan.action_type.value, plan.risk_level.value
        )

        # ── 2. Enforce safety (tiered) ─────────────────────────────────────
        if not await self._enforce_safety(plan, message):
            return None  # blocked by policy or safety classifier

        # ── 3. Execute the action ──────────────────────────────────────────
        await self._execute_action(plan, message, original_query)

        return None

    # ── Safety Enforcement (Tiered) ───────────────────────────────────────

    async def _enforce_safety(self, plan: ActionPlan, message: Message) -> bool:
        """
        Apply tiered safety checks based on the action's risk level.

        Returns ``True`` if the action is approved, ``False`` if blocked.

        - LOW risk (text, audio, clarify): PolicyEngine only.
        - MEDIUM risk (web search, API): PolicyEngine only.
        - HIGH risk (code, IoT, MCP): PolicyEngine + LLM safety classifier.
        """
        content = plan.content

        # ── PolicyEngine (all risk levels) ─────────────────────────────────
        if self.policy_engine:
            try:
                result = self.policy_engine.evaluate(
                    text=content,
                    tenant_id=message.tenant_id or "default",
                    domain="",
                )
                if not result.passed:
                    violations = "; ".join(result.violations)
                    logger.warning("[DecisionNode] Policy violation: %s", violations)
                    await self._publish_output(
                        message,
                        f"I cannot fulfill this request — policy violation: {violations}",
                    )
                    return False
                # Apply any policy transformations (e.g., appended disclaimers)
                plan.content = result.modified_text
                if result.warnings:
                    logger.info("[DecisionNode] Policy warnings: %s", result.warnings)
            except Exception as e:
                logger.warning("[DecisionNode] PolicyEngine evaluation failed: %s", e)

        # ── LLM Safety Classifier (HIGH risk only) ─────────────────────────
        if plan.requires_safety_llm and self.llm:
            try:
                safety = await self.llm.generate_json(
                    f"You are a safety classifier for an AI system. Evaluate whether the following "
                    f"response content is safe to present to a user. Check for:\n"
                    f"- Harmful or dangerous instructions\n"
                    f"- Personal information exposure\n"
                    f"- Illegal activity guidance\n"
                    f"- Explicit or violent content\n\n"
                    f'Content: "{content[:500]}"\n\n'
                    f'Output JSON: {{"safe": true/false, "reason": "brief explanation"}}'
                )

                if not safety.get("safe", True):
                    reason = safety.get("reason", "Content flagged by safety classifier")
                    logger.warning("[DecisionNode] Thought rejected: %s", reason)
                    await self._publish_output(
                        message,
                        f"I cannot fulfill this request due to safety constraints: {reason}",
                    )
                    return False
            except Exception as e:
                logger.warning(
                    "[DecisionNode] Safety classification failed, proceeding cautiously: %s", e
                )

        return True

    # ── Action Execution (Dispatch) ───────────────────────────────────────

    async def _execute_action(
        self,
        plan: ActionPlan,
        message: Message,
        original_query: dict[str, Any],
    ) -> None:
        """Dispatch the approved ActionPlan to the appropriate execution channel."""

        dispatch = {
            ActionType.AUDIO_OUTPUT: self._exec_audio_output,
            ActionType.CODE_EXECUTION: self._exec_code_execution,
            ActionType.WEB_SEARCH: self._exec_web_search,
            ActionType.API_CALL: self._exec_api_call,
            ActionType.IOT_COMMAND: self._exec_iot_command,
            ActionType.MCP_TOOL: self._exec_mcp_tool,
            ActionType.CLARIFY: self._exec_clarify,
            ActionType.TEXT_RESPONSE: self._exec_text_response,
        }

        handler = dispatch.get(plan.action_type, self._exec_text_response)
        await handler(plan, message, original_query)

        # Record experience for salience detection
        await self.bus.publish(
            "system.experience",
            self._make_msg(
                message,
                "system.experience",
                {
                    "text": plan.content[:500],
                    "intent": original_query.get("intent", "answer"),
                    "thought_type": original_query.get("thought_type", ""),
                    "action_type": plan.action_type.value,
                },
            ),
        )

    # ── Individual Execution Handlers ─────────────────────────────────────

    async def _exec_text_response(
        self, plan: ActionPlan, message: Message, original_query: dict[str, Any]
    ) -> None:
        """Publish plain text to the user interface."""
        logger.info("[DecisionNode] Dispatching to User Interface.")
        thought_type = plan.metadata.get("thought_type", "intuition_general")
        await self._publish_output(message, plan.content, source=thought_type)

    async def _exec_audio_output(
        self, plan: ActionPlan, message: Message, original_query: dict[str, Any]
    ) -> None:
        """Dispatch text to the Audio Output Node for TTS."""
        logger.info("[DecisionNode] Dispatching to AudioOutputNode.")
        await self.bus.publish(
            "sensory.audio.out",
            self._make_msg(message, "sensory.audio.out", {"text": plan.content}),
        )
        await self._publish_output(message, plan.content, source="audio_speak")

    async def _exec_code_execution(
        self, plan: ActionPlan, message: Message, original_query: dict[str, Any]
    ) -> None:
        """Execute Python code in the sandbox and synthesize results."""
        logger.info("[DecisionNode] Dispatching to ExecutionNode Sandbox.")
        code = plan.content

        try:
            exec_msg = self._make_msg(
                message, "task.execute.python", {"code": code}, MessageType.QUERY
            )
            exec_resp = await self.bus.request("task.execute.python", exec_msg, timeout=15.0)
            if exec_resp.type == MessageType.ERROR:
                result_text = f"Execution failed: {exec_resp.payload.get('error')}"
            else:
                stdout = exec_resp.payload.get("output", "")
                stderr = exec_resp.payload.get("error", "")
                result_text = f"STDOUT:\n{stdout}"
                if stderr:
                    result_text += f"\nSTDERR:\n{stderr}"
        except Exception as e:
            result_text = f"Execution timeout or error: {e}"

        # Synthesize or directly present the result
        final_text = await self._synthesize_result(
            result_text,
            original_query,
            synthesis_prompt=(
                "You are a helpful assistant. Present the results of the Python code execution "
                "to the user. If the code printed the answer, explain it. If it failed, explain why.\n\n"
                f"User Query: {original_query.get('text', '')}\n\n"
                f"Code Executed:\n{code}\n\n"
                f"Execution Output:\n{result_text}"
            ),
            fallback_prefix="Executed code. Result:",
        )
        await self._publish_output(message, final_text, source="code_execution")

    async def _exec_web_search(
        self, plan: ActionPlan, message: Message, original_query: dict[str, Any]
    ) -> None:
        """Execute a web search, optionally resolving vague queries from context."""
        logger.info("[DecisionNode] Dispatching to BrowserNode.")
        query = plan.content

        # Resolve vague queries using conversation history
        if plan.metadata.get("needs_context_resolution", False):
            query = await self._resolve_vague_query(query, message)

        try:
            search_msg = self._make_msg(
                message,
                "task.execute.search",
                {"query": query, "max_results": plan.metadata.get("max_results", 3)},
                MessageType.QUERY,
            )
            search_resp = await self.bus.request("task.execute.search", search_msg, timeout=15.0)
            if search_resp.type == MessageType.ERROR:
                result_text = f"Search failed: {search_resp.payload.get('error')}"
            else:
                result_text = search_resp.payload.get("text", "No search results found.")
        except Exception as e:
            result_text = f"Search timeout or error: {e}"

        final_text = await self._synthesize_result(
            result_text,
            original_query,
            synthesis_prompt=(
                "You are a helpful assistant. Synthesize a comprehensive, direct response to the user's query "
                "based on the following real-time search results. Cite sources/URLs if available.\n\n"
                f"User Query: {original_query.get('text', query)}\n\n"
                f"Search Results:\n{result_text}"
            ),
            fallback_prefix="Search results:",
        )
        await self._publish_output(message, final_text, source="browser")

    async def _exec_api_call(
        self, plan: ActionPlan, message: Message, original_query: dict[str, Any]
    ) -> None:
        """Dispatch to the API execution node."""
        logger.info("[DecisionNode] Dispatching to ApiNode.")
        intent = plan.metadata.get("intent", "tool_synthesis")

        try:
            api_msg = self._make_msg(
                message,
                "task.execute.api",
                {"schema": plan.content, "intent": intent},
                MessageType.QUERY,
            )
            api_resp = await self.bus.request("task.execute.api", api_msg, timeout=15.0)
            if api_resp.type == MessageType.ERROR:
                result_text = f"API execution failed: {api_resp.payload.get('error')}"
            else:
                result_text = api_resp.payload.get("text", str(api_resp.payload))
        except Exception as e:
            result_text = f"API timeout or error: {e}"

        await self._publish_output(message, result_text, source="api_execution")

    async def _exec_iot_command(
        self, plan: ActionPlan, message: Message, original_query: dict[str, Any]
    ) -> None:
        """Dispatch an IoT command via MQTT."""
        logger.info("[DecisionNode] Dispatching to IoT MQTT Node.")
        iot_topic = plan.metadata.get("iot_topic", "hbllm/command")

        await self.bus.publish(
            "iot.publish",
            self._make_msg(
                message,
                "iot.publish",
                {"topic": iot_topic, "payload": plan.content},
            ),
        )
        await self._publish_output(
            message,
            f"Dispatched IoT command to topic: {iot_topic}",
            source="iot",
        )

    async def _exec_mcp_tool(
        self, plan: ActionPlan, message: Message, original_query: dict[str, Any]
    ) -> None:
        """Dispatch an MCP tool call."""
        logger.info("[DecisionNode] Dispatching to MCP Client Node.")
        try:
            mcp_msg = self._make_msg(
                message,
                "mcp.tool_call",
                {
                    "tool_name": plan.metadata.get("tool_name", ""),
                    "arguments": plan.metadata.get("arguments", {}),
                    "content": plan.content,
                },
                MessageType.QUERY,
            )
            mcp_resp = await self.bus.request("mcp.tool_call", mcp_msg, timeout=15.0)
            if mcp_resp.type == MessageType.ERROR:
                result_text = f"MCP tool call failed: {mcp_resp.payload.get('error')}"
            else:
                result_text = mcp_resp.payload.get("text", str(mcp_resp.payload))
        except Exception as e:
            result_text = f"MCP timeout or error: {e}"

        await self._publish_output(message, result_text, source="mcp")

    async def _exec_clarify(
        self, plan: ActionPlan, message: Message, original_query: dict[str, Any]
    ) -> None:
        """Ask the user to clarify when confidence is too low."""
        logger.info(
            "[DecisionNode] Confidence too low (%.2f), requesting clarification.",
            plan.metadata.get("confidence", 0.0),
        )
        await self._publish_output(
            message,
            "I'm not confident enough in my understanding of your request. Could you please clarify or provide more details?",
            source="clarify",
        )

    # ── Shared Helpers ────────────────────────────────────────────────────

    async def _resolve_vague_query(self, query: str, message: Message) -> str:
        """Resolve a vague search query using conversation history from episodic memory."""
        try:
            tenant_id = message.tenant_id or "default"
            session_id = message.session_id or "default"

            req_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                tenant_id=tenant_id,
                topic="memory.retrieve_recent",
                payload={"session_id": session_id, "limit": 6, "tenant_id": tenant_id},
            )
            resp = await self.bus.request("memory.retrieve_recent", req_msg, timeout=3.0)
            turns = resp.payload.get("turns", [])

            if turns and self.llm:
                history_str = ""
                for turn in turns:
                    role = turn.get("role", "user")
                    content_str = turn.get("content", "")
                    history_str += f"{role.capitalize()}: {content_str}\n"

                resolver_prompt = (
                    "You are an AI assistant helping to resolve search queries. "
                    "The user has requested to search for something, but their query is vague or context-dependent (e.g. 'search it'). "
                    "Based on the recent conversation history, determine what specific, clear search query they intend. "
                    "Do not output any introductory or concluding remarks. Respond ONLY with the resolved search query.\n\n"
                    f"Conversation History:\n{history_str}\n"
                    f"User Request: {query}\n\n"
                    "Resolved Search Query:"
                )
                resolved_query = await self.llm.generate(resolver_prompt)
                resolved_query = resolved_query.strip().strip("\"'")
                if resolved_query:
                    logger.info(
                        "[DecisionNode] Resolved vague query '%s' to '%s'", query, resolved_query
                    )
                    return resolved_query
        except Exception as e:
            logger.warning("[DecisionNode] Failed to resolve vague query context: %s", e)

        return query

    async def _synthesize_result(
        self,
        result_text: str,
        original_query: dict[str, Any],
        synthesis_prompt: str,
        fallback_prefix: str,
    ) -> str:
        """Use the LLM to synthesize raw results into a user-friendly response."""
        if self.llm:
            try:
                return await self.llm.generate(synthesis_prompt)
            except Exception as e:
                logger.warning("Synthesis failed: %s", e)
        return f"{fallback_prefix}\n{result_text}"

    def _make_msg(
        self,
        original: Message,
        topic: str,
        payload: dict[str, Any],
        msg_type: MessageType = MessageType.EVENT,
    ) -> Message:
        """Create a new Message inheriting identifiers from the original."""
        return Message(
            type=msg_type,
            source_node_id=self.node_id,
            tenant_id=original.tenant_id,
            session_id=original.session_id,
            topic=topic,
            payload=payload,
            correlation_id=original.correlation_id,
        )

    async def _publish_output(self, message: Message, text: str, source: str = "decision") -> None:
        """Publish a response to ``sensory.output``."""
        await self.bus.publish(
            "sensory.output",
            self._make_msg(message, "sensory.output", {"text": text, "source": source}),
        )
