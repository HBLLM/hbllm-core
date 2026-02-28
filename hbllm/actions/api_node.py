"""
System 2 API Node (Tool Synthesis Engine).

Monitors the Workspace for queries that ask to generate schemas, 
OpenAPI specs, or payloads for REST requests. Uses the base LLM to 
synthesize structured JSON representations and posts them back to the Blackboard.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class ApiNode(Node):
    """
    Service node that generates structured API payloads and OpenAPI Schemas
    using the base LLM for dynamic extraction.
    """

    def __init__(self, node_id: str, llm=None):
        super().__init__(node_id=node_id, node_type=NodeType.DOMAIN_MODULE, capabilities=["openapi", "json_schema", "tool_synthesis"])
        self.llm = llm  # LLMInterface instance

    async def on_start(self) -> None:
        logger.info("Starting ApiNode (Tool Synthesis Engine)")
        await self.bus.subscribe("module.evaluate", self.evaluate_workspace_query)

    async def on_stop(self) -> None:
        logger.info("Stopping ApiNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def evaluate_workspace_query(self, message: Message) -> Message | None:
        """
        Triggered when a new query lands on the Global Workspace Blackboard.
        """
        payload = message.payload
        text = payload.get("text", "")
        
        if not self.llm:
            return None
        
        # 1. LLM-Based Intent Detection
        classification = await self.llm.generate_json(
            f"Determine if the following query is asking to generate an API schema, "
            f"JSON payload, OpenAPI specification, REST endpoint structure, or tool definition. "
            f"Query: \"{text}\"\n"
            f"Output JSON: {{\"is_api_request\": true/false, \"request_type\": \"schema|payload|openapi|tool\"}}"
        )
        
        if not classification.get("is_api_request", False):
            return None
            
        logger.info("[ApiNode] Detected API/Schema synthesis request.")
        
        # 2. LLM-Based Schema Generation
        try:
            request_type = classification.get("request_type", "schema")
            schema_content = await self._synthesize_schema(text, request_type)
            
            if not schema_content:
                return None
            
            # 3. Blackboard Proposal
            thought_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="workspace.thought",
                payload={
                    "type": "api_synthesis",
                    "confidence": 0.90,
                    "content": schema_content
                },
                correlation_id=message.correlation_id
            )
            await self.bus.publish("workspace.thought", thought_msg)
            
        except Exception as e:
            logger.error("[ApiNode] Schema synthesis failed: %s", e)
            
        return None

    async def _synthesize_schema(self, query: str, request_type: str) -> str | None:
        """
        Uses the LLM to generate the requested API schema or payload.
        """
        prompts = {
            "schema": (
                f"Generate a valid OpenAPI 3.0 Tool/Function JSON schema based on the "
                f"following description. Include parameter names, types, descriptions, "
                f"and required fields.\n\n"
                f"Description: \"{query}\"\n\n"
                f"Output ONLY valid JSON:"
            ),
            "payload": (
                f"Generate a complete example JSON request payload based on the following "
                f"API description. Include realistic example values.\n\n"
                f"Description: \"{query}\"\n\n"
                f"Output ONLY valid JSON:"
            ),
            "openapi": (
                f"Generate a minimal OpenAPI 3.0 specification (as JSON) for the following "
                f"API endpoint description. Include paths, methods, request/response schemas.\n\n"
                f"Description: \"{query}\"\n\n"
                f"Output ONLY valid JSON:"
            ),
            "tool": (
                f"Generate an LLM Tool/Function calling schema (compatible with OpenAI function "
                f"calling format) for the following description.\n\n"
                f"Description: \"{query}\"\n\n"
                f"Output ONLY valid JSON:"
            ),
        }
        
        prompt = prompts.get(request_type, prompts["schema"])
        raw = await self.llm.generate(prompt, max_tokens=512, temperature=0.3)
        
        if not raw or not raw.strip():
            return None
        
        # Wrap in code fences for clean presentation if not already
        cleaned = raw.strip()
        if not cleaned.startswith("```"):
            cleaned = f"```json\n{cleaned}\n```"
        
        return cleaned
