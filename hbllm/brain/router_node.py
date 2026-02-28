"""
Cognitive Router Node.

Analyzes incoming user queries using the base LLM to determine intent, 
and routes the message to the Global Workspace for multi-node evaluation.
If the query spans multiple domains, it delegates to the PlannerNode.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from collections import defaultdict
from hbllm.network.messages import Message, MessageType, QueryPayload, RouteDecisionPayload, SpawnRequestPayload
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class RouterNode(Node):
    """
    Central router that uses LLM-based intent classification to direct 
    traffic to the correct specialized experts via the Global Workspace.
    """

    def __init__(self, node_id: str, default_domain: str = "general", llm=None):
        super().__init__(node_id=node_id, node_type=NodeType.ROUTER, capabilities=["routing", "intent_classification"])
        self.default_domain = default_domain
        self.topic_sub = "router.query"
        self.llm = llm  # LLMInterface instance
        
        # Self-expansion tracking
        self.unknown_threshold = 0.3
        self.spawn_trigger_count = 2
        self.unknown_counts = defaultdict(int)
        
        # Idle tracking for SleepNode
        self.last_activity_time = time.time()

    async def on_start(self) -> None:
        """Subscribe to router queries."""
        logger.info("Starting RouterNode")
        await self.bus.subscribe(self.topic_sub, self.handle_message)

    async def on_stop(self) -> None:
        logger.info("Stopping RouterNode")

    async def handle_message(self, message: Message) -> Message | None:
        """Route the incoming query to the appropriate subsystem."""
        if message.type != MessageType.QUERY:
            return None

        try:
            payload = QueryPayload(**message.payload)
        except Exception as e:
            return message.create_error(f"Invalid QueryPayload: {e}")

        text = payload.text
        logger.info("Router analyzing query: '%s...'", text[:30])
        
        self.last_activity_time = time.time()

        # 1. LLM-Based Intent Classification
        target_domain = self.default_domain
        confidence = 0.5
        intent = "general_knowledge"

        if self.llm:
            classification = await self.llm.generate_json(
                f"You are an intent classifier for a modular AI system. Classify the following "
                f"query into exactly one domain and intent.\n\n"
                f"Available domains: general, coding, math, planner, api_synth, fuzzy\n"
                f"Available intents: general_knowledge, code_generation, math_reasoning, "
                f"complex_reasoning, web_search, tool_synthesis, fuzzy_reasoning, unknown_topic\n\n"
                f"Query: \"{text}\"\n\n"
                f"Output JSON: {{\"domain\": \"...\", \"intent\": \"...\", \"confidence\": 0.0-1.0}}"
            )
            
            if "error" not in classification:
                target_domain = classification.get("domain", self.default_domain)
                intent = classification.get("intent", "general_knowledge")
                try:
                    confidence = float(classification.get("confidence", 0.5))
                except (ValueError, TypeError):
                    confidence = 0.5

        # 2. Self-Expansion Logic
        if confidence < self.unknown_threshold:
            logger.warning("Router confidence too low (%.2f) for query: '%s'. Marking unknown.", confidence, text[:30])
            self.unknown_counts["general_unknown"] += 1
            
            if self.unknown_counts["general_unknown"] >= self.spawn_trigger_count:
                logger.warning("Unknown threshold reached! Triggering Module Spawning...")
                
                # Extract topic name via LLM
                topic_guess = target_domain if target_domain != self.default_domain else "new_domain"
                if self.llm:
                    topic_result = await self.llm.generate_json(
                        f"What academic or technical domain does this query belong to? "
                        f"Query: \"{text}\"\n"
                        f"Output JSON: {{\"topic\": \"one_word_domain_name\"}}"
                    )
                    topic_guess = topic_result.get("topic", topic_guess)
                        
                spawn_req = Message(
                    type=MessageType.SPAWN_REQUEST,
                    source_node_id=self.node_id,
                    target_node_id="", 
                    tenant_id=message.tenant_id,
                    session_id=message.session_id,
                    topic="system.spawn",
                    payload=SpawnRequestPayload(
                        topic=topic_guess,
                        trigger_query=text,
                        confidence_score=confidence
                    ).model_dump()
                )
                await self.bus.publish("system.spawn", spawn_req)
                
                self.unknown_counts["general_unknown"] = 0
                
                return message.create_response({
                    "text": f"I don't know much about this topic yet. I am spawning a new '{topic_guess}' module to learn about it. Please try asking again in a few moments!",
                    "domain": "system"
                })
        else:
            if self.unknown_counts["general_unknown"] > 0:
                self.unknown_counts["general_unknown"] -= 1

        # 3. Publish to Global Workspace
        logger.info("Routing to Workspace with dominant intent: %s (confidence: %.2f)", intent, confidence)

        workspace_payload = message.payload.copy()
        workspace_payload["intent"] = intent
        workspace_payload["domain_hint"] = target_domain
        
        routed_query = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic="workspace.update",
            payload=workspace_payload,
            correlation_id=message.id
        )
        await self.bus.publish("workspace.update", routed_query)
        
        return None
