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
        
        # Dynamic domain registry (updated by SpawnerNode when new modules are created)
        self.known_domains: set[str] = {"general", "coding", "math", "planner", "api_synth", "fuzzy"}
        self.known_intents: set[str] = {
            "general_knowledge", "code_generation", "math_reasoning",
            "complex_reasoning", "web_search", "tool_synthesis", "fuzzy_reasoning", "unknown_topic"
        }
        
        # Self-expansion tracking
        self.unknown_threshold = 0.3
        self.spawn_trigger_count = 2
        self.unknown_counts = defaultdict(int)
        
        # Idle tracking for SleepNode
        self.last_activity_time = time.time()

    async def on_start(self) -> None:
        """Subscribe to router queries and feedback for adaptive routing."""
        logger.info("Starting RouterNode")
        await self.bus.subscribe(self.topic_sub, self.handle_message)
        await self.bus.subscribe("system.feedback", self._handle_feedback)
        await self.bus.subscribe("system.domain_registered", self._handle_domain_registered)
        # Phase 12: Swarm integration
        await self.bus.subscribe("system.swarm.transfer", self._handle_swarm_transfer)

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
            domains_str = ", ".join(sorted(self.known_domains))
            intents_str = ", ".join(sorted(self.known_intents))
            try:
                classification = await self.llm.generate_json(
                    f"You are an intent classifier for a modular AI system. Classify the following "
                    f"query into exactly one domain and intent.\n\n"
                    f"Available domains: {domains_str}\n"
                    f"Available intents: {intents_str}\n\n"
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
            except Exception as e:
                logger.warning("Router LLM classification failed, using defaults: %s", e)

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
            correlation_id=message.correlation_id or message.id
        )
        await self.bus.publish("workspace.update", routed_query)
        
        return None

    async def _handle_feedback(self, message: Message) -> Message | None:
        """
        Adapt routing heuristics based on user feedback.
        Positive feedback → lower unknown_threshold (more permissive).
        Negative feedback → raise threshold (more cautious, trigger spawns sooner).
        """
        rating = message.payload.get("rating", 0)
        domain = message.payload.get("domain", "")
        
        if rating > 0:
            # Positive: routing was good, become slightly more permissive
            self.unknown_threshold = max(0.15, self.unknown_threshold - 0.02)
            if domain and domain not in self.known_domains:
                self.known_domains.add(domain)
                logger.info("RouterNode learned new domain from feedback: %s", domain)
        elif rating < 0:
            # Negative: routing may have been wrong, become slightly more cautious
            self.unknown_threshold = min(0.6, self.unknown_threshold + 0.02)
        
        logger.debug(
            "RouterNode threshold adjusted to %.2f after feedback (rating=%d)",
            self.unknown_threshold, rating,
        )
        return None

    async def _handle_domain_registered(self, message: Message) -> Message | None:
        """Auto-learn new domains when SpawnerNode creates them."""
        domain = message.payload.get("domain", "")
        if domain:
            self.known_domains.add(domain)
            logger.info("RouterNode registered new domain: %s", domain)
        return None

    async def _handle_swarm_transfer(self, message: Message) -> Message | None:
        """
        Phase 12: Intercept autonomous multi-agent Swarm Transfers.
        Receive historical context, update target domain, and republish to the Workspace
        under the same correlation_id so the user session never drops.
        """
        target_domain = message.payload.get("target_domain", self.default_domain)
        original_query = message.payload.get("original_query", {})
        history = message.payload.get("history", [])
        
        logger.info(
            "Router executing Swarm Handoff. Bouncing session %s from Workspace back to domain '%s'", 
            message.correlation_id, target_domain
        )
        
        # We rewrite the query payload to include the historical work of the previous agents!
        workspace_payload = original_query.copy()
        workspace_payload["domain_hint"] = target_domain
        workspace_payload["intent"] = "complex_reasoning"
        workspace_payload["swarm_history"] = history
        
        # Inject context directly into the text for the next agent to read
        history_text = "\n".join([f"- [{h['node']}]: {h['type']} (confidence: {h['confidence']})" for h in history])
        workspace_payload["text"] = f"[SWARM TRANSFER] The previous agent transferred this to you ({target_domain}).\n\nPrevious Agent Work:\n{history_text}\n\nOriginal Request:\n{original_query.get('text', '')}"
        
        routed_query = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic="workspace.update",
            payload=workspace_payload,
            correlation_id=message.correlation_id
        )
        await self.bus.publish("workspace.update", routed_query)
        
        return None
