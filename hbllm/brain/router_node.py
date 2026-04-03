"""
Cognitive Router Node.

Analyzes incoming user queries using the base LLM to determine intent,
and routes the message to the Global Workspace for multi-node evaluation.
If the query spans multiple domains, it delegates to the PlannerNode.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from hbllm.network.messages import (
    Message,
    MessageType,
    QueryPayload,
    SpawnRequestPayload,
)
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class RouterNode(Node):
    """
    Central router that uses LLM-based intent classification to direct
    traffic to the correct specialized experts via the Global Workspace.
    """

    def __init__(
        self, node_id: str, default_domain: str = "general", llm=None, use_vectors: bool = True
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.ROUTER,
            capabilities=["routing", "intent_classification"],
        )
        self.default_domain = default_domain
        self.topic_sub = "router.query"
        self.llm = llm  # LLMInterface instance

        # Dynamic domain registry (updated by SpawnerNode when new modules are created)
        self.known_domains: set[str] = {
            "general",
            "coding",
            "math",
            "planner",
            "api_synth",
            "fuzzy",
        }
        self.known_intents: set[str] = {
            "general_knowledge",
            "code_generation",
            "math_reasoning",
            "complex_reasoning",
            "web_search",
            "tool_synthesis",
            "fuzzy_reasoning",
            "unknown_topic",
        }

        # Self-expansion tracking
        self.unknown_threshold = 0.3
        self.spawn_trigger_count = 2
        self.unknown_counts = defaultdict(int)

        # Idle tracking for SleepNode
        self.last_activity_time = time.time()

        # Vector Routing Setup
        self.use_vectors = use_vectors
        self.domain_centroids = {}
        self.encoder = None

    def _bootstrap_centroids(self):
        """Bootstrap default domain embeddings to initialize the vector space."""
        if not self.use_vectors:
            return

        if self.encoder is None:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            logger.info("Initializing ONNX Edge Vector Model (paraphrase-MiniLM-L3-v2)")

            # Using INT8 quantized model for extreme low-memory footprint (~15MB RAM)
            model_path = hf_hub_download(
                repo_id="Xenova/paraphrase-MiniLM-L3-v2", filename="onnx/model_quantized.onnx"
            )
            tokenizer_path = hf_hub_download(
                repo_id="Xenova/paraphrase-MiniLM-L3-v2", filename="tokenizer.json"
            )

            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            self._tokenizer.enable_truncation(max_length=128)
            self._tokenizer.enable_padding(length=128)

            # Start strict C++ CPU Execution Provider
            self.encoder = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self._onnx_inputs = [i.name for i in self.encoder.get_inputs()]

        for d in self.known_domains:
            # Synthetic query generation
            if d == "general":
                text_repr = "Hello, can you help me with a general question about daily life, chatting, or common facts?"
            elif d == "coding":
                text_repr = "Write a python script, fix this bug, create an HTML React component, explain this logic error."
            elif d == "math":
                text_repr = "Calculate the integral, solve this equation, what is the square root, number theory."
            elif d == "planner":
                text_repr = "Can you design a multi-step plan, architect a system layout, or outline the workflow?"
            elif d == "api_synth":
                text_repr = "Make a POST request, fetch data from the REST backend, build an endpoint payload."
            else:
                text_repr = f"I have a specific query regarding {d} topics."

            emb = self._encode_text(text_repr)
            self.domain_centroids[d] = emb

    def _encode_text(self, text: str):
        """Encodes text using the ONNX lightweight CPU architecture natively."""
        import numpy as np

        enc = self._tokenizer.encode(text)
        ort_inputs = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        }

        if "token_type_ids" in self._onnx_inputs:
            ort_inputs["token_type_ids"] = np.array([enc.type_ids], dtype=np.int64)

        outputs = self.encoder.run(None, ort_inputs)
        last_hidden_state = outputs[0]

        # Mean Pooling
        attention_mask = ort_inputs["attention_mask"].astype(np.float32)
        expanded_mask = np.expand_dims(attention_mask, axis=-1)

        sum_embeddings = np.sum(last_hidden_state * expanded_mask, axis=1)
        sum_mask = np.clip(np.sum(expanded_mask, axis=1), a_min=1e-9, a_max=None)

        sentence_emb = sum_embeddings / sum_mask
        # L2 Normalize
        sentence_emb = sentence_emb / np.linalg.norm(sentence_emb, axis=1, keepdims=True)
        return sentence_emb[0]

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

        # 1. Similarity-Based Classification
        target_domain = self.default_domain
        confidence = 0.5
        intent = "general_knowledge"

        if self.use_vectors:
            if not self.domain_centroids:
                self._bootstrap_centroids()

            import numpy as np

            query_emb = self._encode_text(text)

            scores = {}
            for domain, centroid in self.domain_centroids.items():
                norm_q = np.linalg.norm(query_emb)
                norm_c = np.linalg.norm(centroid)
                if norm_q > 0 and norm_c > 0:
                    scores[domain] = float(np.dot(query_emb, centroid) / (norm_q * norm_c))

            # Sort domains by score
            sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            if not sorted_domains:
                target_domain = "general"
                confidence = 0.5
            else:
                top_domain, top_score = sorted_domains[0]
                confidence = top_score

                if top_score > self.unknown_threshold:
                    # Check if second place is close enough for MoE blending
                    if len(sorted_domains) > 1:
                        second_domain, second_score = sorted_domains[1]
                        # Only blend if second score is also very high and relatively close
                        if (
                            second_score > self.unknown_threshold
                            and (top_score - second_score) < 0.15
                        ):
                            total = top_score + second_score
                            target_domain = {
                                top_domain: round(top_score / total, 3),
                                second_domain: round(second_score / total, 3),
                            }
                            logger.info(
                                "Vector Router triggered MoE Hybrid mapping: %s", target_domain
                            )
                        else:
                            target_domain = top_domain
                    else:
                        target_domain = top_domain
                else:
                    target_domain = "general"

            logger.debug(
                "Vector Router chose domain '%s' with confidence %.3f", target_domain, confidence
            )
        elif self.llm:
            # Fallback to LLM if vector encoder isn't available
            domains_str = ", ".join(sorted(self.known_domains))
            intents_str = ", ".join(sorted(self.known_intents))
            try:
                classification = await self.llm.generate_json(
                    f"You are an intent classifier for a modular AI system. Classify the following "
                    f"query into exactly one domain and intent.\n\n"
                    f"Available domains: {domains_str}\n"
                    f"Available intents: {intents_str}\n\n"
                    f'Query: "{text}"\n\n'
                    f'Output JSON: {{"domain": "...", "intent": "...", "confidence": 0.0-1.0}}'
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
            logger.warning(
                "Router confidence too low (%.2f) for query: '%s'. Marking unknown.",
                confidence,
                text[:30],
            )
            self.unknown_counts["general_unknown"] += 1

            if self.unknown_counts["general_unknown"] >= self.spawn_trigger_count:
                logger.warning("Unknown threshold reached! Triggering Module Spawning...")

                # Extract topic name via LLM
                topic_guess = (
                    target_domain if target_domain != self.default_domain else "new_domain"
                )
                if self.llm:
                    topic_result = await self.llm.generate_json(
                        f"What academic or technical domain does this query belong to? "
                        f'Query: "{text}"\n'
                        f'Output JSON: {{"topic": "one_word_domain_name"}}'
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
                        topic=topic_guess, trigger_query=text, confidence_score=confidence
                    ).model_dump(),
                )
                await self.bus.publish("system.spawn", spawn_req)

                self.unknown_counts["general_unknown"] = 0

                return message.create_response(
                    {
                        "text": f"I don't know much about this topic yet. I am spawning a new '{topic_guess}' module to learn about it. Please try asking again in a few moments!",
                        "domain": "system",
                    }
                )
        else:
            if self.unknown_counts["general_unknown"] > 0:
                self.unknown_counts["general_unknown"] -= 1

        # 3. Publish to Global Workspace
        logger.info(
            "Routing to Workspace with dominant intent: %s (confidence: %.2f)", intent, confidence
        )

        workspace_payload = message.payload.copy()
        workspace_payload["intent"] = intent
        workspace_payload["domain_hint"] = target_domain
        workspace_payload["confidence"] = confidence

        routed_query = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic="workspace.update",
            payload=workspace_payload,
            correlation_id=message.correlation_id or message.id,
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
        prompt = message.payload.get("prompt", "")

        # Self-Learning Embedding Update
        if self.use_vectors and domain in self.domain_centroids and prompt:
            import numpy as np

            query_emb = self._encode_text(prompt)
            current_centroid = self.domain_centroids[domain]

            # EMA alpha: how aggressively a single piece of feedback shifts the centroid
            alpha = 0.1

            if rating > 0:
                # Pull centroid towards the successful query
                new_centroid = (1 - alpha) * current_centroid + alpha * query_emb
                logger.info(
                    "RouterNode pulled centroid for domain '%s' towards query (Positive Rating)",
                    domain,
                )
            elif rating < 0:
                # Push centroid away from the unsuccessful query
                new_centroid = current_centroid - (alpha * query_emb)
                logger.info(
                    "RouterNode pushed centroid for domain '%s' away from query (Negative Rating)",
                    domain,
                )
            else:
                new_centroid = current_centroid

            # Normalize vector
            norm = np.linalg.norm(new_centroid)
            if norm > 0:
                new_centroid = new_centroid / norm
            self.domain_centroids[domain] = new_centroid

        # Threshold logic
        if rating > 0:
            # Positive: routing was good, become slightly more permissive
            self.unknown_threshold = max(0.15, self.unknown_threshold - 0.02)
            if domain and domain not in self.known_domains:
                self.known_domains.add(domain)
                logger.info("RouterNode learned new domain from feedback: %s", domain)
                if self.use_vectors:
                    self.domain_centroids[domain] = self._encode_text(
                        f"Topics relating to {domain}"
                    )
        elif rating < 0:
            # Negative: routing may have been wrong, become slightly more cautious
            self.unknown_threshold = min(0.6, self.unknown_threshold + 0.02)

        logger.debug(
            "RouterNode threshold adjusted to %.2f after feedback (rating=%d)",
            self.unknown_threshold,
            rating,
        )
        return None

    async def _handle_domain_registered(self, message: Message) -> Message | None:
        """Auto-learn new domains when SpawnerNode creates them."""
        domain = message.payload.get("domain", "")
        if domain and domain not in self.known_domains:
            self.known_domains.add(domain)
            logger.info("RouterNode registered new domain: %s", domain)
            if self.use_vectors:
                self.domain_centroids[domain] = self._encode_text(f"Topics relating to {domain}")
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
            message.correlation_id,
            target_domain,
        )

        # We rewrite the query payload to include the historical work of the previous agents!
        workspace_payload = original_query.copy()
        workspace_payload["domain_hint"] = target_domain
        workspace_payload["intent"] = "complex_reasoning"
        workspace_payload["swarm_history"] = history

        # Inject context directly into the text for the next agent to read
        history_text = "\n".join(
            [f"- [{h['node']}]: {h['type']} (confidence: {h['confidence']})" for h in history]
        )
        workspace_payload["text"] = (
            f"[SWARM TRANSFER] The previous agent transferred this to you ({target_domain}).\n\nPrevious Agent Work:\n{history_text}\n\nOriginal Request:\n{original_query.get('text', '')}"
        )

        routed_query = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic="workspace.update",
            payload=workspace_payload,
            correlation_id=message.correlation_id,
        )
        await self.bus.publish("workspace.update", routed_query)

        return None
