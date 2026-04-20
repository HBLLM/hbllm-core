"""
Cognitive Router Node.

Analyzes incoming user queries using the base LLM to determine intent,
and routes the message to the Global Workspace for multi-node evaluation.
If the query spans multiple domains, it delegates to the PlannerNode.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

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
        self,
        node_id: str,
        default_domain: str = "general",
        llm: Any = None,
        use_vectors: bool = True,
        domain_registry: Any | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.ROUTER,
            capabilities=["routing", "intent_classification"],
        )
        self.default_domain = default_domain
        self.topic_sub = "router.query"
        self.llm = llm  # LLMInterface instance

        # Hierarchical domain registry
        if domain_registry is None:
            from hbllm.modules.domain_registry import DomainRegistry

            domain_registry = DomainRegistry()
        self.domain_registry = domain_registry

        # Derive known_domains from registry for backward compatibility
        self.known_domains: set[str] = set(self.domain_registry.all_domains)
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
        self.unknown_counts: dict[str, int] = defaultdict(int)

        # Idle tracking for SleepNode
        self.last_activity_time = time.time()

        # Vector Routing Setup
        self.use_vectors = use_vectors
        self.domain_centroids: dict[str, Any] = {}
        self.encoder: Any | None = None
        self._tokenizer: Any | None = None
        self._onnx_inputs: list[str] = []

        # Weighted routing config
        self._softmax_temperature: float = 0.1
        self._max_blend_domains: int = 3
        self._min_blend_weight: float = 0.05

        # [Improvement 1] Dynamic Temperature — auto-adjusts based on score spread
        self._dynamic_temperature: bool = True
        self._temp_min: float = 0.05  # Very sharp (near-argmax)
        self._temp_max: float = 0.5  # Broad blending for ambiguous queries

        # [Improvement 2] Contextual Routing — use conversation history
        self._context_window: int = 3  # Number of recent messages to include
        self._session_history: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=10))

        # [Improvement 5] Confidence Calibration — Platt scaling params
        # sigmoid(a * score + b) — trained via feedback, defaults are identity-ish
        self._platt_a: float = 5.0  # Steepness
        self._platt_b: float = -2.0  # Shift (so 0.4 raw → ~0.5 calibrated)

        # [Improvement 8] Persistent Centroids — survive restarts
        self._centroids_path: Path | None = None  # Set by factory

    def _bootstrap_centroids(self) -> None:
        """Bootstrap domain embeddings from the DomainRegistry."""
        if not self.use_vectors:
            return

        if self.encoder is None:
            import onnxruntime as ort  # type: ignore[import-untyped]
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer  # type: ignore[import-untyped]

            logger.info("Initializing ONNX Edge Vector Model (paraphrase-MiniLM-L3-v2)")

            # Using INT8 quantized model for extreme low-memory footprint (~15MB RAM)
            model_path = hf_hub_download(
                repo_id="Xenova/paraphrase-MiniLM-L3-v2", filename="onnx/model_quantized.onnx"
            )
            tokenizer_path = hf_hub_download(
                repo_id="Xenova/paraphrase-MiniLM-L3-v2", filename="tokenizer.json"
            )

            self._tokenizer = Tokenizer.from_file(tokenizer_path)
            if self._tokenizer:
                self._tokenizer.enable_truncation(max_length=128)
                self._tokenizer.enable_padding(length=128)

            # Start strict C++ CPU Execution Provider
            self.encoder = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            if self.encoder:
                self._onnx_inputs = [i.name for i in self.encoder.get_inputs()]

        # Bootstrap from registry centroid texts
        for domain_name, centroid_text in self.domain_registry.centroid_texts().items():
            emb = self._encode_text(centroid_text)
            self.domain_centroids[domain_name] = emb

        # [Improvement 8] Load persisted centroids (overwrite bootstrapped ones)
        self._load_centroids()

    def _encode_text(self, text: str) -> Any:
        """Encodes text using the ONNX lightweight CPU architecture natively."""
        import numpy as np

        if self._tokenizer is None or self.encoder is None:
            return np.zeros(384)  # Default dimension for MiniLM

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
        """Save learned centroids on shutdown."""
        logger.info("Stopping RouterNode")
        self._save_centroids()

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
        target_domain: Any = self.default_domain
        confidence = 0.5
        intent = "general_knowledge"

        if self.use_vectors:
            if not self.domain_centroids:
                self._bootstrap_centroids()

            import numpy as np

            # [Improvement 2] Contextual Routing — encode with conversation history
            session_id = message.session_id or ""
            context_text = text
            if session_id and session_id in self._session_history:
                recent = list(self._session_history[session_id])[-self._context_window :]
                if recent:
                    context_text = " | ".join(recent) + " | " + text

            query_emb = self._encode_text(context_text)

            # Store this query in session history for future context
            if session_id:
                self._session_history[session_id].append(text[:200])

            scores = {}
            raw_scores_map: dict[str, float] = {}
            for domain, centroid in self.domain_centroids.items():
                norm_q = float(np.linalg.norm(query_emb))
                norm_c = float(np.linalg.norm(centroid))
                if norm_q > 0 and norm_c > 0:
                    raw_score = float(np.dot(query_emb, centroid) / (norm_q * norm_c))
                    raw_scores_map[domain] = raw_score
                    # [Improvement 5] Platt Confidence Calibration (for routing decisions)
                    scores[domain] = self._calibrate(raw_score)

            # Sort domains by calibrated score
            sorted_domains = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            if not sorted_domains:
                target_domain = "general"
                confidence = 0.5
            else:
                top_domain, top_score = sorted_domains[0]
                # Use raw score for confidence (monotonic with centroid distance)
                confidence = raw_scores_map.get(top_domain, top_score)

                if top_score > self.unknown_threshold:
                    # Filter domains above noise floor
                    eligible = [(d, s) for d, s in sorted_domains if s > self.unknown_threshold]

                    if len(eligible) <= 1:
                        target_domain = top_domain
                    else:
                        # [Improvement 1] Dynamic Temperature
                        temperature = self._softmax_temperature
                        if self._dynamic_temperature and len(eligible) >= 2:
                            score_spread = eligible[0][1] - eligible[-1][1]
                            # Low spread = ambiguous → higher temp → more blending
                            # High spread = clear winner → lower temp → sharper
                            temperature = self._temp_min + (
                                (self._temp_max - self._temp_min)
                                * (1.0 - min(score_spread / 0.3, 1.0))
                            )

                        # Softmax-weighted top-2 (max 3) blend
                        top_n = eligible[: self._max_blend_domains]
                        raw_scores = np.array([s for _, s in top_n])

                        # Temperature-scaled softmax
                        exp_scores = np.exp((raw_scores - raw_scores.max()) / temperature)
                        weights = exp_scores / exp_scores.sum()

                        # Build blend dict, pruning domains below 5%
                        blend: dict[str, float] = {}
                        for (d, _), w in zip(top_n, weights):
                            if float(w) >= self._min_blend_weight:
                                blend[d] = round(float(w), 3)

                        if len(blend) <= 1:
                            target_domain = top_domain
                        else:
                            target_domain = blend
                            logger.info(
                                "Vector Router softmax blend (top-%d): %s",
                                len(blend),
                                target_domain,
                            )
                else:
                    target_domain = "general"

            logger.debug(
                "Vector Router chose domain '%s' with confidence %.3f",
                target_domain,
                confidence,
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

                if classification and "error" not in classification:
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
                    if topic_result:
                        topic_guess = topic_result.get("topic", str(topic_guess))

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

        # [Improvement 4] Router Distillation — emit routing decision for SleepNode
        try:
            distill_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="system.router_decision",
                payload={
                    "query": text[:200],
                    "domain": target_domain if isinstance(target_domain, str) else target_domain,
                    "confidence": confidence,
                    "intent": intent,
                },
                correlation_id=message.correlation_id or message.id,
            )
            await self.bus.publish("system.router_decision", distill_msg)
        except Exception:
            pass  # Non-critical — don't fail routing over telemetry

        return None

    async def _handle_feedback(self, message: Message) -> Message | None:
        """
        Adapt routing heuristics based on user feedback.
        Positive feedback → lower unknown_threshold (more permissive).
        Negative feedback → raise threshold (more cautious, trigger spawns sooner).
        """
        rating: int = int(message.payload.get("rating", 0))
        domain: str = str(message.payload.get("domain", ""))
        prompt: str = str(message.payload.get("prompt", ""))

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
                # [Improvement 6] Negative Contrastive — push AWAY from wrong domain
                new_centroid = current_centroid - (alpha * query_emb)
                logger.info(
                    "RouterNode pushed centroid for domain '%s' away from query (Negative Rating)",
                    domain,
                )

                # Also push the query TOWARDS the correct domain if provided
                correct_domain: str = str(message.payload.get("correct_domain", ""))
                if correct_domain and correct_domain in self.domain_centroids:
                    correct_centroid = self.domain_centroids[correct_domain]
                    self.domain_centroids[correct_domain] = (
                        1 - alpha
                    ) * correct_centroid + alpha * query_emb
                    # Normalize the corrected centroid
                    c_norm = float(np.linalg.norm(self.domain_centroids[correct_domain]))
                    if c_norm > 0:
                        self.domain_centroids[correct_domain] /= c_norm
                    logger.info(
                        "RouterNode pulled centroid for '%s' towards query (Contrastive Correction)",
                        correct_domain,
                    )
            else:
                new_centroid = current_centroid

            # Normalize vector
            norm = float(np.linalg.norm(new_centroid))
            if norm > 0:
                new_centroid = new_centroid / norm
            self.domain_centroids[domain] = new_centroid

            # [Improvement 8] Persist updated centroids
            self._save_centroids()

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

        # Keep known_domains in sync with registry
        self.known_domains = set(self.domain_registry.all_domains)

        logger.debug(
            "RouterNode threshold adjusted to %.2f after feedback (rating=%d)",
            self.unknown_threshold,
            rating,
        )
        return None

    async def _handle_domain_registered(self, message: Message) -> Message | None:
        """Auto-learn new domains when SpawnerNode creates them."""
        from hbllm.modules.domain_registry import DomainSpec

        domain: str = str(message.payload.get("domain", ""))
        if domain and not self.domain_registry.exists(domain):
            centroid = str(message.payload.get("centroid_text", f"Topics relating to {domain}"))
            self.domain_registry.register(DomainSpec(name=domain, centroid_text=centroid))
            self.known_domains = set(self.domain_registry.all_domains)
            logger.info("RouterNode registered new domain: %s", domain)
            if self.use_vectors:
                self.domain_centroids[domain] = self._encode_text(centroid)
        return None

    async def _handle_swarm_transfer(self, message: Message) -> Message | None:
        """
        Phase 12: Intercept autonomous multi-agent Swarm Transfers.
        Receive historical context, update target domain, and republish to the Workspace
        under the same correlation_id so the user session never drops.
        """
        target_domain: str = str(message.payload.get("target_domain", self.default_domain))
        original_query: dict[str, Any] = dict(message.payload.get("original_query", {}))
        history: list[dict[str, Any]] = list(message.payload.get("history", []))

        logger.info(
            "Router executing Swarm Handoff. Bouncing session %s from Workspace back to domain '%s'",
            str(message.correlation_id),
            target_domain,
        )

        # We rewrite the query payload to include the historical work of the previous agents!
        workspace_payload = original_query.copy()
        workspace_payload["domain_hint"] = target_domain
        workspace_payload["intent"] = "complex_reasoning"
        workspace_payload["swarm_history"] = history

        # Inject context directly into the text for the next agent to read
        history_text = "\n".join(
            [
                f"- [{h.get('node', 'unknown')}]: {h.get('type', 'event')} (confidence: {h.get('confidence', 0.0)})"
                for h in history
            ]
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

    # ── Helper Methods ────────────────────────────────────────────────────

    def _calibrate(self, raw_score: float) -> float:
        """
        [Improvement 5] Platt Confidence Calibration.

        Converts raw cosine similarity into a calibrated probability via
        sigmoid(a * score + b).  This produces better-calibrated confidence
        values for downstream blending decisions.
        """
        import math

        return 1.0 / (1.0 + math.exp(-(self._platt_a * raw_score + self._platt_b)))

    def _save_centroids(self) -> None:
        """
        [Improvement 8] Persist learned centroids to disk.

        Saves as a JSON file mapping domain names to centroid vectors.
        Called on shutdown and after feedback-driven centroid updates.
        """
        if self._centroids_path is None or not self.domain_centroids:
            return

        try:
            import numpy as np

            data = {
                "version": 2,
                "timestamp": time.time(),
                "threshold": self.unknown_threshold,
                "platt_a": self._platt_a,
                "platt_b": self._platt_b,
                "centroids": {
                    name: centroid.tolist() if isinstance(centroid, np.ndarray) else centroid
                    for name, centroid in self.domain_centroids.items()
                },
            }
            self._centroids_path.parent.mkdir(parents=True, exist_ok=True)
            self._centroids_path.write_text(json.dumps(data), encoding="utf-8")
            logger.debug(
                "Saved %d centroids to %s", len(self.domain_centroids), self._centroids_path
            )
        except Exception as e:
            logger.warning("Failed to save centroids: %s", e)

    def _load_centroids(self) -> None:
        """
        [Improvement 8] Load persisted centroids from disk.

        Overwrites bootstrapped centroids with learned ones.
        """
        if self._centroids_path is None or not self._centroids_path.exists():
            return

        try:
            import numpy as np

            data = json.loads(self._centroids_path.read_text(encoding="utf-8"))

            if data.get("version", 1) >= 2:
                # Restore calibration params
                self.unknown_threshold = data.get("threshold", self.unknown_threshold)
                self._platt_a = data.get("platt_a", self._platt_a)
                self._platt_b = data.get("platt_b", self._platt_b)

            centroids = data.get("centroids", {})
            loaded = 0
            for name, vec in centroids.items():
                self.domain_centroids[name] = np.array(vec, dtype=np.float32)
                loaded += 1

            logger.info(
                "Loaded %d persisted centroids from %s (threshold=%.2f)",
                loaded,
                self._centroids_path,
                self.unknown_threshold,
            )
        except Exception as e:
            logger.warning("Failed to load centroids: %s", e)
