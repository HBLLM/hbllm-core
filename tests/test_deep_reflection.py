"""
Tests for Deep Reflection — ExperienceNode's _write_reflection and
end-to-end reflection pipeline via bus.
"""

import asyncio
import json
import pytest
import shutil
import tempfile
from pathlib import Path

from hbllm.brain.experience_node import ExperienceNode
from hbllm.memory.memory_node import MemoryNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


@pytest.fixture
def tmp_reflection_dir():
    """Create a temp directory for reflection output."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
async def reflection_system(tmp_reflection_dir):
    """Boot ExperienceNode + MemoryNode connected via bus."""
    bus = InProcessBus()
    await bus.start()

    experience = ExperienceNode(
        node_id="experience",
        importance_threshold=0.3,  # Low threshold so test content triggers priority
        reflection_dir=tmp_reflection_dir,
    )
    memory = MemoryNode(node_id="memory", db_path=":memory:")

    await experience.start(bus)
    await memory.start(bus)

    yield bus, experience, memory, tmp_reflection_dir

    await experience.stop()
    await memory.stop()
    await bus.stop()


# ── Unit tests for reflection analysis ───────────────────────────────────────

class TestEventCategorization:
    def test_security_category(self):
        node = ExperienceNode(node_id="test", reflection_dir="/tmp/test_ref")
        assert node._categorize_event("critical security breach detected in auth module") == "security"

    def test_error_category(self):
        node = ExperienceNode(node_id="test", reflection_dir="/tmp/test_ref")
        assert node._categorize_event("unhandled exception in payment processing") == "error"

    def test_performance_category(self):
        node = ExperienceNode(node_id="test", reflection_dir="/tmp/test_ref")
        assert node._categorize_event("database query timeout after 30 seconds of latency") == "performance"

    def test_user_preference_category(self):
        node = ExperienceNode(node_id="test", reflection_dir="/tmp/test_ref")
        assert node._categorize_event("user prefers dark mode and always uses compact view") == "user_preference"

    def test_learning_category(self):
        node = ExperienceNode(node_id="test", reflection_dir="/tmp/test_ref")
        assert node._categorize_event("discovered a new pattern in user behavior insights") == "learning"

    def test_general_fallback(self):
        node = ExperienceNode(node_id="test", reflection_dir="/tmp/test_ref")
        assert node._categorize_event("today is a great day") == "general"


class TestCausalAnalysis:
    def test_finds_because_causes(self):
        node = ExperienceNode(node_id="test", reflection_dir="/tmp/test_ref")
        result = node._heuristic_causal_analysis(
            "The server crashed because the memory limit was exceeded. This resulted in data loss.",
            "the server crashed because the memory limit was exceeded. this resulted in data loss.",
        )
        assert len(result["likely_causes"]) >= 1
        assert "memory" in result["likely_causes"][0].lower()
        assert result["method"] == "heuristic"

    def test_no_explicit_causal_language_gets_hypothesis(self):
        node = ExperienceNode(node_id="test", reflection_dir="/tmp/test_ref")
        result = node._heuristic_causal_analysis(
            "High CPU usage observed at 3am. Alert triggered.",
            "high cpu usage observed at 3am. alert triggered.",
        )
        assert len(result["likely_causes"]) >= 1
        assert "Triggered by:" in result["likely_causes"][0]


class TestCounterfactual:
    def test_security_counterfactual(self):
        node = ExperienceNode(node_id="test", reflection_dir="/tmp/test_ref")
        result = node._heuristic_counterfactual("security breach in authentication", "security")
        assert "prompt" in result
        assert "chosen" in result
        assert "rejected" in result
        assert len(result["chosen"]) > len(result["rejected"])
        assert "Isolate" in result["chosen"] or "security" in result["chosen"].lower()

    def test_error_counterfactual(self):
        node = ExperienceNode(node_id="test", reflection_dir="/tmp/test_ref")
        result = node._heuristic_counterfactual("database exception thrown", "error")
        assert "stack trace" in result["chosen"].lower() or "root cause" in result["chosen"].lower()
        assert result["rejected"] == "That's a known issue, just retry."


class TestEntityExtraction:
    def test_extracts_capitalized_phrases(self):
        node = ExperienceNode(node_id="test", reflection_dir="/tmp/test_ref")
        entities = node._extract_key_entities("The Google Cloud Platform integration failed")
        labels = [e["label"] for e in entities]
        assert any("Google Cloud Platform" in l for l in labels)

    def test_extracts_technical_terms(self):
        node = ExperienceNode(node_id="test", reflection_dir="/tmp/test_ref")
        entities = node._extract_key_entities("The camelCase and snake_case variables need refactoring")
        labels = [e["label"] for e in entities]
        assert any("camelCase" in l for l in labels)
        assert any("snake_case" in l for l in labels)

    def test_extracts_quoted_references(self):
        node = ExperienceNode(node_id="test", reflection_dir="/tmp/test_ref")
        entities = node._extract_key_entities('The "authentication service" needs updates')
        labels = [e["label"] for e in entities]
        assert any("authentication service" in l for l in labels)

    def test_extracts_domain_concepts(self):
        node = ExperienceNode(node_id="test", reflection_dir="/tmp/test_ref")
        entities = node._extract_key_entities("The training pipeline node handles checkpoint gradient updates")
        types = {e["type"] for e in entities}
        assert "domain_concept" in types

    def test_caps_at_20(self):
        node = ExperienceNode(node_id="test", reflection_dir="/tmp/test_ref")
        # Generate content with many entities
        content = " ".join(f'"entity_{i}"' for i in range(30))
        entities = node._extract_key_entities(content)
        assert len(entities) <= 20


class TestRuleExtraction:
    def test_when_then_rule(self):
        node = ExperienceNode(node_id="test", reflection_dir="/tmp/test_ref")
        rules = node._extract_reflection_rules(
            "When the cache is invalidated, then the system should refetch from database",
            "when the cache is invalidated, then the system should refetch from database",
            "performance",
        )
        assert len(rules) >= 1
        assert rules[0]["condition"]
        assert rules[0]["action"]

    def test_leads_to_rule(self):
        node = ExperienceNode(node_id="test", reflection_dir="/tmp/test_ref")
        rules = node._extract_reflection_rules(
            "excessive logging output leads to disk space exhaustion and system failure",
            "excessive logging output leads to disk space exhaustion and system failure",
            "error",
        )
        assert len(rules) >= 1

    def test_security_category_boosts_confidence(self):
        node = ExperienceNode(node_id="test", reflection_dir="/tmp/test_ref")
        rules_security = node._extract_reflection_rules(
            "When unauthorized access occurs, then lock the account immediately",
            "when unauthorized access occurs, then lock the account immediately",
            "security",
        )
        rules_general = node._extract_reflection_rules(
            "When unauthorized access occurs, then lock the account immediately",
            "when unauthorized access occurs, then lock the account immediately",
            "general",
        )
        if rules_security and rules_general:
            assert rules_security[0]["confidence"] > rules_general[0]["confidence"]


# ── Integration: JSONL writing + bus publishing ──────────────────────────────

class TestDeepReflectionIntegration:
    @pytest.mark.asyncio
    async def test_reflection_writes_jsonl(self, reflection_system):
        bus, experience, memory, ref_dir = reflection_system

        # Send a high-salience event
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            tenant_id="t1",
            session_id="s1",
            topic="sensory.output",
            payload={"text": "Critical security breach detected in authentication module! Unauthorized access from unknown IP."},
            correlation_id="ref_001",
        )
        await bus.publish("sensory.output", msg)
        await asyncio.sleep(0.5)

        # Check JSONL file was written
        jsonl_path = Path(ref_dir) / "reflections.jsonl"
        assert jsonl_path.exists(), f"Expected {jsonl_path} to exist"

        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) >= 1

        entry = json.loads(lines[0])
        assert entry["category"] == "security"
        assert entry["severity"] in ("critical", "high")
        assert "causal_analysis" in entry
        assert "counterfactual" in entry
        assert "entities" in entry
        assert "rules" in entry

    @pytest.mark.asyncio
    async def test_reflection_publishes_to_bus(self, reflection_system):
        bus, experience, memory, ref_dir = reflection_system

        reflections = []
        async def capture(msg):
            reflections.append(msg)
        await bus.subscribe("system.reflection", capture)

        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="sensory.output",
            payload={"text": "Fatal exception in payment processing module. Traceback shows null pointer error."},
            correlation_id="ref_002",
        )
        await bus.publish("sensory.output", msg)
        await asyncio.sleep(0.5)

        assert len(reflections) >= 1
        payload = reflections[0].payload
        assert payload["category"] == "error"
        assert "counterfactual" in payload
        assert payload["counterfactual"]["method"] in ("heuristic", "llm")

    @pytest.mark.asyncio
    async def test_counterfactual_has_dpo_fields(self, reflection_system):
        bus, experience, memory, ref_dir = reflection_system

        reflections = []
        async def capture(msg):
            reflections.append(msg)
        await bus.subscribe("system.reflection", capture)

        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="sensory.output",
            payload={"text": "Security vulnerability discovered in the API gateway! Exploit allows unauthorized access."},
            correlation_id="ref_003",
        )
        await bus.publish("sensory.output", msg)
        await asyncio.sleep(0.5)

        assert len(reflections) >= 1
        cf = reflections[0].payload["counterfactual"]
        assert "prompt" in cf
        assert "chosen" in cf
        assert "rejected" in cf
        assert len(cf["chosen"]) > len(cf["rejected"])

    @pytest.mark.asyncio
    async def test_reflection_updates_knowledge_graph(self, reflection_system):
        bus, experience, memory, ref_dir = reflection_system

        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="sensory.output",
            payload={"text": "The authentication module has a critical vulnerability. Python is a programming language used by the system."},
            correlation_id="ref_004",
        )
        await bus.publish("sensory.output", msg)
        await asyncio.sleep(0.5)

        # KnowledgeGraph should have ingested entities from the reflection
        assert memory.knowledge_graph.entity_count >= 1
