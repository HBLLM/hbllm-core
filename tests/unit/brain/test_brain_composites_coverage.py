"""Unit tests for brain composite nodes, simulation, skill induction, web research, world model."""

import ast
from unittest.mock import AsyncMock, MagicMock

import pytest

from hbllm.brain.composites.governance_guard import GovernanceGuard
from hbllm.brain.composites.learning_loop import LearningLoop
from hbllm.brain.composites.resource_manager import ResourceManager
from hbllm.brain.composites.social_layer import SocialLayer


class TestLearningLoop:
    def test_init(self):
        node = LearningLoop(node_id="learning_loop")
        assert node.node_id == "learning_loop"

    @pytest.mark.asyncio
    async def test_health_check(self):
        node = LearningLoop(node_id="learning_loop")
        health = await node.health_check()
        assert health is not None  # Returns NodeHealth object


class TestResourceManager:
    def test_init(self):
        node = ResourceManager(node_id="resource_mgr")
        assert node.node_id == "resource_mgr"

    @pytest.mark.asyncio
    async def test_health_check(self):
        node = ResourceManager(node_id="resource_mgr")
        health = await node.health_check()
        assert health is not None


class TestGovernanceGuard:
    def test_init(self):
        node = GovernanceGuard(node_id="governance")
        assert node.node_id == "governance"

    @pytest.mark.asyncio
    async def test_health_check(self):
        node = GovernanceGuard(node_id="governance")
        health = await node.health_check()
        assert health is not None


class TestSocialLayer:
    def test_init(self):
        node = SocialLayer(node_id="social")
        assert node.node_id == "social"

    @pytest.mark.asyncio
    async def test_health_check(self):
        node = SocialLayer(node_id="social")
        health = await node.health_check()
        assert health is not None


# ── Environment Simulation ───────────────────────────────────────────────────

from hbllm.brain.simulation.environment_sim import EnvironmentSimulator


class TestEnvironmentSimulator:
    def test_init(self):
        bus = MagicMock()
        sim = EnvironmentSimulator(bus=bus, tick_rate_seconds=0.1)
        assert sim is not None

    @pytest.mark.asyncio
    async def test_start_stop(self):
        bus = MagicMock()
        bus.publish = AsyncMock()
        sim = EnvironmentSimulator(bus=bus, tick_rate_seconds=60.0)
        await sim.start()
        await sim.stop()


# ── Skill Induction Node ─────────────────────────────────────────────────────

from hbllm.brain.skills.skill_induction_node import SecurityInterceptor, SkillInductionNode


class TestSecurityInterceptor:
    def test_safe_code(self):
        interceptor = SecurityInterceptor()
        tree = ast.parse("x = 1 + 2")
        interceptor.visit(tree)
        assert len(interceptor.errors) == 0

    def test_dangerous_import(self):
        interceptor = SecurityInterceptor()
        tree = ast.parse("import os")
        interceptor.visit(tree)
        assert len(interceptor.errors) > 0

    def test_dangerous_from_import(self):
        interceptor = SecurityInterceptor()
        tree = ast.parse("from subprocess import call")
        interceptor.visit(tree)
        assert len(interceptor.errors) > 0


class TestSkillInductionNode:
    def test_init(self):
        node = SkillInductionNode(node_id="skill_inductor")
        assert node.node_id == "skill_inductor"

    @pytest.mark.asyncio
    async def test_on_stop(self):
        node = SkillInductionNode(node_id="skill_inductor")
        await node.on_stop()


# ── Web Research Node ────────────────────────────────────────────────────────

from hbllm.brain.world.web_research_node import ResearchFinding, ResearchTier, classify_tier


class TestResearchTier:
    def test_tiers_exist(self):
        assert ResearchTier.INFORMATION is not None
        assert ResearchTier.TASK_KNOWLEDGE is not None
        assert ResearchTier.CORE_KNOWLEDGE is not None

    def test_classify_tier(self):
        tier = classify_tier("simple question")
        assert isinstance(tier, ResearchTier)


class TestResearchFinding:
    def test_creation(self):
        finding = ResearchFinding(
            content="test content",
            url="https://example.com",
            title="Test",
            domain="example.com",
            tier=ResearchTier.INFORMATION,
            trust_score=0.8,
        )
        assert finding.url == "https://example.com"
        assert finding.trust_score == 0.8


# ── World Model Node ─────────────────────────────────────────────────────────

from hbllm.brain.world.world_model_node import WorldModelNode


class TestWorldModelNode:
    def test_init(self):
        node = WorldModelNode(node_id="world_model")
        assert node.node_id == "world_model"

    def test_simulate_parse_imports(self):
        node = WorldModelNode(node_id="world_model")
        imports = node.simulate_parse_imports(
            "test.py", "import os\nimport sys\nfrom pathlib import Path"
        )
        assert isinstance(imports, list)

    def test_simulate_compilation(self):
        node = WorldModelNode(node_id="world_model")
        result = node.simulate_compilation("test.py")
        assert isinstance(result, dict)

    def test_simulate_ast(self):
        node = WorldModelNode(node_id="world_model")
        result = node._simulate_ast("x = 1 + 2\nprint(x)")
        assert isinstance(result, dict)


# ── LLM Interface ────────────────────────────────────────────────────────────

from hbllm.brain.core.llm_interface import LLMInterface


class TestLLMInterface:
    def test_extract_json_valid(self):
        text = 'Here is some text {"key": "value"} and more'
        result = LLMInterface._extract_json(text)
        assert result == {"key": "value"}

    def test_extract_json_no_json(self):
        text = "No JSON here"
        result = LLMInterface._extract_json(text)
        assert isinstance(result, dict)

    def test_extract_json_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = LLMInterface._extract_json(text)
        assert isinstance(result, dict)
