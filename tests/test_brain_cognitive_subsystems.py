"""Integration test: BrainFactory creates Brain with all cognitive subsystems wired."""

from collections.abc import AsyncIterator

import pytest

from hbllm.actions.tool_memory import ToolMemory
from hbllm.brain.cognitive_metrics import CognitiveMetrics
from hbllm.brain.confidence_estimator import ConfidenceEstimator
from hbllm.brain.factory import BrainConfig, BrainFactory
from hbllm.brain.goal_manager import GoalManager
from hbllm.brain.revision_node import RevisionNode
from hbllm.brain.self_model import SelfModel
from hbllm.brain.skill_registry import SkillRegistry
from hbllm.brain.world_simulator import WorldSimulator
from hbllm.data.interaction_miner import InteractionMiner
from hbllm.memory.concept_extractor import ConceptExtractor
from hbllm.network.cognition_router import CognitionRouter
from hbllm.serving.provider import LLMProvider, LLMResponse
from hbllm.serving.token_optimizer import TokenOptimizer
from hbllm.training.policy_optimizer import PolicyOptimizer
from hbllm.training.reward_model import RewardModel

pytestmark = pytest.mark.asyncio


class _MockProvider(LLMProvider):
    """Mock provider implementing all abstract methods."""

    @property
    def name(self) -> str:
        return "mock"

    async def generate(
        self, messages, max_tokens=1024, temperature=0.7, top_p=0.9, **kw
    ) -> LLMResponse:
        return LLMResponse(
            content="Mock response about the topic.",
            model="mock",
            usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        )

    async def stream(self, messages, max_tokens=1024, temperature=0.7, **kw) -> AsyncIterator[str]:
        for word in "Mock response about the topic.".split():
            yield word + " "


class TestBrainCognitiveSubsystems:
    """Verify BrainFactory wires all cognitive subsystems into Brain."""

    @pytest.fixture
    async def brain(self, tmp_path):
        config = BrainConfig(
            inject_perception=False,
            data_dir=str(tmp_path),
        )
        brain = await BrainFactory.create(
            provider=_MockProvider(),
            config=config,
        )
        yield brain
        await brain.shutdown()

    # ─── Subsystem Initialization ────────────────────────────────────

    async def test_skill_registry_initialized(self, brain):
        assert brain.skill_registry is not None
        assert isinstance(brain.skill_registry, SkillRegistry)

    async def test_goal_manager_initialized(self, brain):
        assert brain.goal_manager is not None
        assert isinstance(brain.goal_manager, GoalManager)

    async def test_self_model_initialized(self, brain):
        assert brain.self_model is not None
        assert isinstance(brain.self_model, SelfModel)

    async def test_cognitive_metrics_initialized(self, brain):
        assert brain.cognitive_metrics is not None
        assert isinstance(brain.cognitive_metrics, CognitiveMetrics)

    async def test_world_simulator_initialized(self, brain):
        assert brain.world_simulator is not None
        assert isinstance(brain.world_simulator, WorldSimulator)

    async def test_revision_node_initialized(self, brain):
        assert brain.revision_node is not None
        assert isinstance(brain.revision_node, RevisionNode)

    async def test_confidence_estimator_initialized(self, brain):
        assert brain.confidence_estimator is not None
        assert isinstance(brain.confidence_estimator, ConfidenceEstimator)

    async def test_tool_memory_initialized(self, brain):
        assert brain.tool_memory is not None
        assert isinstance(brain.tool_memory, ToolMemory)

    async def test_concept_extractor_initialized(self, brain):
        assert brain.concept_extractor is not None
        assert isinstance(brain.concept_extractor, ConceptExtractor)

    async def test_cognition_router_initialized(self, brain):
        assert brain.cognition_router is not None
        assert isinstance(brain.cognition_router, CognitionRouter)

    async def test_token_optimizer_initialized(self, brain):
        assert brain.token_optimizer is not None
        assert isinstance(brain.token_optimizer, TokenOptimizer)

    async def test_reward_model_initialized(self, brain):
        assert brain.reward_model is not None
        assert isinstance(brain.reward_model, RewardModel)

    async def test_policy_optimizer_initialized(self, brain):
        assert brain.policy_optimizer is not None
        assert isinstance(brain.policy_optimizer, PolicyOptimizer)

    async def test_interaction_miner_initialized(self, brain):
        assert brain.interaction_miner is not None
        assert isinstance(brain.interaction_miner, InteractionMiner)

    # ─── Subsystem Functional Tests ──────────────────────────────────

    async def test_cognitive_stats_returns_all_subsystems(self, brain):
        stats = brain.cognitive_stats()
        assert "metrics" in stats
        assert "self_model" in stats
        assert "skills" in stats
        assert "goals" in stats
        assert "tool_memory" in stats
        assert "token_optimizer" in stats
        assert "rewards" in stats

    async def test_skill_registry_works(self, brain):
        skill = brain.skill_registry.extract_and_store(
            task_description="Test task",
            execution_trace=[{"action": "step1"}],
            tools_used=["browser"],
            success=True,
            category="test",
        )
        assert skill is not None
        assert brain.skill_registry.stats()["total_skills"] >= 1

    async def test_goal_manager_creates_goals(self, brain):
        goal = brain.goal_manager.create_goal("Test goal", "A test")
        assert goal is not None
        assert brain.goal_manager.stats()["total_goals"] == 1

    async def test_cognitive_metrics_record(self, brain):
        brain.cognitive_metrics.record_reasoning(0.85)
        brain.cognitive_metrics.record_latency(200.0, "test")
        snap = brain.cognitive_metrics.snapshot()
        assert snap.reasoning_score == 0.85

    async def test_self_model_tracks_domains(self, brain):
        brain.self_model.record_outcome("coding", success=True, confidence=0.9)
        cap = brain.self_model.get_capability("coding")
        assert cap is not None

    async def test_token_optimizer_routes_model(self, brain):
        result = brain.token_optimizer.optimize("Hello")
        assert result.recommended_model == "small"
        result = brain.token_optimizer.optimize(
            "Explain quantum entanglement and derive the equations"
        )
        assert result.recommended_model == "large"

    # ─── Config Toggling ─────────────────────────────────────────────

    async def test_disable_revision(self, tmp_path):
        config = BrainConfig(inject_revision=False, inject_perception=False, data_dir=str(tmp_path))
        brain = await BrainFactory.create(provider=_MockProvider(), config=config)
        assert brain.revision_node is None
        assert brain.confidence_estimator is None
        await brain.shutdown()

    async def test_disable_goals(self, tmp_path):
        config = BrainConfig(inject_goals=False, inject_perception=False, data_dir=str(tmp_path))
        brain = await BrainFactory.create(provider=_MockProvider(), config=config)
        assert brain.goal_manager is None
        await brain.shutdown()
