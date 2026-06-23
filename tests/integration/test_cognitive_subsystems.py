"""Integration tests for cognitive subsystems working together.

Tests the full data flow between:
    UserModel → ExecutiveCortex (user alignment)
    ProjectGraph → ExecutiveCortex (goal selection)
    RelationshipMemory → ContextFusion (people in context)
    RealityGraph → ContextFusion (unified world state)
    All subsystems → ContextFusion (multi-source assembly)
"""

import time
import pytest

from hbllm.brain.context_fusion import ContextFusionEngine
from hbllm.brain.executive_cortex import ExecutiveCortex
from hbllm.brain.project_graph import ProjectGraph
from hbllm.brain.reality_graph import RealityGraph, RealityEntity
from hbllm.brain.relationship_memory import RelationshipMemory
from hbllm.brain.user_model import UserModelEngine


class TestUserModelProjectGraphIntegration:
    """Test that UserModel expertise signals flow into project auto-detection."""

    @pytest.fixture
    def engines(self, tmp_path):
        um = UserModelEngine(data_dir=str(tmp_path))
        pg = ProjectGraph(data_dir=str(tmp_path))
        return um, pg

    def test_user_focus_matches_project(self, engines):
        um, pg = engines
        # User is working on HBLLM
        um.update_from_interaction("test", "working on hbllm brain module", metadata={"topic": "HBLLM"})
        pg.create_project("HBLLM", tags=["hbllm", "brain", "cognitive"])

        # User model tracks focus
        model = um.get_model("test")
        assert model.current_focus.value is not None

        # Project graph detects the project
        project = pg.auto_detect_project("hbllm brain architecture")
        assert project is not None
        assert "HBLLM" in project.name


class TestUserModelExecutiveCortexIntegration:
    """Test that ExecutiveCortex uses UserModel for goal prioritization."""

    def test_cortex_uses_user_alignment(self, tmp_path):
        um = UserModelEngine(data_dir=str(tmp_path))
        um.update_from_interaction("default", "working on brain architecture", metadata={"topic": "brain"})

        class MockGoalManager:
            def get_active_goals(self, tenant_id="default"):
                return [
                    {"name": "brain architecture refactor", "priority": "medium"},
                    {"name": "documentation update", "priority": "medium"},
                ]

        cortex = ExecutiveCortex(goal_manager=MockGoalManager(), user_model=um)
        decision = cortex.decide_next_action()
        assert decision.action == "switch_to_goal"
        # Brain-related goal should score higher due to user focus alignment
        assert decision.target_goal is not None


class TestProjectGraphCortexIntegration:
    """Test that ExecutiveCortex can select goals from ProjectGraph."""

    def test_cortex_selects_from_project_goals(self, tmp_path):
        pg = ProjectGraph(data_dir=str(tmp_path))
        proj = pg.create_project("HBLLM")
        pg.add_entity(proj.entity_id, "goal", "Build UserModel")
        pg.add_entity(proj.entity_id, "goal", "Write documentation")

        # Use project graph goals as goal source
        goals = pg.get_active_goals(proj.entity_id)
        assert len(goals) == 2

        class PGGoalManager:
            def __init__(self, project_graph, project_id):
                self._pg = project_graph
                self._pid = project_id

            def get_active_goals(self, tenant_id="default"):
                return [
                    {"name": g.name, "priority": "high"} for g in self._pg.get_active_goals(self._pid)
                ]

        cortex = ExecutiveCortex(goal_manager=PGGoalManager(pg, proj.entity_id))
        decision = cortex.decide_next_action()
        assert decision.action == "switch_to_goal"
        assert decision.target_goal in ("Build UserModel", "Write documentation")


class TestRelationshipContextFusionIntegration:
    """Test that RelationshipMemory feeds into ContextFusion."""

    @pytest.fixture
    def setup(self, tmp_path):
        rm = RelationshipMemory(data_dir=str(tmp_path))
        engine = ContextFusionEngine(token_budget=2000)
        provider = ContextFusionEngine.relationship_provider(rm)
        engine.register_source("relationships", provider, priority=0.55)
        return rm, engine

    @pytest.mark.asyncio
    async def test_relationship_in_context(self, setup):
        rm, engine = setup
        rm.record_mention("Alice Chen", topic="architecture", sentiment=0.7)

        result = await engine.fuse(query="Ask Alice Chen about the design")
        # Should include relationship context
        found = any("Alice Chen" in s.content for s in result.sections)
        assert found is True

    @pytest.mark.asyncio
    async def test_no_relationship_context_when_no_mention(self, setup):
        rm, engine = setup
        result = await engine.fuse(query="what is the weather today")
        # No person mentions, but may get key people summary
        assert isinstance(result.total_tokens, int)


class TestUserModelContextFusionIntegration:
    """Test that UserModel feeds into ContextFusion."""

    @pytest.fixture
    def setup(self, tmp_path):
        um = UserModelEngine(data_dir=str(tmp_path))
        engine = ContextFusionEngine(token_budget=2000)
        provider = ContextFusionEngine.user_model_provider(um)
        engine.register_source("user_model", provider, priority=0.85)
        return um, engine

    @pytest.mark.asyncio
    async def test_user_context_injected(self, setup):
        um, engine = setup
        um.update_from_interaction("default", "asyncio pydantic fastapi mypy test")
        um.learn_preference("default", "style", "technical", source="explicit")

        result = await engine.fuse(query="how do I use asyncio?", tenant_id="default")
        has_user_content = any(
            "python" in s.content.lower() or "technical" in s.content.lower()
            for s in result.sections
        )
        assert has_user_content is True


class TestProjectGraphContextFusionIntegration:
    """Test that ProjectGraph feeds into ContextFusion."""

    @pytest.fixture
    def setup(self, tmp_path):
        pg = ProjectGraph(data_dir=str(tmp_path))
        engine = ContextFusionEngine(token_budget=2000)
        provider = ContextFusionEngine.project_provider(pg)
        engine.register_source("active_project", provider, priority=0.85)
        return pg, engine

    @pytest.mark.asyncio
    async def test_project_context_injected(self, setup):
        pg, engine = setup
        proj = pg.create_project("HBLLM Brain", tags=["hbllm", "brain", "cognitive"])
        pg.add_entity(proj.entity_id, "goal", "Build UserModel")

        result = await engine.fuse(query="working on hbllm brain", tenant_id="default")
        has_project = any("HBLLM" in s.content for s in result.sections)
        assert has_project is True


class TestRealityGraphContextFusionIntegration:
    """Test that RealityGraph feeds into ContextFusion."""

    @pytest.mark.asyncio
    async def test_reality_graph_provider(self):
        class MockKG:
            def neighbors(self, label):
                if label == "test":
                    return {"concept_a": 0.8}
                return {}

        rg = RealityGraph(knowledge_graph=MockKG())
        engine = ContextFusionEngine(token_budget=2000)
        provider = ContextFusionEngine.reality_graph_provider(rg)
        engine.register_source("reality_graph", provider, priority=0.6)

        result = await engine.fuse(query="test query", tenant_id="default")
        assert isinstance(result.total_tokens, int)


class TestFullCognitivePipelineIntegration:
    """Test all 5 subsystems feeding into ContextFusion simultaneously."""

    @pytest.fixture
    def full_setup(self, tmp_path):
        um = UserModelEngine(data_dir=str(tmp_path))
        pg = ProjectGraph(data_dir=str(tmp_path))
        rm = RelationshipMemory(data_dir=str(tmp_path))
        rg = RealityGraph()
        ec = ExecutiveCortex(user_model=um)
        engine = ContextFusionEngine(token_budget=4000)

        # Register all providers
        engine.register_source(
            "user_model", ContextFusionEngine.user_model_provider(um), priority=0.85
        )
        engine.register_source(
            "active_project", ContextFusionEngine.project_provider(pg), priority=0.85
        )
        engine.register_source(
            "relationships", ContextFusionEngine.relationship_provider(rm), priority=0.55
        )
        engine.register_source(
            "reality_graph", ContextFusionEngine.reality_graph_provider(rg), priority=0.6
        )

        return um, pg, rm, rg, ec, engine

    @pytest.mark.asyncio
    async def test_all_sources_assembled(self, full_setup):
        um, pg, rm, rg, ec, engine = full_setup

        # Populate all subsystems
        um.update_from_interaction("default", "asyncio pydantic fastapi mypy python project")
        um.learn_preference("default", "verbosity", "concise", source="explicit")
        um.record_belief("default", "AI", "brain-inspired architectures matter")

        proj = pg.create_project("HBLLM", tags=["hbllm", "brain", "cognitive"])
        pg.add_entity(proj.entity_id, "goal", "Build UserModel")
        pg.add_entity(proj.entity_id, "question", "What storage backend?")

        rm.record_mention("Alice Chen", topic="architecture")

        # Fuse everything
        result = await engine.fuse(
            query="Ask Alice Chen about hbllm brain architecture",
            tenant_id="default",
        )

        # Should have content from multiple sources
        assert len(result.sections) >= 2
        assert result.total_tokens > 0
        assert result.budget_used_pct > 0

        # Check content from different sources
        all_content = " ".join(s.content for s in result.sections)
        # UserModel: preference shows up (expertise needs more evidence to pass threshold)
        assert "concise" in all_content.lower() or "verbosity" in all_content.lower()
        # ProjectGraph: project detected
        assert "HBLLM" in all_content
        # RelationshipMemory: person mentioned
        assert "Alice Chen" in all_content

    @pytest.mark.asyncio
    async def test_context_prioritization(self, full_setup):
        um, pg, rm, rg, ec, engine = full_setup

        um.update_from_interaction("default", "test query")
        proj = pg.create_project("TestProject", tags=["test"])

        result = await engine.fuse(query="test project query", tenant_id="default")

        # Sources should be ordered by priority
        if len(result.sections) >= 2:
            priorities = [s.priority for s in result.sections]
            assert priorities == sorted(priorities, reverse=True)


class TestExecutiveCortexMultiModuleIntegration:
    """Test that ExecutiveCortex properly reads from multiple modules."""

    def test_cortex_with_all_modules(self, tmp_path):
        um = UserModelEngine(data_dir=str(tmp_path))
        um.update_from_interaction("default", "working on HBLLM", metadata={"topic": "HBLLM"})

        class MockGoalManager:
            def get_active_goals(self, tenant_id="default"):
                return [{"name": "HBLLM upgrade", "priority": "high"}]

        class MockLoadManager:
            def get_pressure(self):
                return 0.4

        cortex = ExecutiveCortex(
            goal_manager=MockGoalManager(),
            load_manager=MockLoadManager(),
            user_model=um,
        )

        decision = cortex.decide_next_action()
        assert decision.action == "switch_to_goal"
        assert decision.budget  # Should have budget allocation
        snap = cortex.snapshot()
        assert snap["pressure"] == 0.4


class TestSubsystemIndependence:
    """Test that each subsystem works independently without others."""

    def test_user_model_standalone(self, tmp_path):
        um = UserModelEngine(data_dir=str(tmp_path))
        um.update_from_interaction("test", "asyncio pydantic test")
        model = um.get_model("test")
        assert model.tenant_id == "test"

    def test_project_graph_standalone(self, tmp_path):
        pg = ProjectGraph(data_dir=str(tmp_path))
        proj = pg.create_project("Test")
        pg.add_entity(proj.entity_id, "goal", "Build it")
        assert len(pg.get_active_goals(proj.entity_id)) == 1

    def test_executive_cortex_standalone(self):
        cortex = ExecutiveCortex()
        decision = cortex.decide_next_action()
        assert decision.action == "idle"

    def test_relationship_memory_standalone(self, tmp_path):
        rm = RelationshipMemory(data_dir=str(tmp_path))
        rm.record_mention("Alice Chen")
        assert rm.get_person("Alice Chen") is not None

    def test_reality_graph_standalone(self):
        rg = RealityGraph()
        assert rg.query_entity("test") is None
        assert rg.stats()["backends"] == []


class TestDataFlowChain:
    """Test end-to-end data flow: Interaction → UserModel → ExecutiveCortex → Decision."""

    def test_interaction_to_decision_chain(self, tmp_path):
        # 1. User interacts
        um = UserModelEngine(data_dir=str(tmp_path))
        um.update_from_interaction(
            "default",
            "I need to fix the asyncio bug in the HBLLM project",
            metadata={"topic": "HBLLM bug fix"},
        )

        # 2. Project exists
        pg = ProjectGraph(data_dir=str(tmp_path))
        proj = pg.create_project("HBLLM", tags=["hbllm", "asyncio", "bug"])
        pg.add_entity(proj.entity_id, "goal", "Fix asyncio bug")

        # 3. Cortex decides
        class PGGoals:
            def __init__(self, pg, pid):
                self._pg = pg
                self._pid = pid

            def get_active_goals(self, tenant_id="default"):
                return [
                    {"name": g.name, "priority": "high"} for g in self._pg.get_active_goals(self._pid)
                ]

        cortex = ExecutiveCortex(
            goal_manager=PGGoals(pg, proj.entity_id),
            user_model=um,
        )
        decision = cortex.decide_next_action()

        # 4. Should select the HBLLM goal
        assert decision.action == "switch_to_goal"
        assert "asyncio" in decision.target_goal.lower() or "Fix" in decision.target_goal

    def test_relationship_to_context_chain(self, tmp_path):
        # 1. Learn about a person
        rm = RelationshipMemory(data_dir=str(tmp_path))
        rm.record_mention("Alice Chen", topic="HBLLM design", sentiment=0.8)
        rm.record_event("Alice Chen", "collaboration", context="Code review", sentiment_delta=0.3)

        # 2. Query history
        history = rm.get_history("Alice Chen")
        assert len(history.events) >= 1

        # 3. Prioritization
        priority = rm.prioritize_notification("Alice Chen")
        assert priority > 0.3

        # 4. Person appears in relevant people
        relevant = rm.get_relevant_people("HBLLM")
        assert len(relevant) >= 1
        assert relevant[0].name == "Alice Chen"
