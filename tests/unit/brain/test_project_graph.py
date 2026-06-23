"""Unit tests for ProjectGraph — graph-based project cognition."""

import time
import pytest

from hbllm.brain.project_graph import (
    ProjectEntity,
    ProjectGraph,
    ProjectRelation,
    ProjectContext,
)


class TestProjectEntity:
    def test_defaults(self):
        entity = ProjectEntity(
            entity_id="proj_1", entity_type="project", name="HBLLM"
        )
        assert entity.status == "active"
        assert entity.tenant_id == "default"
        assert entity.entity_type == "project"

    def test_to_dict(self):
        entity = ProjectEntity(
            entity_id="proj_1", entity_type="project", name="Test"
        )
        d = entity.to_dict()
        assert d["entity_id"] == "proj_1"
        assert d["name"] == "Test"
        assert d["entity_type"] == "project"


class TestProjectRelation:
    def test_defaults(self):
        rel = ProjectRelation(
            source_id="proj_1", target_id="goal_1", relation_type="has_goal"
        )
        assert rel.relation_type == "has_goal"

    def test_to_dict(self):
        rel = ProjectRelation(
            source_id="a", target_id="b", relation_type="blocked_by"
        )
        d = rel.to_dict()
        assert d["relation_type"] == "blocked_by"


class TestProjectGraph:
    @pytest.fixture
    def graph(self, tmp_path):
        return ProjectGraph(data_dir=str(tmp_path))

    def test_create_project(self, graph):
        proj = graph.create_project("HBLLM", description="Brain-inspired AI")
        assert proj.name == "HBLLM"
        assert proj.entity_type == "project"
        assert proj.status == "active"
        assert proj.entity_id.startswith("proj_")

    def test_create_project_with_tags(self, graph):
        proj = graph.create_project("HBLLM", tags=["ai", "cognitive"])
        ctx = graph._load_context(proj.entity_id)
        assert "ai" in ctx.tags
        assert "cognitive" in ctx.tags

    def test_add_entity_goal(self, graph):
        proj = graph.create_project("Test Project")
        goal = graph.add_entity(proj.entity_id, "goal", "Build core")
        assert goal.entity_type == "goal"
        assert goal.name == "Build core"

    def test_add_entity_creates_relation(self, graph):
        proj = graph.create_project("Test")
        goal = graph.add_entity(proj.entity_id, "goal", "Goal A")
        # Check the relation was auto-created
        children = graph.get_children(proj.entity_id, relation_type="has_goal")
        assert len(children) == 1
        assert children[0].name == "Goal A"

    def test_add_multiple_entity_types(self, graph):
        proj = graph.create_project("Test")
        graph.add_entity(proj.entity_id, "goal", "Build it")
        graph.add_entity(proj.entity_id, "question", "What design?")
        graph.add_entity(proj.entity_id, "blocker", "Dependency issue")
        graph.add_entity(proj.entity_id, "decision", "Use graph structure")

        goals = graph.get_children(proj.entity_id, relation_type="has_goal")
        questions = graph.get_children(proj.entity_id, relation_type="has_question")
        blockers = graph.get_children(proj.entity_id, relation_type="has_blocker")
        decisions = graph.get_children(proj.entity_id, relation_type="has_decision")

        assert len(goals) == 1
        assert len(questions) == 1
        assert len(blockers) == 1
        assert len(decisions) == 1

    def test_get_entity(self, graph):
        proj = graph.create_project("Test")
        loaded = graph.get_entity(proj.entity_id)
        assert loaded is not None
        assert loaded.name == "Test"

    def test_get_entity_missing(self, graph):
        assert graph.get_entity("nonexistent") is None

    def test_update_status(self, graph):
        proj = graph.create_project("Test")
        goal = graph.add_entity(proj.entity_id, "goal", "Build it")
        updated = graph.update_status(goal.entity_id, "completed")
        assert updated is True
        loaded = graph.get_entity(goal.entity_id)
        assert loaded.status == "completed"

    def test_resolve(self, graph):
        proj = graph.create_project("Test")
        question = graph.add_entity(proj.entity_id, "question", "What design?")
        result = graph.resolve(question.entity_id, resolution="Use graphs")
        assert result is True
        loaded = graph.get_entity(question.entity_id)
        assert loaded.status == "resolved"
        assert loaded.metadata.get("resolution") == "Use graphs"

    def test_resolve_nonexistent(self, graph):
        assert graph.resolve("fake_id") is False

    def test_add_relation(self, graph):
        proj = graph.create_project("Test")
        goal = graph.add_entity(proj.entity_id, "goal", "Build core")
        milestone = graph.add_entity(proj.entity_id, "milestone", "v1 done")
        graph.add_relation(goal.entity_id, milestone.entity_id, "has_milestone")
        children = graph.get_children(goal.entity_id, relation_type="has_milestone")
        assert len(children) == 1
        assert children[0].name == "v1 done"

    def test_cross_project_dependency(self, graph):
        proj_a = graph.create_project("ProjectA")
        proj_b = graph.create_project("ProjectB")
        graph.add_relation(proj_a.entity_id, proj_b.entity_id, "depends_on")
        deps = graph.get_dependencies(proj_a.entity_id)
        assert len(deps) == 1
        assert deps[0].name == "ProjectB"

    def test_get_blockers(self, graph):
        proj = graph.create_project("Test")
        blocker = graph.add_entity(proj.entity_id, "blocker", "Auth system down")
        blockers = graph.get_blockers(proj.entity_id)
        assert len(blockers) == 1
        assert blockers[0].name == "Auth system down"

    def test_get_blockers_excludes_resolved(self, graph):
        proj = graph.create_project("Test")
        b = graph.add_entity(proj.entity_id, "blocker", "Fixed issue")
        graph.resolve(b.entity_id)
        blockers = graph.get_blockers(proj.entity_id)
        assert len(blockers) == 0

    def test_get_open_questions(self, graph):
        proj = graph.create_project("Test")
        graph.add_entity(proj.entity_id, "question", "Open Q")
        q2 = graph.add_entity(proj.entity_id, "question", "Resolved Q")
        graph.resolve(q2.entity_id)

        open_qs = graph.get_open_questions(proj.entity_id)
        assert len(open_qs) == 1
        assert open_qs[0].name == "Open Q"

    def test_get_active_goals(self, graph):
        proj = graph.create_project("Test")
        graph.add_entity(proj.entity_id, "goal", "Active goal")
        g2 = graph.add_entity(proj.entity_id, "goal", "Completed goal")
        graph.update_status(g2.entity_id, "completed")

        active = graph.get_active_goals(proj.entity_id)
        assert len(active) == 1
        assert active[0].name == "Active goal"

    def test_get_milestones(self, graph):
        proj = graph.create_project("Test")
        graph.add_entity(proj.entity_id, "milestone", "M1")
        goal = graph.add_entity(proj.entity_id, "goal", "Goal A")
        m2 = ProjectEntity(
            entity_id=f"mile_{int(time.time())}",
            entity_type="milestone",
            name="M2",
        )
        graph._save_entity(m2)
        graph.add_relation(goal.entity_id, m2.entity_id, "has_milestone")

        milestones = graph.get_milestones(proj.entity_id)
        assert len(milestones) == 2

    def test_get_active_projects(self, graph):
        graph.create_project("Active1")
        p2 = graph.create_project("Archived")
        graph.update_status(p2.entity_id, "archived")

        active = graph.get_active_projects()
        assert len(active) == 1
        assert active[0].name == "Active1"

    def test_auto_detect_project(self, graph):
        graph.create_project("HBLLM Brain Architecture", tags=["hbllm", "brain", "ai"])
        detected = graph.auto_detect_project("working on hbllm brain module")
        assert detected is not None
        assert "HBLLM" in detected.name

    def test_auto_detect_no_match(self, graph):
        graph.create_project("HBLLM", tags=["ai", "cognitive"])
        detected = graph.auto_detect_project("cooking dinner tonight")
        assert detected is None

    def test_associate_conversation(self, graph):
        proj = graph.create_project("Test")
        graph.associate_conversation(proj.entity_id, topic="architecture", files=["main.py"])
        ctx = graph._load_context(proj.entity_id)
        assert "architecture" in ctx.last_topics
        assert "main.py" in ctx.last_files_touched
        assert ctx.conversation_count == 1

    def test_record_decision(self, graph):
        proj = graph.create_project("Test")
        dec = graph.record_decision(proj.entity_id, "Use KG as canonical store")
        assert dec.entity_type == "decision"
        ctx = graph._load_context(proj.entity_id)
        assert "Use KG as canonical store" in ctx.last_decisions

    def test_reactivate(self, graph):
        proj = graph.create_project("HBLLM")
        graph.add_entity(proj.entity_id, "goal", "Build UserModel")
        graph.add_entity(proj.entity_id, "question", "What storage?")
        graph.add_entity(proj.entity_id, "blocker", "Missing dependency")
        graph.record_decision(proj.entity_id, "Use SQLite")

        context = graph.reactivate(proj.entity_id)
        assert "HBLLM" in context
        assert "Build UserModel" in context
        assert "What storage?" in context
        assert "Missing dependency" in context
        assert "Use SQLite" in context

    def test_reactivate_nonexistent(self, graph):
        assert graph.reactivate("nonexistent") == ""

    @pytest.mark.asyncio
    async def test_get_context_no_match(self, graph):
        ctx = await graph.get_context("random query", "default", 100)
        assert ctx == ""

    @pytest.mark.asyncio
    async def test_get_context_with_match(self, graph):
        graph.create_project("HBLLM Brain", tags=["hbllm", "brain"])
        ctx = await graph.get_context("working on hbllm brain", "default", 500)
        assert "HBLLM" in ctx

    def test_stats(self, graph):
        graph.create_project("Test")
        s = graph.stats()
        assert s["total_entities"] >= 1
        assert s["active_projects"] >= 1

    def test_time_ago(self):
        assert ProjectGraph._time_ago(time.time() - 30) == "just now"
        assert "m ago" in ProjectGraph._time_ago(time.time() - 120)
        assert "h ago" in ProjectGraph._time_ago(time.time() - 7200)
        assert "d ago" in ProjectGraph._time_ago(time.time() - 172800)
