"""Tests for DiscoveryWorkspace — research program lifecycle."""

from __future__ import annotations

from hbllm.brain.epistemics.workspace import DiscoveryWorkspace
from hbllm.hcir.graph import (
    CognitiveGraph,
    EvidenceNode,
    ExperimentNode,
    HypothesisNode,
)
from hbllm.hcir.types import EvidenceStrength


class TestProgramLifecycle:
    """Test research program CRUD."""

    def test_create_program(
        self,
        workspace: DiscoveryWorkspace,
    ) -> None:
        prog = workspace.create_program("Test Program", "Why does X happen?")
        assert prog.program_id != ""
        assert prog.title == "Test Program"
        assert prog.status == "active"

    def test_get_program(self, workspace: DiscoveryWorkspace) -> None:
        prog = workspace.create_program("Get Test", "Question")
        retrieved = workspace.get_program(prog.program_id)
        assert retrieved is not None
        assert retrieved.title == "Get Test"

    def test_list_programs(self, workspace: DiscoveryWorkspace) -> None:
        workspace.create_program("Prog 1", "Q1")
        workspace.create_program("Prog 2", "Q2")
        programs = workspace.list_programs()
        assert len(programs) >= 2

    def test_list_programs_by_status(self, workspace: DiscoveryWorkspace) -> None:
        p1 = workspace.create_program("Active", "Q")
        workspace.create_program("Also Active", "Q")
        workspace.update_program_status(p1.program_id, "completed")

        active = workspace.list_programs(status="active")
        completed = workspace.list_programs(status="completed")
        assert len(completed) == 1
        assert len(active) >= 1

    def test_update_status(self, workspace: DiscoveryWorkspace) -> None:
        prog = workspace.create_program("Status Test", "Q")
        workspace.update_program_status(prog.program_id, "paused")
        updated = workspace.get_program(prog.program_id)
        assert updated is not None
        assert updated.status == "paused"


class TestObjectivesAndQuestions:
    """Test objectives and questions within a program."""

    def test_add_objective(self, workspace: DiscoveryWorkspace) -> None:
        prog = workspace.create_program("Obj Test", "Q")
        obj_id = workspace.add_objective(prog.program_id, "Find mechanism")
        assert obj_id != ""

    def test_add_question(self, workspace: DiscoveryWorkspace) -> None:
        prog = workspace.create_program("Q Test", "Q")
        obj = workspace.add_objective(prog.program_id, "Find")
        q_id = workspace.add_question(
            prog.program_id,
            obj,
            "Why X?",
            importance=0.8,
        )
        assert q_id != ""

    def test_multiple_questions(self, workspace: DiscoveryWorkspace) -> None:
        prog = workspace.create_program("Multi Q", "Q")
        obj = workspace.add_objective(prog.program_id, "Find")
        q1 = workspace.add_question(prog.program_id, obj, "Why X?", importance=0.9)
        q2 = workspace.add_question(prog.program_id, obj, "How Y?", importance=0.5)
        assert q1 != q2


class TestHypothesisLifecycle:
    """Test hypothesis tracking within workspace."""

    def test_add_hypothesis(
        self,
        workspace: DiscoveryWorkspace,
        graph: CognitiveGraph,
    ) -> None:
        prog = workspace.create_program("Hyp Test", "Q")
        hyp = HypothesisNode(claim="Z causes X")
        hyp_id = workspace.add_hypothesis(prog.program_id, hyp)
        assert hyp_id != ""

    def test_update_hypothesis_lifecycle(
        self,
        workspace: DiscoveryWorkspace,
        graph: CognitiveGraph,
    ) -> None:
        from hbllm.hcir.graph import HypothesisLifecycle

        prog = workspace.create_program("Lifecycle", "Q")
        hyp = HypothesisNode(claim="Z causes X")
        hyp_id = workspace.add_hypothesis(prog.program_id, hyp)
        workspace.update_hypothesis_lifecycle(
            prog.program_id,
            hyp_id,
            HypothesisLifecycle.SUPPORTED,
        )
        updated = workspace.get_program(prog.program_id)
        assert updated is not None


class TestUnknownTracking:
    """Test unknown/question resolution tracking."""

    def test_add_and_resolve_unknown(
        self,
        workspace: DiscoveryWorkspace,
    ) -> None:
        prog = workspace.create_program("Unknown Test", "Q")
        u_id = workspace.add_unknown(
            prog.program_id,
            "Why does Z modulate X?",
            importance=0.7,
        )
        assert u_id != ""

        workspace.resolve_unknown(
            prog.program_id,
            u_id,
            "Experiment E3 showed Z activates pathway P",
        )
        # Should not raise


class TestEvidenceAndExperiments:
    """Test evidence and experiment tracking."""

    def test_add_evidence(
        self,
        workspace: DiscoveryWorkspace,
        graph: CognitiveGraph,
    ) -> None:
        prog = workspace.create_program("Evidence Test", "Q")
        ev = EvidenceNode(
            evidence_type=EvidenceStrength.EXPERIMENTAL,
            methodology="RCT n=200",
            sample_size=200,
        )
        ev_id = workspace.add_evidence(prog.program_id, ev)
        assert ev_id != ""

    def test_add_experiment(
        self,
        workspace: DiscoveryWorkspace,
        graph: CognitiveGraph,
    ) -> None:
        prog = workspace.create_program("Experiment Test", "Q")
        exp = ExperimentNode(
            design="Compare Z+ vs Z- groups",
            hypothesis_ids=["hyp1"],
        )
        exp_id = workspace.add_experiment(prog.program_id, exp)
        assert exp_id != ""
