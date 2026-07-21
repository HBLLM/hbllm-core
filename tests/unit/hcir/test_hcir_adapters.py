"""Unit tests for HCIR Integration Adapters."""

# Import existing cognitive state types
from hbllm.brain.core.cognitive_state import (
    CognitiveStateDelta,
    CognitiveStateSnapshot,
)
from hbllm.hcir.adapters.cognitive_state_adapter import CognitiveStateAdapter
from hbllm.hcir.adapters.memory_adapter import MemoryAdapter
from hbllm.hcir.graph import (
    BeliefNode,
    ConceptNode,
    EpisodeNode,
    HCIREdgeType,
    HCIRNodeType,
    ObservationNode,
    SkillNode,
    ValueNode,
)
from hbllm.hcir.workspace import HCIRWorkspaceState

# ═══════════════════════════════════════════════════════════════════════════
# CognitiveStateAdapter Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCognitiveStateAdapter:
    def _make_snapshot(self, **overrides) -> CognitiveStateSnapshot:
        defaults = {
            "confidence": 0.8,
            "uncertainty": 0.3,
            "relevance": 0.9,
            "novelty": 0.5,
            "motivation": 0.7,
            "valence": 0.6,
            "arousal": 0.4,
            "intention_strength": 0.85,
            "fatigue": 0.1,
            "curiosity": 0.6,
            "stress": 0.2,
            "focus_target": "planning",
            "version": 1,
        }
        defaults.update(overrides)
        return CognitiveStateSnapshot(**defaults)

    def test_snapshot_to_nodes(self):
        adapter = CognitiveStateAdapter()
        snapshot = self._make_snapshot()
        nodes = adapter.snapshot_to_nodes(snapshot)
        # 11 float fields + 1 focus_target
        assert len(nodes) == 12
        assert all(isinstance(n, ObservationNode) for n in nodes)

    def test_sync_to_workspace(self):
        adapter = CognitiveStateAdapter()
        ws = HCIRWorkspaceState()
        snapshot = self._make_snapshot()
        adapter.sync_to_workspace(snapshot, ws)
        # Verify nodes exist in workspace
        confidence_node = ws.get_node("cs_confidence")
        assert confidence_node is not None
        assert isinstance(confidence_node, ObservationNode)
        assert confidence_node.payload["value"] == 0.8

    def test_sync_idempotent(self):
        adapter = CognitiveStateAdapter()
        ws = HCIRWorkspaceState()
        snapshot = self._make_snapshot(confidence=0.5)
        adapter.sync_to_workspace(snapshot, ws)
        # Update confidence
        snapshot2 = self._make_snapshot(confidence=0.9)
        adapter.sync_to_workspace(snapshot2, ws)
        # Should have updated, not duplicated
        node = ws.get_node("cs_confidence")
        assert isinstance(node, ObservationNode)
        assert node.payload["value"] == 0.9
        # Total observation nodes should not exceed 12
        obs = ws.graph.nodes_by_type(HCIRNodeType.OBSERVATION)
        assert len(obs) == 12

    def test_read_from_workspace(self):
        adapter = CognitiveStateAdapter()
        ws = HCIRWorkspaceState()
        snapshot = self._make_snapshot()
        adapter.sync_to_workspace(snapshot, ws)
        result = adapter.read_from_workspace(ws)
        assert result["confidence"] == 0.8
        assert result["focus_target"] == "planning"
        assert result["stress"] == 0.2

    def test_delta_to_transaction(self):
        adapter = CognitiveStateAdapter()
        delta = CognitiveStateDelta(
            source_node="critic_node",
            changes={"confidence": 0.95, "stress": 0.5},
        )
        tx = adapter.delta_to_transaction(delta)
        assert tx.author == "critic_node"
        assert tx.operation_count == 2

    def test_tags_and_provenance(self):
        adapter = CognitiveStateAdapter()
        snapshot = self._make_snapshot()
        nodes = adapter.snapshot_to_nodes(snapshot, tenant_id="acme")
        for node in nodes:
            assert "cognitive_state" in node.tags
            assert node.scope.tenant_id == "acme"
            assert node.sensor_source == "cognitive_state"


# ═══════════════════════════════════════════════════════════════════════════
# MemoryAdapter Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMemoryAdapterEpisodes:
    def test_import_episode(self):
        adapter = MemoryAdapter()
        ws = HCIRWorkspaceState()
        node = adapter.import_episode(
            workspace=ws,
            session_id="sess_42",
            summary="User asked about weather",
            outcome="Provided forecast",
            reward=0.9,
        )
        assert isinstance(node, EpisodeNode)
        assert node.id == "ep_sess_42"
        assert ws.get_node("ep_sess_42") is not None
        assert node.reward == 0.9

    def test_episode_appears_in_memory_view(self):
        adapter = MemoryAdapter()
        ws = HCIRWorkspaceState()
        adapter.import_episode(ws, "s1", "Test episode")
        memory_view = ws.graph.memory_view()
        ids = {n.id for n in memory_view}
        assert "ep_s1" in ids


class TestMemoryAdapterConcepts:
    def test_import_concept(self):
        adapter = MemoryAdapter()
        ws = HCIRWorkspaceState()
        node = adapter.import_concept(
            workspace=ws,
            label="Machine Learning",
            domain="AI",
            confidence=0.9,
        )
        assert isinstance(node, ConceptNode)
        assert node.domain == "AI"
        assert ws.get_node("concept_machine_learning") is not None

    def test_concept_appears_in_knowledge_view(self):
        adapter = MemoryAdapter()
        ws = HCIRWorkspaceState()
        adapter.import_concept(ws, "Neural Networks", domain="AI")
        knowledge_view = ws.graph.knowledge_view()
        ids = {n.id for n in knowledge_view}
        assert "concept_neural_networks" in ids


class TestMemoryAdapterSkills:
    def test_import_skill(self):
        adapter = MemoryAdapter()
        ws = HCIRWorkspaceState()
        node = adapter.import_skill(
            workspace=ws,
            skill_name="code_review",
            success_rate=0.85,
            invocation_count=42,
        )
        assert isinstance(node, SkillNode)
        assert node.success_rate == 0.85
        assert ws.get_node("skill_code_review") is not None


class TestMemoryAdapterValues:
    def test_import_value(self):
        adapter = MemoryAdapter()
        ws = HCIRWorkspaceState()
        node = adapter.import_value(ws, dimension="helpfulness", weight=0.9)
        assert isinstance(node, ValueNode)
        assert node.weight == 0.9
        assert ws.get_node("value_helpfulness") is not None


class TestMemoryAdapterTriples:
    def test_import_triple(self):
        adapter = MemoryAdapter()
        ws = HCIRWorkspaceState()
        subj, obj, edge = adapter.import_knowledge_triple(
            workspace=ws,
            subject="Python",
            relation="is_a",
            obj="Programming Language",
        )
        assert isinstance(subj, BeliefNode)
        assert isinstance(obj, BeliefNode)
        assert edge.edge_type == HCIREdgeType.PART_OF
        assert ws.get_node("belief_python") is not None
        assert ws.get_node("belief_programming_language") is not None

    def test_triple_idempotent(self):
        adapter = MemoryAdapter()
        ws = HCIRWorkspaceState()
        adapter.import_knowledge_triple(ws, "A", "supports", "B")
        adapter.import_knowledge_triple(ws, "A", "supports", "B")
        # Should not duplicate nodes
        assert ws.graph.node_count == 2

    def test_triple_with_different_relations(self):
        adapter = MemoryAdapter()
        ws = HCIRWorkspaceState()
        _, _, e1 = adapter.import_knowledge_triple(ws, "X", "causes", "Y")
        assert e1.edge_type == HCIREdgeType.CAUSES
        _, _, e2 = adapter.import_knowledge_triple(ws, "A", "contradicts", "B")
        assert e2.edge_type == HCIREdgeType.CONTRADICTS
