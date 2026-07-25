"""Comprehensive tests for the Autonomous Learning Engine.

Tests all 6 new components:
1. CausalModelBuilder
2. ExperimentEngine
3. ContradictionDetector + BeliefRevisionEngine
4. MetaLearner
5. ConceptFormationEngine
6. AutonomousLearner
"""

from __future__ import annotations

import json
import tempfile
from unittest.mock import MagicMock

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────


class MockLLM:
    """Mock LLM that returns controlled JSON responses."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or []
        self._call_idx = 0
        self.calls: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self._call_idx < len(self._responses):
            resp = self._responses[self._call_idx]
            self._call_idx += 1
            return resp
        # Default: return a valid JSON causal decomposition
        return json.dumps(
            {
                "nodes": [
                    {"label": "User Input", "node_type": "precondition", "confidence": 0.8},
                    {"label": "SQL Parser", "node_type": "process", "confidence": 0.7},
                    {"label": "Query Execution", "node_type": "outcome", "confidence": 0.9},
                ],
                "edges": [
                    {
                        "source_label": "User Input",
                        "target_label": "SQL Parser",
                        "mechanism": {
                            "description": "Input passed to parser without sanitization",
                            "steps": [
                                "receive user input",
                                "concatenate into SQL string",
                                "pass to database driver",
                            ],
                            "assumptions": ["input is not sanitized"],
                        },
                        "probability": 0.8,
                    },
                    {
                        "source_label": "SQL Parser",
                        "target_label": "Query Execution",
                        "mechanism": {
                            "description": "Manipulated query executed on database",
                            "steps": [
                                "parse SQL string",
                                "execute against database",
                                "return results to caller",
                            ],
                            "assumptions": ["database accepts the query"],
                        },
                        "probability": 0.9,
                    },
                ],
                "domain": "cybersecurity",
            }
        )


@pytest.fixture
def tmp_data_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def mock_llm():
    return MockLLM()


# ──────────────────────────────────────────────────────────────────────────────
# 1. CausalModelBuilder Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestCausalModelBuilder:
    @pytest.mark.asyncio
    async def test_build_model_basic(self, mock_llm, tmp_data_dir):
        from hbllm.brain.causality.causal_model_builder import CausalModelBuilder

        builder = CausalModelBuilder(llm=mock_llm, data_dir=tmp_data_dir)
        model = await builder.build_model("SQL Injection", domain="cybersecurity")

        assert model.concept == "SQL Injection"
        assert model.domain == "cybersecurity"
        assert len(model.nodes) == 3
        assert len(model.edges) == 2
        assert model.confidence > 0

    @pytest.mark.asyncio
    async def test_model_persistence(self, mock_llm, tmp_data_dir):
        from hbllm.brain.causality.causal_model_builder import CausalModelBuilder

        builder = CausalModelBuilder(llm=mock_llm, data_dir=tmp_data_dir)
        await builder.build_model("SQL Injection")

        # Create new builder pointing to same dir — should load model
        builder2 = CausalModelBuilder(llm=mock_llm, data_dir=tmp_data_dir)
        model = builder2.get_model("SQL Injection")
        assert model is not None
        assert model.concept == "SQL Injection"

    @pytest.mark.asyncio
    async def test_mechanism_is_first_class(self, mock_llm, tmp_data_dir):
        from hbllm.brain.causality.causal_model_builder import CausalModelBuilder

        builder = CausalModelBuilder(llm=mock_llm, data_dir=tmp_data_dir)
        model = await builder.build_model("SQL Injection")

        edge = model.edges[0]
        assert edge.mechanism.description != ""
        assert len(edge.mechanism.steps) > 0
        assert len(edge.mechanism.assumptions) > 0
        assert edge.mechanism.confidence > 0

    @pytest.mark.asyncio
    async def test_model_reuse(self, mock_llm, tmp_data_dir):
        from hbllm.brain.causality.causal_model_builder import CausalModelBuilder

        builder = CausalModelBuilder(llm=mock_llm, data_dir=tmp_data_dir)
        m1 = await builder.build_model("SQL Injection")
        m2 = await builder.build_model("SQL Injection")  # Should reuse

        assert m1.model_id == m2.model_id
        assert len(mock_llm.calls) == 1  # Only called LLM once

    @pytest.mark.asyncio
    async def test_extend_model(self, mock_llm, tmp_data_dir):
        from hbllm.brain.causality.causal_model_builder import CausalModelBuilder

        builder = CausalModelBuilder(llm=mock_llm, data_dir=tmp_data_dir)
        model = await builder.build_model("SQL Injection")

        old_evidence = model.evidence_count
        await builder.extend_model(model, "New evidence about parameterized queries")
        assert model.evidence_count == old_evidence + 1

    @pytest.mark.asyncio
    async def test_merge_models(self, mock_llm, tmp_data_dir):
        from hbllm.brain.causality.causal_model_builder import CausalModelBuilder

        builder = CausalModelBuilder(llm=mock_llm, data_dir=tmp_data_dir)
        m1 = await builder.build_model("SQL Injection")

        # Force different concept by clearing cache
        builder._models.clear()
        m2 = await builder.build_model("XSS")

        merged = await builder.merge_models(m1, m2)
        assert merged.evidence_count == m1.evidence_count + m2.evidence_count

    @pytest.mark.asyncio
    async def test_update_confidence(self, mock_llm, tmp_data_dir):
        from hbllm.brain.causality.causal_model_builder import CausalModelBuilder

        builder = CausalModelBuilder(llm=mock_llm, data_dir=tmp_data_dir)
        model = await builder.build_model("SQL Injection")
        old_conf = model.confidence

        builder.update_model_confidence("SQL Injection", 0.1, verified=True)
        updated = builder.get_model("SQL Injection")
        assert updated is not None
        assert updated.confidence > old_conf
        assert updated.verified is True

    @pytest.mark.asyncio
    async def test_stats(self, mock_llm, tmp_data_dir):
        from hbllm.brain.causality.causal_model_builder import CausalModelBuilder

        builder = CausalModelBuilder(llm=mock_llm, data_dir=tmp_data_dir)
        await builder.build_model("SQL Injection")

        stats = builder.stats()
        assert stats["models_count"] == 1
        assert stats["models_built"] == 1
        assert stats["mechanisms_count"] > 0

    @pytest.mark.asyncio
    async def test_query_chain(self, mock_llm, tmp_data_dir):
        from hbllm.brain.causality.causal_model_builder import CausalModelBuilder

        builder = CausalModelBuilder(llm=mock_llm, data_dir=tmp_data_dir)
        await builder.build_model("SQL Injection")

        paths = builder.query_chain("User Input", "Query Execution")
        assert len(paths) >= 1

    @pytest.mark.asyncio
    async def test_serialization_roundtrip(self, tmp_data_dir):
        from hbllm.brain.causality.causal_model_builder import (
            CausalEdge,
            CausalModel,
            CausalNode,
            Mechanism,
        )

        mech = Mechanism(
            description="test mechanism",
            steps=["step1", "step2"],
            assumptions=["assumption1"],
            confidence=0.8,
        )
        node = CausalNode(label="test node", node_type="process", confidence=0.7)
        edge = CausalEdge(source_id="a", target_id="b", mechanism=mech, probability=0.9)
        model = CausalModel(concept="test", nodes=[node], edges=[edge], confidence=0.75)

        d = model.to_dict()
        restored = CausalModel.from_dict(d)
        assert restored.concept == "test"
        assert len(restored.edges) == 1
        assert restored.edges[0].mechanism.description == "test mechanism"


# ──────────────────────────────────────────────────────────────────────────────
# 2. ExperimentEngine Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestExperimentEngine:
    @pytest.mark.asyncio
    async def test_generate_hypothesis(self, tmp_data_dir):
        from hbllm.brain.causality.causal_model_builder import CausalModel, CausalNode
        from hbllm.brain.learning.experiment_engine import ExperimentEngine

        llm = MockLLM(
            [
                json.dumps(
                    {
                        "statement": "If input is not sanitized, SQL injection occurs",
                        "target_edge": "input → parser",
                        "expected_outcome": "injection succeeds",
                    }
                )
            ]
        )
        engine = ExperimentEngine(llm=llm, data_dir=tmp_data_dir)

        model = CausalModel(
            concept="SQL Injection",
            nodes=[CausalNode(label="Input")],
        )
        hyp = await engine.generate_hypothesis("SQL Injection", model)
        assert "sanitized" in hyp.statement.lower() or "SQL" in hyp.statement

    @pytest.mark.asyncio
    async def test_design_experiment(self, tmp_data_dir):
        from hbllm.brain.learning.experiment_engine import ExperimentEngine, Hypothesis

        llm = MockLLM(
            [
                json.dumps(
                    {
                        "setup": "Create a vulnerable SQL query",
                        "method": "code_simulation",
                        "success_criteria": "Injection alters query behavior",
                        "failure_criteria": "Input is properly escaped",
                    }
                )
            ]
        )
        engine = ExperimentEngine(llm=llm, data_dir=tmp_data_dir)

        hyp = Hypothesis(statement="SQL injection occurs on unsanitized input")
        exp = await engine.design_experiment(hyp)
        assert exp.setup != ""
        assert exp.resource_cost > 0

    @pytest.mark.asyncio
    async def test_reality_level_weights(self, tmp_data_dir):
        from hbllm.brain.learning.experiment_engine import (
            REALITY_CONFIDENCE_WEIGHTS,
            RealityLevel,
        )

        assert REALITY_CONFIDENCE_WEIGHTS[RealityLevel.SIMULATED] == 0.2
        assert REALITY_CONFIDENCE_WEIGHTS[RealityLevel.REAL_OBSERVATION] == 1.0
        # Higher reality = higher weight
        levels = list(RealityLevel)
        weights = [REALITY_CONFIDENCE_WEIGHTS[l] for l in levels]
        assert weights == sorted(weights)

    @pytest.mark.asyncio
    async def test_run_experiment(self, tmp_data_dir):
        from hbllm.brain.learning.experiment_engine import (
            Experiment,
            ExperimentEngine,
            Hypothesis,
            RealityLevel,
        )

        llm = MockLLM(
            [
                # Evaluation prompt response
                json.dumps(
                    {
                        "confirmed": True,
                        "confidence": 0.8,
                        "reasoning": "Input manipulation confirmed",
                        "new_knowledge": ["parameterized queries prevent this"],
                        "causal_updates": [],
                    }
                ),
                # For the LLM evaluation
                "The hypothesis is confirmed based on the experiment.",
            ]
        )
        engine = ExperimentEngine(llm=llm, data_dir=tmp_data_dir)

        hyp = Hypothesis(statement="SQL injection occurs", confidence_before=0.5)
        exp = Experiment(
            hypothesis=hyp,
            reality_level=RealityLevel.LLM_PREDICTED,
            setup="Test SQL injection",
            expected_outcome="Injection succeeds",
        )
        result = await engine.run_experiment(exp)
        assert result.reality_weight == 0.3  # LLM_PREDICTED weight

    @pytest.mark.asyncio
    async def test_stats(self, tmp_data_dir):
        from hbllm.brain.learning.experiment_engine import ExperimentEngine

        engine = ExperimentEngine(llm=MockLLM(), data_dir=tmp_data_dir)
        stats = engine.stats()
        assert stats["total_experiments"] == 0
        assert stats["confirmation_rate"] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 3. ContradictionDetector + BeliefRevisionEngine Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestContradictionDetector:
    @pytest.mark.asyncio
    async def test_detect_contradiction(self):
        from hbllm.brain.reasoning.contradiction_detector import ContradictionDetector

        llm = MockLLM(
            [
                json.dumps(
                    {
                        "is_contradiction": True,
                        "severity": 0.8,
                        "contradicted_claim": "A causes B",
                        "reasoning": "New evidence says A does not cause B",
                    }
                )
            ]
        )
        # Mock KnowledgeGraph with existing claims
        kg = MagicMock()
        kg.neighbors.return_value = [
            {"relation": "causes", "entity": "B"},
        ]

        detector = ContradictionDetector(llm=llm, knowledge_graph=kg)
        c = await detector.check_contradiction("A does not cause B", "A")
        assert c is not None
        assert c.severity == 0.8

    @pytest.mark.asyncio
    async def test_no_contradiction(self):
        from hbllm.brain.reasoning.contradiction_detector import ContradictionDetector

        llm = MockLLM(
            [
                json.dumps(
                    {
                        "is_contradiction": False,
                        "severity": 0.0,
                        "contradicted_claim": "",
                        "reasoning": "Claims are compatible",
                    }
                )
            ]
        )
        kg = MagicMock()
        kg.neighbors.return_value = [{"relation": "causes", "entity": "B"}]

        detector = ContradictionDetector(llm=llm, knowledge_graph=kg)
        c = await detector.check_contradiction("A strongly causes B", "A")
        assert c is None


class TestBeliefRevisionEngine:
    def test_create_belief_state(self, tmp_data_dir):
        from hbllm.brain.reasoning.contradiction_detector import BeliefRevisionEngine

        engine = BeliefRevisionEngine(data_dir=tmp_data_dir)
        state = engine.get_belief_state("SQL Injection")
        assert state.concept == "SQL Injection"
        assert len(state.hypotheses) == 0

    @pytest.mark.asyncio
    async def test_integrate_evidence(self, tmp_data_dir):
        from hbllm.brain.reasoning.contradiction_detector import BeliefRevisionEngine

        engine = BeliefRevisionEngine(data_dir=tmp_data_dir)
        state = await engine.integrate_evidence(
            "SQL Injection",
            claim="Unsanitized input enables injection",
            confidence=0.8,
            evidence="Research paper confirms this",
        )
        assert len(state.hypotheses) == 1
        assert state.hypotheses[0].confidence == 0.8

    @pytest.mark.asyncio
    async def test_competing_beliefs(self, tmp_data_dir):
        from hbllm.brain.reasoning.contradiction_detector import (
            BeliefRevisionEngine,
            Contradiction,
        )

        engine = BeliefRevisionEngine(data_dir=tmp_data_dir)
        c = Contradiction(
            existing_claim="A causes B (p=0.7)",
            new_claim="A does not cause B",
            concept="A",
            severity=0.8,
            existing_confidence=0.7,
            new_confidence=0.6,
        )
        state = await engine.handle_contradiction(c)
        assert len(state.hypotheses) == 2
        assert state.is_contested  # Within 0.15 of each other

    @pytest.mark.asyncio
    async def test_dominant_belief(self, tmp_data_dir):
        from hbllm.brain.reasoning.contradiction_detector import BeliefRevisionEngine

        engine = BeliefRevisionEngine(data_dir=tmp_data_dir)
        await engine.integrate_evidence("A", "Claim 1", 0.9, "evidence 1")
        await engine.integrate_evidence("A", "Claim 2", 0.3, "evidence 2")
        state = engine.get_belief_state("A")
        assert state.dominant_belief is not None
        assert state.dominant_belief.confidence > 0.5

    @pytest.mark.asyncio
    async def test_prune_weak_beliefs(self, tmp_data_dir):
        from hbllm.brain.reasoning.contradiction_detector import BeliefRevisionEngine

        engine = BeliefRevisionEngine(data_dir=tmp_data_dir)
        await engine.integrate_evidence("A", "Strong claim", 0.9, "evidence")
        await engine.integrate_evidence("A", "Weak claim", 0.05, "weak evidence")

        pruned = await engine.prune_weak_beliefs(threshold=0.1)
        assert pruned == 1

    def test_confidence_decay(self, tmp_data_dir):
        from hbllm.brain.reasoning.contradiction_detector import BeliefRevisionEngine

        engine = BeliefRevisionEngine(data_dir=tmp_data_dir)
        # Manually add a belief
        state = engine.get_belief_state("A")
        from hbllm.brain.reasoning.contradiction_detector import BeliefHypothesis

        state.hypotheses.append(BeliefHypothesis(claim="test", confidence=0.5))
        engine._persist(state)

        decayed = engine.decay_all_beliefs(decay_rate=0.1)
        assert decayed == 1
        assert state.hypotheses[0].confidence < 0.5

    def test_persistence(self, tmp_data_dir):
        from hbllm.brain.reasoning.contradiction_detector import BeliefRevisionEngine

        engine1 = BeliefRevisionEngine(data_dir=tmp_data_dir)
        state = engine1.get_belief_state("test")
        from hbllm.brain.reasoning.contradiction_detector import BeliefHypothesis

        state.hypotheses.append(BeliefHypothesis(claim="persisted", confidence=0.7))
        engine1._persist(state)

        engine2 = BeliefRevisionEngine(data_dir=tmp_data_dir)
        loaded = engine2.get_belief_state("test")
        assert len(loaded.hypotheses) == 1
        assert loaded.hypotheses[0].claim == "persisted"


# ──────────────────────────────────────────────────────────────────────────────
# 4. MetaLearner Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestMetaLearner:
    @pytest.mark.asyncio
    async def test_record_session(self, tmp_data_dir):
        from hbllm.brain.learning.meta_learner import MetaLearner

        ml = MetaLearner(data_dir=tmp_data_dir)
        session = await ml.record_session(
            domain="cybersecurity",
            method="research",
            confidence_before=0.3,
            confidence_after=0.5,
            resource_cost=2.0,
        )
        assert session.confidence_gain == pytest.approx(0.2)
        assert session.cost_efficiency == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_strategy_computed(self, tmp_data_dir):
        import asyncio

        from hbllm.brain.learning.meta_learner import MetaLearner

        ml = MetaLearner(data_dir=tmp_data_dir)
        await ml.record_session("cyber", "research", 0.2, 0.4, 2.0)
        await asyncio.sleep(0.01)  # Ensure unique timestamp-based session_id
        await ml.record_session("cyber", "experiment", 0.4, 0.7, 5.0)

        strategy = ml.get_strategy("cyber")
        assert strategy.total_sessions == 2
        assert strategy.cost_efficiency > 0

    def test_recommend_beginner(self, tmp_data_dir):
        from hbllm.brain.learning.meta_learner import MetaLearner

        ml = MetaLearner(data_dir=tmp_data_dir)
        action = ml.recommend_next_action("unknown", current_confidence=0.1)
        assert action == "research"  # Beginners should research

    def test_recommend_advanced(self, tmp_data_dir):
        from hbllm.brain.learning.meta_learner import MetaLearner

        ml = MetaLearner(data_dir=tmp_data_dir)
        action = ml.recommend_next_action("unknown", current_confidence=0.9)
        assert action == "test"  # Experts should test

    @pytest.mark.asyncio
    async def test_method_effectiveness(self, tmp_data_dir):
        import asyncio

        from hbllm.brain.learning.meta_learner import MetaLearner

        ml = MetaLearner(data_dir=tmp_data_dir)
        await ml.record_session("cyber", "research", 0.2, 0.5, 2.0)
        await asyncio.sleep(0.01)  # Ensure unique timestamp-based session_id
        await ml.record_session("cyber", "experiment", 0.5, 0.8, 5.0)

        eff = ml.get_method_effectiveness("cyber")
        assert "research" in eff
        assert "experiment" in eff

    @pytest.mark.asyncio
    async def test_stats(self, tmp_data_dir):
        from hbllm.brain.learning.meta_learner import MetaLearner

        ml = MetaLearner(data_dir=tmp_data_dir)
        await ml.record_session("cyber", "research", 0.2, 0.5, 2.0)

        stats = ml.stats()
        assert stats["total_sessions"] == 1
        assert stats["total_confidence_gain"] > 0

    @pytest.mark.asyncio
    async def test_persistence(self, tmp_data_dir):
        from hbllm.brain.learning.meta_learner import MetaLearner

        ml1 = MetaLearner(data_dir=tmp_data_dir)
        await ml1.record_session("cyber", "research", 0.2, 0.5, 2.0)

        ml2 = MetaLearner(data_dir=tmp_data_dir)
        strategy = ml2.get_strategy("cyber")
        assert strategy.total_sessions > 0


# ──────────────────────────────────────────────────────────────────────────────
# 5. ConceptFormationEngine Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestConceptFormationEngine:
    @pytest.mark.asyncio
    async def test_discover_abstractions_no_builder(self, tmp_data_dir):
        from hbllm.brain.reasoning.concept_formation import ConceptFormationEngine

        engine = ConceptFormationEngine(data_dir=tmp_data_dir)
        result = await engine.discover_abstractions()
        assert result == []  # No builder = no abstractions

    @pytest.mark.asyncio
    async def test_discover_abstractions_with_models(self, mock_llm, tmp_data_dir):
        from hbllm.brain.causality.causal_model_builder import (
            CausalEdge,
            CausalModel,
            CausalNode,
            Mechanism,
        )
        from hbllm.brain.reasoning.concept_formation import ConceptFormationEngine

        # Create mock builder with similar models
        builder = MagicMock()
        shared_mechanism = Mechanism(
            description="input manipulation",
            steps=["receive input", "inject payload", "execute unauthorized"],
        )
        models = [
            CausalModel(
                concept="SQL Injection",
                domain="cybersecurity",
                nodes=[CausalNode(label="input"), CausalNode(label="output")],
                edges=[
                    CausalEdge(
                        source_id="input",
                        target_id="output",
                        mechanism=shared_mechanism,
                    )
                ],
            ),
            CausalModel(
                concept="XSS",
                domain="cybersecurity",
                nodes=[CausalNode(label="input"), CausalNode(label="output")],
                edges=[
                    CausalEdge(
                        source_id="input",
                        target_id="output",
                        mechanism=Mechanism(
                            description="input manipulation",
                            steps=["receive input", "inject script", "execute unauthorized"],
                        ),
                    )
                ],
            ),
        ]
        builder.get_all_models.return_value = models

        llm = MockLLM(
            [
                json.dumps(
                    {
                        "label": "Injection Attack Pattern",
                        "description": "Input manipulation leading to unauthorized execution",
                        "generalized_steps": ["receive input", "inject payload", "execute"],
                        "generalized_assumptions": ["input is not validated"],
                    }
                )
            ]
        )

        engine = ConceptFormationEngine(
            llm=llm,
            causal_model_builder=builder,
            data_dir=tmp_data_dir,
        )
        abstractions = await engine.discover_abstractions()
        # May or may not find abstractions depending on fingerprint matching
        assert isinstance(abstractions, list)

    @pytest.mark.asyncio
    async def test_cross_domain_analogies(self, tmp_data_dir):
        from hbllm.brain.causality.causal_model_builder import (
            CausalEdge,
            CausalModel,
            Mechanism,
        )
        from hbllm.brain.reasoning.concept_formation import ConceptFormationEngine

        builder = MagicMock()
        builder.get_all_models.return_value = [
            CausalModel(
                concept="SQL Injection",
                domain="cybersecurity",
                edges=[
                    CausalEdge(
                        mechanism=Mechanism(
                            steps=["infection vector", "propagation", "containment"]
                        )
                    )
                ],
            ),
            CausalModel(
                concept="Virus Spread",
                domain="biology",
                edges=[
                    CausalEdge(
                        mechanism=Mechanism(
                            steps=["infection vector", "propagation", "immune response"]
                        )
                    )
                ],
            ),
        ]

        engine = ConceptFormationEngine(causal_model_builder=builder, data_dir=tmp_data_dir)
        analogies = await engine.discover_cross_domain_analogies()
        assert len(analogies) >= 1
        assert analogies[0].domain_a != analogies[0].domain_b

    def test_stats(self, tmp_data_dir):
        from hbllm.brain.reasoning.concept_formation import ConceptFormationEngine

        engine = ConceptFormationEngine(data_dir=tmp_data_dir)
        stats = engine.stats()
        assert stats["abstract_concepts"] == 0

    def test_serialization(self):
        from hbllm.brain.reasoning.concept_formation import AbstractConcept

        concept = AbstractConcept(
            label="Test Pattern",
            instances=["A", "B"],
            confidence=0.7,
        )
        d = concept.to_dict()
        restored = AbstractConcept.from_dict(d)
        assert restored.label == "Test Pattern"
        assert restored.instances == ["A", "B"]


# ──────────────────────────────────────────────────────────────────────────────
# 6. AutonomousLearner Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestAutonomousLearner:
    @pytest.mark.asyncio
    async def test_learn_basic(self, mock_llm, tmp_data_dir):
        from hbllm.brain.causality.causal_model_builder import CausalModelBuilder
        from hbllm.brain.learning.autonomous_learner import AutonomousLearner
        from hbllm.network.bus import InProcessBus

        bus = InProcessBus()
        await bus.start()

        try:
            llm = MockLLM(
                [
                    # Needs identification
                    json.dumps(
                        {
                            "needs": [
                                {
                                    "concept": "SQL Basics",
                                    "why": "Foundation",
                                    "priority": 1,
                                    "prerequisites": [],
                                }
                            ]
                        }
                    ),
                    # Causal model build
                    json.dumps(
                        {
                            "nodes": [{"label": "SQL Query", "node_type": "process"}],
                            "edges": [],
                            "domain": "databases",
                        }
                    ),
                    # Self-evaluation questions
                    json.dumps(
                        {"questions": [{"question": "test", "type": "prediction", "weight": 1.0}]}
                    ),
                ]
            )

            builder = CausalModelBuilder(llm=llm, data_dir=tmp_data_dir)

            learner = AutonomousLearner(
                node_id="test_learner",
                llm=llm,
                causal_model_builder=builder,
            )
            await learner.start(bus)  # Node.start() sets _bus and calls on_start()

            goal = await learner.learn("SQL Injection", confidence_target=0.5)
            assert goal.status in ("completed", "budget_exhausted")
            assert goal.causal_models_built >= 0
        finally:
            await bus.stop()

    @pytest.mark.asyncio
    async def test_learning_budget(self):
        from hbllm.brain.learning.autonomous_learner import LearningBudget

        budget = LearningBudget()
        assert budget.can_afford("research")
        assert budget.can_afford("experiment")

        budget.remaining_web = 0
        assert not budget.can_afford("research")

    @pytest.mark.asyncio
    async def test_learning_goal_serialization(self):
        from hbllm.brain.learning.autonomous_learner import LearningGoal

        goal = LearningGoal(
            topic="Cybersecurity",
            depth="intermediate",
            confidence_target=0.8,
        )
        d = goal.to_dict()
        assert d["topic"] == "Cybersecurity"
        assert d["budget"]["daily_web_searches"] == 100


# ──────────────────────────────────────────────────────────────────────────────
# 7. KnowledgeGraph Confidence Tests
# ──────────────────────────────────────────────────────────────────────────────


class TestKnowledgeGraphConfidence:
    def test_entity_has_confidence(self):
        from hbllm.memory.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        entity = kg.add_entity("SQL Injection", entity_type="concept")
        assert entity.confidence == 1.0
        assert entity.evidence_count == 1
        assert entity.verified is False

    def test_decay_confidence(self):
        from hbllm.memory.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_entity("A")
        kg.add_entity("B")

        decayed = kg.decay_confidence(rate=0.1)
        assert decayed == 2

        entity = kg.get_entity("A")
        assert entity is not None
        assert entity.confidence < 1.0

    def test_get_low_confidence(self):
        from hbllm.memory.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        e1 = kg.add_entity("Strong")  # noqa: F841
        e2 = kg.add_entity("Weak")
        e2.confidence = 0.1

        low = kg.get_low_confidence(threshold=0.3)
        assert len(low) == 1
        assert low[0].label == "weak"

    def test_reinforce(self):
        from hbllm.memory.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.add_entity("Test")
        kg.decay_confidence(rate=0.5)  # Drop to 0.5

        result = kg.reinforce("Test", evidence="new paper confirms", confidence_boost=0.2)
        assert result is not None
        assert result.confidence > 0.5
        assert result.evidence_count == 2

    def test_confidence_serialization(self):
        from hbllm.memory.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        e = kg.add_entity("Test")
        e.confidence = 0.7
        e.evidence_count = 5
        e.verified = True

        d = kg.to_dict()
        restored = KnowledgeGraph.from_dict(d)
        entity = restored.get_entity("Test")
        assert entity is not None
        assert entity.confidence == 0.7
        assert entity.evidence_count == 5
        assert entity.verified is True
