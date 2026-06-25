"""Integration tests: causal learning within the main cognitive loop.

Tests the full flow:
    Query → Causal Context → Skill Selection (belief-weighted) →
    Execute → Success/Failure → Learn → Sleep Consolidation
"""

from __future__ import annotations

import pytest

from hbllm.brain.failure_analyzer import FailureAnalyzer, FailureCategory
from hbllm.brain.mechanism_store import MechanismStore
from hbllm.brain.skill_intelligence_node import SkillIntelligenceNode
from hbllm.brain.skill_registry import Skill, SkillRegistry


@pytest.fixture
def mechanism_store(tmp_path):
    return MechanismStore(data_dir=str(tmp_path / "mechs"))


@pytest.fixture
def skill_registry(tmp_path):
    return SkillRegistry(data_dir=str(tmp_path / "skills"))


class TestBeliefWeightedSkillSelection:
    """SIL should demote skills whose underlying mechanisms have weak beliefs."""

    def test_adjusted_confidence_without_mechanisms(self, skill_registry, mechanism_store):
        sil = SkillIntelligenceNode("sil", skill_registry, mechanism_store)
        skill = Skill(
            skill_id="s1",
            name="Test Skill",
            description="test",
            category="test",
            steps=["step1"],
            tools_used=[],
            success_criteria="ok",
            confidence_score=0.95,
            mechanism_ids=[],
        )
        # No mechanisms → raw confidence
        assert sil._adjusted_confidence(skill) == 0.95

    def test_adjusted_confidence_with_strong_mechanism(self, skill_registry, mechanism_store):
        sil = SkillIntelligenceNode("sil", skill_registry, mechanism_store)
        mech = mechanism_store.create(
            description="Strong mechanism",
            preconditions=["a"],
            process_steps=["b"],
            expected_outcomes=["c"],
            confidence=0.95,
        )
        skill = Skill(
            skill_id="s2",
            name="Strong Skill",
            description="test",
            category="test",
            steps=["step1"],
            tools_used=[],
            success_criteria="ok",
            confidence_score=0.95,
            mechanism_ids=[mech.id],
        )
        # 0.95 × 0.95 = 0.9025
        adj = sil._adjusted_confidence(skill)
        assert adj == pytest.approx(0.9025)

    def test_adjusted_confidence_with_weak_mechanism(self, skill_registry, mechanism_store):
        sil = SkillIntelligenceNode("sil", skill_registry, mechanism_store)
        mech = mechanism_store.create(
            description="Weak mechanism",
            preconditions=["a"],
            process_steps=["b"],
            expected_outcomes=["c"],
            confidence=0.30,
        )
        skill = Skill(
            skill_id="s3",
            name="Weak Foundation Skill",
            description="test",
            category="test",
            steps=["step1"],
            tools_used=[],
            success_criteria="ok",
            confidence_score=0.95,
            mechanism_ids=[mech.id],
        )
        # 0.95 × 0.30 = 0.285 — skill is correctly demoted
        adj = sil._adjusted_confidence(skill)
        assert adj == pytest.approx(0.285)

    def test_multiple_mechanisms_averaged(self, skill_registry, mechanism_store):
        sil = SkillIntelligenceNode("sil", skill_registry, mechanism_store)
        m1 = mechanism_store.create(
            description="m1", preconditions=["a"],
            process_steps=["b"], expected_outcomes=["c"],
            confidence=0.9,
        )
        m2 = mechanism_store.create(
            description="m2", preconditions=["a"],
            process_steps=["b"], expected_outcomes=["c"],
            confidence=0.5,
        )
        skill = Skill(
            skill_id="s4",
            name="Multi-mech Skill",
            description="test",
            category="test",
            steps=["step1"],
            tools_used=[],
            success_criteria="ok",
            confidence_score=0.8,
            mechanism_ids=[m1.id, m2.id],
        )
        # avg mech conf = (0.9 + 0.5) / 2 = 0.7
        # adjusted = 0.8 * 0.7 = 0.56
        adj = sil._adjusted_confidence(skill)
        assert adj == pytest.approx(0.56)


class TestSkillMechanismLinking:
    """SkillRegistry should support mechanism-linked skills."""

    def test_link_mechanism_to_skill(self, skill_registry):
        skill = skill_registry.extract_and_store(
            task_description="Install PostgreSQL",
            execution_trace=[{"action": "apt install postgresql"}],
            tools_used=["apt"],
            success=True,
            category="devops",
        )
        assert skill is not None

        skill_registry.link_mechanism(skill.skill_id, "mech_dep_resolution")

        retrieved = skill_registry.get_skill(skill.skill_id)
        assert "mech_dep_resolution" in retrieved.mechanism_ids

    def test_find_skill_by_mechanism(self, skill_registry):
        skill = skill_registry.extract_and_store(
            task_description="Install Redis",
            execution_trace=[{"action": "apt install redis-server"}],
            tools_used=["apt"],
            success=True,
            category="devops",
        )
        skill_registry.link_mechanism(skill.skill_id, "mech_pkg_mgr")

        found = skill_registry.find_skill_by_mechanism("mech_pkg_mgr")
        assert len(found) >= 1
        assert found[0].skill_id == skill.skill_id

    def test_causal_model_id_stored(self, skill_registry):
        skill = skill_registry.extract_and_store(
            task_description="Deploy to K8s",
            execution_trace=[{"action": "kubectl apply"}],
            tools_used=["kubectl"],
            success=True,
            category="devops",
        )
        skill_registry.link_mechanism(
            skill.skill_id, "mech_container", causal_model_id="model_123"
        )

        retrieved = skill_registry.get_skill(skill.skill_id)
        assert retrieved.causal_model_id == "model_123"


class TestFailureRootCauseAnalysis:
    """FailureAnalyzer should correctly classify failures."""

    def test_auth_not_belief_error(self):
        analyzer = FailureAnalyzer()
        root = analyzer.analyze(
            expected="API returns data",
            actual="401 Unauthorized",
        )
        assert root.category == FailureCategory.AUTH_FAILURE
        assert not root.requires_belief_revision

    def test_stale_knowledge_is_belief_error(self):
        analyzer = FailureAnalyzer()
        root = analyzer.analyze(
            expected="Database schema has column 'email'",
            actual="Column 'email' not found",
            context={"previous_success": True},
        )
        assert root.category == FailureCategory.STALE_KNOWLEDGE
        assert root.requires_belief_revision

    def test_true_contradiction_is_belief_error(self):
        analyzer = FailureAnalyzer()
        root = analyzer.analyze(
            expected="Service is running",
            actual="Service is stopped",
        )
        assert root.category == FailureCategory.TRUE_CONTRADICTION
        assert root.requires_belief_revision


class TestMechanismTransferLearning:
    """Mechanisms enable transfer learning across domains."""

    def test_same_mechanism_different_skills(self, skill_registry, mechanism_store):
        """Install PostgreSQL, Install Redis, Install Docker all share
        Package Manager Dependency Resolution."""
        mech = mechanism_store.create(
            description="Package Manager Dependency Resolution",
            preconditions=["package manager installed", "internet access"],
            process_steps=["resolve dependencies", "download packages", "install"],
            expected_outcomes=["packages installed", "dependencies satisfied"],
            domain="devops",
            abstraction_level=1,
        )

        for name in ["Install PostgreSQL", "Install Redis", "Install Docker"]:
            skill = skill_registry.extract_and_store(
                task_description=name,
                execution_trace=[{"action": f"apt install {name.split()[-1].lower()}"}],
                tools_used=["apt"],
                success=True,
                category="devops",
            )
            skill_registry.link_mechanism(skill.skill_id, mech.id)

        # All three skills share the same mechanism
        linked_skills = skill_registry.find_skill_by_mechanism(mech.id)
        assert len(linked_skills) == 3

        # When mechanism is reinforced, all skills benefit
        mechanism_store.record_usage(mech.id, success=True)
        mechanism_store.record_usage(mech.id, success=True)
        updated = mechanism_store.get(mech.id)
        assert updated.usage_count == 2
        assert updated.confidence > 0.8


class TestMechanismAbstraction:
    """Sleep consolidation: SQL+XSS+LDAP → Injection Vulnerability."""

    def test_abstraction_level_hierarchy(self, mechanism_store):
        # Concrete mechanisms
        sql = mechanism_store.create(
            description="SQL Injection Prevention",
            preconditions=["user input", "SQL query"],
            process_steps=["parameterize query"],
            expected_outcomes=["safe query"],
            domain="security",
            abstraction_level=0,
        )
        xss = mechanism_store.create(
            description="XSS Prevention",
            preconditions=["user input", "HTML rendering"],
            process_steps=["escape HTML entities"],
            expected_outcomes=["safe rendering"],
            domain="security",
            abstraction_level=0,
        )
        ldap = mechanism_store.create(
            description="LDAP Injection Prevention",
            preconditions=["user input", "LDAP query"],
            process_steps=["escape LDAP special chars"],
            expected_outcomes=["safe LDAP query"],
            domain="security",
            abstraction_level=0,
        )

        # Abstract mechanism (would be discovered by ConceptFormation during sleep)
        injection = mechanism_store.create(
            description="Injection Vulnerability Prevention",
            preconditions=["user input", "interpreter", "query language"],
            process_steps=["sanitize input", "use safe API"],
            expected_outcomes=["no code injection possible"],
            domain="security",
            abstraction_level=2,  # Cross-domain
        )

        # Link concrete to abstract
        for m in [sql, xss, ldap]:
            m.parent_mechanism_id = injection.id
            mechanism_store.store(m)

        # Verify hierarchy
        for m_id in [sql.id, xss.id, ldap.id]:
            m = mechanism_store.get(m_id)
            assert m.parent_mechanism_id == injection.id

        # Abstract mechanism should match "user input" situations
        results = mechanism_store.find_by_preconditions(
            ["user", "input", "form", "submission"]
        )
        # Higher abstraction_level gets boosted in scoring
        assert any(r.id == injection.id for r in results)
