from hbllm.brain.cognitive_state import CandidatePlan
from hbllm.brain.concept_formation import ConceptFormationEngine, CrossDomainAnalogy
from hbllm.brain.skill_registry import Skill, SkillRegistry


def test_generate_analogous_hypothesis(tmp_path):
    # Setup engines
    concept_engine = ConceptFormationEngine()
    skill_registry = SkillRegistry(data_dir=str(tmp_path))

    # 1. Setup mock CrossDomainAnalogy in concept_engine
    analogy = CrossDomainAnalogy(
        analogy_id="anlg_db_fs",
        domain_a="database",
        domain_b="filesystem",
        concept_a="db_write",
        concept_b="fs_write",
        shared_structure={},
        similarity_score=0.85,
    )
    concept_engine._analogies.append(analogy)

    # 2. Add counterpart skill in registry (under concept_b: 'fs_write')
    # Use simple flat steps for fallback conversion
    counterpart_skill = Skill(
        skill_id="fs_write_skill",
        name="fs_write",
        description="Write text to a file system",
        category="filesystem",
        steps=["open file", "write data to file", "close file"],
        tools_used=[],
        success_criteria="ok",
    )
    skill_registry._store(counterpart_skill)

    # 3. Request analogous plan hypothesis for concept_a: 'db_write'
    context = {"analogy_mappings": {"file": "database table", "write": "insert"}}

    hypothesis = concept_engine.generate_analogous_hypothesis(
        concept="db_write", skill_registry=skill_registry, target_context=context
    )

    assert isinstance(hypothesis, CandidatePlan)
    assert hypothesis.origin == "analogy"
    assert hypothesis.analogy_used == "anlg_db_fs"
    assert hypothesis.confidence == 0.6

    # Check that nodes/edges are translated correctly
    nodes = hypothesis.graph["nodes"]
    assert len(nodes) == 3
    assert nodes[0]["action"] == "open database table"
    assert nodes[1]["action"] == "insert data to database table"
    assert nodes[2]["action"] == "close database table"

    edges = hypothesis.graph["edges"]
    assert len(edges) == 2
    assert edges[0]["source"] == "step_0"
    assert edges[0]["target"] == "step_1"
