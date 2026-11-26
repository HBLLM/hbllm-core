"""Phase M3 tests: Embodied Construction Grammar, procedural reading & 4-step unknown resolution."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from hbllm.hcir import HCIRCompiler, HCIRInterpreter, SyscallAllowlist
from hbllm.hcir.graph import (
    BeliefNode,
    ConstructionNode,
    HCIRNodeType,
    PhysicalEntityNode,
    SkillNode,
    UnknownNode,
)
from hbllm.hcir.grounding.concepts.model import GroundedConcept
from hbllm.hcir.grounding.concepts.store import WorkingConceptStore
from hbllm.hcir.grounding.contracts.enums import ConceptLifecycle
from hbllm.hcir.grounding.contracts.references import GroundedConceptRef
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.language.core.discourse import ExpositoryProseProcessor
from hbllm.hcir.language.core.semantic_frame import FrameType
from hbllm.hcir.language.reading_curriculum import (
    ReadingCurriculumIngestor,
    extract_procedural_definitions,
)
from hbllm.hcir.language.runtime import MultilingualLanguageRuntime
from hbllm.hcir.seed import hydrate_core_priors
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Scope, SecurityLevel
from hbllm.hcir.workspace_tiers import TieredWorkspace


class TestMultilingualConstructionsSeedAndAST:
    def test_hydrate_constructions_multilingual_en_si_ta_and_fork_isolation(self) -> None:
        tw = TieredWorkspace()
        KernelServices.create_default(tw.working_state, tiered_workspace=tw)
        count = hydrate_core_priors(tw)
        assert count > 10

        cxns = [
            n
            for n in tw.brain.graph.nodes_by_type(HCIRNodeType.CONSTRUCTION)
            if isinstance(n, ConstructionNode)
        ]
        langs = {c.language for c in cxns}
        assert {"en", "si", "ta"}.issubset(langs)
        assert all(c.scope.security_level == SecurityLevel.SYSTEM for c in cxns)

        # Verify CoW fork isolation + multilingual compilation across en, si, ta
        forked = tw.brain.fork("fork_m3_test")
        assert len(forked.graph.nodes_by_type(HCIRNodeType.CONSTRUCTION)) == len(cxns)

        for lang, utterance, exp_action in (
            ("en", "push the gate", "push"),
            ("si", "දොර අරින්න", "අරින්න"),
            ("ta", "கதவு திற", "திற"),
        ):
            compiler = HCIRCompiler(tiered_workspace=tw, language=lang)
            ast_obj = compiler.parse_utterance(utterance)
            assert ast_obj.action == exp_action
            stream = compiler.compile(ast_obj)
            assert stream.length == 2
            assert stream.instructions[0].opcode.value == "ASSERT"
            assert stream.instructions[1].opcode.value == "EXECUTE"

    def test_compiler_py_has_zero_hardcoded_verb_or_color_tables(self) -> None:
        src = (Path(__file__).resolve().parents[3] / "hbllm" / "hcir" / "compiler.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(src)
        literals = {
            node.value.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        forbidden_hardcoded = {
            "push",
            "grab",
            "pick up",
            "navigate to",
            "find why",
            "diagnose",
            "magenta",
            "cyan",
            "adjacent to",
        }
        leaked = forbidden_hardcoded & literals
        assert not leaked, f"Hardcoded lexical constants leaked into compiler.py: {leaked}"


class TestCompositionalNovelInstructionExecution:
    @pytest.mark.asyncio
    async def test_teach_novel_primitives_via_text_and_execute_composites_zero_shot(self) -> None:
        tw = TieredWorkspace()
        services = KernelServices.create_default(tw.working_state, tiered_workspace=tw)
        hydrate_core_priors(tw)

        store = WorkingConceptStore()
        prose = ExpositoryProseProcessor()
        ingestor = ReadingCurriculumIngestor()

        # 10 composite novel instructions taught purely through text using coined verbs
        pairs = [
            ("unlatch", "extract", "salvage", "vessel", "interact", "pickup"),
            ("beacon", "dock", "moor", "shuttle", "activate", "navigate"),
            ("tether", "stow", "archive", "module", "manipulate", "drop"),
            ("probe", "calibrate", "commission", "array", "interact", "align"),
            ("scout", "harvest", "forage", "beacon", "navigate", "pickup"),
            ("ping", "latch", "engage", "relay", "activate", "interact"),
            ("haul", "park", "berth", "pod", "manipulate", "drop"),
            ("scan", "balance", "tune", "rotor", "interact", "align"),
            ("deploy", "prime", "arm", "drone", "navigate", "activate"),
            ("hoist", "couple", "assemble", "truss", "pickup", "align"),
        ]

        training_lines: list[str] = []
        for p1, p2, comp, noun, m1, m2 in pairs:
            training_lines.append(f"To {p1} a {noun} means to {m1} the {noun}.")
            training_lines.append(f"To {p2} a {noun} means to {m2} the {noun}.")
            training_lines.append(
                f"To {comp} a {noun} means to {p1} the {noun} and then {p2} the {noun}."
            )

        # Ingest part via ReadingCurriculumIngestor and part via ExpositoryProseProcessor
        midpoint = len(training_lines) // 2
        book_stats = ingestor.ingest_book("\n".join(training_lines[:midpoint]), tw)
        prose_summary = prose.ingest_prose(
            "\n".join(training_lines[midpoint:]),
            workspace=tw.working_state,
            tiered_workspace=tw,
        )
        assert book_stats.nodes_committed > 0
        assert prose_summary.nodes_committed > 0

        # Register grounded scene entities cleanly via transactions
        ent_ops: list[TransactionOperation] = []
        for _, _, _, noun, _, _ in pairs:
            target_id = f"e_{noun}_1"
            ent_ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=target_id,
                    node_data=PhysicalEntityNode(
                        id=target_id, entity_name=f"alpha {noun}", status="ready"
                    ).model_dump(),
                )
            )
            c = GroundedConcept(
                concept_ref=GroundedConceptRef(id=f"C-{noun.upper()}"),
                lifecycle_state=ConceptLifecycle.GROUNDED,
                exemplar_entity_ids={target_id},
            )
            c.add_linguistic_alias(f"alpha {noun}", confidence=1.0)
            store.register_concept(c)
        tw.working_state.commit_operations(ent_ops, author="scene_setup")

        compiler = HCIRCompiler(tiered_workspace=tw, concept_store=store)
        interp = HCIRInterpreter(
            tw.working_state, services, syscall_allowlist=SyscallAllowlist(strict=True)
        )

        # Zero-shot composite execution + multi-clause sequence execution (12 instructions total)
        test_prompts: list[tuple[str, tuple[str, ...]]] = [
            (f"{comp} the alpha {noun}", (f"motor.{m1}", f"motor.{m2}"))
            for _, _, comp, noun, m1, m2 in pairs
        ]
        test_prompts.append(
            (
                "unlatch the alpha vessel and then extract the alpha vessel",
                ("motor.interact", "motor.pickup"),
            )
        )
        test_prompts.append(
            (
                "beacon the alpha shuttle and then dock the alpha shuttle",
                ("motor.activate", "motor.navigate"),
            )
        )

        successes = 0
        for prompt, expected_caps in test_prompts:
            stream = compiler.compile_text(prompt, author="m3_test")
            res, receipt = await interp.execute_with_receipt(stream)
            if res.success and all(cap in receipt.capabilities_used for cap in expected_caps):
                successes += 1

        assert successes / len(test_prompts) >= 0.90
        assert successes == len(test_prompts)


class TestUnknownInstructionResolutionPipeline:
    @pytest.mark.asyncio
    async def test_step_b_memory_recall_resolves_and_caches_construction(self) -> None:
        tw = TieredWorkspace()
        services = KernelServices.create_default(tw.working_state, tiered_workspace=tw)
        store = WorkingConceptStore()
        c = GroundedConcept(
            concept_ref=GroundedConceptRef(id="C-VALVE"),
            lifecycle_state=ConceptLifecycle.GROUNDED,
            exemplar_entity_ids={"e_valve_1"},
        )
        c.add_linguistic_alias("pressure valve", confidence=1.0)
        store.register_concept(c)
        tw.working_state.commit_operations(
            [
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id="e_valve_1",
                    node_data=PhysicalEntityNode(
                        id="e_valve_1", entity_name="pressure valve"
                    ).model_dump(),
                ),
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id="b_def_quell",
                    node_data=BeliefNode(
                        id="b_def_quell",
                        claim="To quell a valve means to activate the valve.",
                        scope=Scope(tenant_id="default"),
                    ).model_dump(),
                ),
            ],
            author="reading_memory",
        )

        compiler = HCIRCompiler(tiered_workspace=tw, concept_store=store)
        stream = compiler.compile_text("quell the pressure valve")
        assert any(i.opcode.value == "CALL" for i in stream.instructions)

        interp = HCIRInterpreter(
            tw.working_state, services, syscall_allowlist=SyscallAllowlist(strict=True)
        )
        res, receipt = await interp.execute_with_receipt(stream)
        assert res.success
        assert "motor.activate" in receipt.capabilities_used
        # Consolidated into a ConstructionNode for future zero-shot reuse
        cxns = tw.brain.graph.nodes_by_type(HCIRNodeType.CONSTRUCTION)
        assert any(isinstance(x, ConstructionNode) and "quell" in x.trigger_tokens for x in cxns)

    def test_step_c_structural_analogy_matches_existing_skill(self) -> None:
        tw = TieredWorkspace()
        KernelServices.create_default(tw.working_state, tiered_workspace=tw)
        from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode

        tw.working_state.commit_operations(
            [
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id="skill_toggle_conduit",
                    node_data=SkillNode(
                        id="skill_toggle_conduit",
                        skill_name="flip_conduit",
                        description="Activate heavy plasma conduit breaker",
                        parameters=[{"name": "target", "role": "target", "required": False}],
                        precondition_pattern={"role": "target", "entity": "plasma conduit"},
                        postcondition_pattern={"intent": "flip_conduit"},
                        bytecode_stream=InstructionStream(
                            instructions=[
                                Instruction(
                                    opcode=Opcode.EXECUTE,
                                    params={
                                        "capability": "motor.activate",
                                        "params": {"target": "$r0"},
                                    },
                                ),
                                Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]}),
                            ]
                        ).model_dump(),
                    ).model_dump(),
                )
            ],
            author="prior_experience",
        )
        store = WorkingConceptStore()
        c = GroundedConcept(
            concept_ref=GroundedConceptRef(id="C-CONDUIT"),
            lifecycle_state=ConceptLifecycle.GROUNDED,
            exemplar_entity_ids={"e_conduit_1"},
        )
        c.add_linguistic_alias("plasma conduit", confidence=1.0)
        store.register_concept(c)

        compiler = HCIRCompiler(tiered_workspace=tw, concept_store=store)
        stream = compiler.compile_text("energize the plasma conduit")
        call_ins = next((i for i in stream.instructions if i.opcode.value == "CALL"), None)
        assert call_ins is not None
        assert call_ins.params["skill_id"] == "skill_toggle_conduit"

    def test_step_d_unresolved_unknown_commits_unknown_node_and_requests_evidence(self) -> None:
        tw = TieredWorkspace()
        KernelServices.create_default(tw.working_state, tiered_workspace=tw)
        store = WorkingConceptStore()
        c = GroundedConcept(
            concept_ref=GroundedConceptRef(id="C-MONOLITH"),
            lifecycle_state=ConceptLifecycle.GROUNDED,
            exemplar_entity_ids={"e_mono_1"},
        )
        c.add_linguistic_alias("monolith", confidence=1.0)
        store.register_concept(c)

        compiler = HCIRCompiler(tiered_workspace=tw, concept_store=store)
        stream = compiler.compile_text("frobnicate the monolith")
        opcodes = [i.opcode.value for i in stream.instructions]
        assert "EXECUTE" not in opcodes
        assert opcodes == ["ASSERT", "FORK", "QUERY", "ROLLBACK", "WAIT"]
        assert stream.instructions[2].params["query_type"] == "request_evidence"
        unknowns = tw.brain.graph.nodes_by_type(HCIRNodeType.UNKNOWN)
        assert any(isinstance(u, UnknownNode) and "frobnicate" in u.question for u in unknowns)

    def test_runtime_syncs_read_verbs_into_english_parser(self) -> None:
        tw = TieredWorkspace()
        KernelServices.create_default(tw.working_state, tiered_workspace=tw)
        hydrate_core_priors(tw)

        assert extract_procedural_definitions("To calibrate a sensor means to align the sensor.")
        prose = ExpositoryProseProcessor()
        summary = prose.ingest_prose(
            "To calibrate a sensor means to align the sensor.",
            workspace=tw.working_state,
            tiered_workspace=tw,
        )
        assert summary.nodes_committed >= 2

        rt = MultilingualLanguageRuntime(tw.persistent.graph, workspace=tw.working_state)
        res = rt.process_utterance("Calibrate the sensor.", language="en")
        assert res.is_success
        assert res.semantic_frame.frame_type == FrameType.COMMAND
        assert res.semantic_frame.predicate == "calibrate"
