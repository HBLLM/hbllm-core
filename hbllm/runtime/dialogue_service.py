"""Dialogue Turn Execution & Formatting Service extracted from runtime/bootstrap.py."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import TYPE_CHECKING, Any

from hbllm.hcir.language.core.semantic_frame import FrameType

if TYPE_CHECKING:
    from hbllm.hcir.kernel.services import KernelServices
    from hbllm.hcir.workspace import HCIRWorkspaceState
    from hbllm.hcir.workspace_tiers import TieredWorkspace

logger = logging.getLogger("hbllm.runtime.dialogue")


def _format_capability_result(receipt: Any) -> str:
    """Format a CapabilityDispatcher receipt safely without leaking raw Python dicts or enums."""
    res = getattr(receipt, "result", None)
    if not isinstance(res, dict):
        return f"Completed {getattr(receipt, 'verb', 'action')}."

    content = res.get("content")
    if isinstance(content, str) and content.strip():
        return content

    entries = res.get("entries")
    if isinstance(entries, list) and entries:
        return "\n".join(
            str(e.get("name", e)) if isinstance(e, dict) else str(e) for e in entries[:20]
        )

    matches = res.get("matches")
    if isinstance(matches, list) and matches:
        return "\n".join(
            str(m.get("name", m.get("path", m))) if isinstance(m, dict) else str(m)
            for m in matches[:20]
        )

    stdout = res.get("stdout")
    if isinstance(stdout, str) and stdout.strip():
        return stdout.strip()

    lines = res.get("lines")
    if isinstance(lines, list) and lines:
        return "\n".join(str(line) for line in lines[:30])

    if "functions" in res or "classes" in res:
        fns = res.get("functions") or []
        clss = res.get("classes") or []
        return f"Code AST analyzed ({len(clss)} classes, {len(fns)} functions)."

    if "candidates_generated" in res:
        count = int(res.get("candidates_generated", 0) or 0)
        selected = res.get("selected_plan", "none")
        return f"Generated {count} candidate plans (selected {selected})."

    if "valid" in res:
        valid = bool(res.get("valid"))
        reasons = res.get("reasons") or []
        detail = f": {'; '.join(str(r) for r in reasons)}" if reasons else "."
        return ("Plan validation passed" if valid else "Plan validation failed") + detail

    subgoal_desc = res.get("active_subgoal_description")
    if isinstance(subgoal_desc, str) and subgoal_desc.strip():
        return f"Active subgoal: {subgoal_desc.strip()}."

    if res.get("error"):
        return str(res["error"])

    return f"Executed {getattr(receipt, 'verb', 'command')}."


class DialogueService:
    """Encapsulates Pure-HCIR multi-clause & single-turn dialogue execution."""

    def __init__(
        self,
        *,
        services: KernelServices,
        workspace: HCIRWorkspaceState,
        tiered_workspace: TieredWorkspace,
        config: Any,
        bridge: Any | None = None,
        brain: Any | None = None,
    ) -> None:
        self.services = services
        self.workspace = workspace
        self.tiered_workspace = tiered_workspace
        self.config = config
        self.bridge = bridge
        self.brain = brain
        self._loop: asyncio.AbstractEventLoop | None = None

    def _run_async(self, coro: Any) -> Any:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None and running.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result(timeout=30.0)
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def attach_ontology_store(self, store: Any) -> None:
        """Attach or update a LexicalOntologyStore across language runtime, parsers, and capability executors."""
        lang_rt = self.services.language_runtime or self.services.extensions.get("language_runtime")
        if lang_rt is not None and hasattr(lang_rt, "attach_ontology_store"):
            lang_rt.attach_ontology_store(store)
        for cap_name in (
            "knowledge.explain",
            "knowledge.explain_why",
            "knowledge.explain_how",
            "memory.recall",
        ):
            for impl in self.services.capability_resolver.list_implementations(cap_name):
                ex = getattr(impl, "executor", None)
                if ex is not None and hasattr(ex, "_ontology_store"):
                    ex._ontology_store = store
                    if hasattr(ex, "_explainer"):
                        ex._explainer = None
        self.services.ontology_store = store
        self.services.extensions["ontology_store"] = store

    def execute_language_turn(self, turn: Any, lang_rt: Any, speaker: str) -> Any:
        """Execute post-parse cognitive actions (curiosity fallback & command dispatch)."""
        if not turn.is_success:
            return turn

        curiosity = (
            self.services.epistemic_curiosity
            or self.services.curiosity_controller
            or self.services.extensions.get("epistemic_curiosity")
        )
        if (
            curiosity is not None
            and turn.semantic_frame.frame_type == FrameType.QUERY
            and turn.epistemic_state is not None
            and not turn.epistemic_state.is_known
            and turn.epistemic_state.target_subject
        ):
            q_target = turn.semantic_frame.query_target or "definition"
            q_mode = (
                "why"
                if q_target == "cause"
                else ("how" if q_target == "procedure" else "definition")
            )
            try:
                cur_res = self._run_async(
                    curiosity.research_concept(
                        turn.epistemic_state.target_subject,
                        query_mode=q_mode,
                        uncertainty=float(turn.epistemic_state.uncertainty),
                    )
                )
                if cur_res.researched:
                    re_state = lang_rt._gateway._lookup_epistemic_prior(
                        turn.epistemic_state.target_subject,
                        query_target=q_target,
                    )
                    if re_state is not None and re_state.is_known:
                        turn.epistemic_state = re_state
                        realizer = lang_rt._realizers.get(turn.language)
                        if realizer is not None:
                            turn.response_text = realizer.realize(
                                re_state, original_frame=turn.semantic_frame
                            )
                    elif cur_res.summary:
                        turn.response_text = cur_res.summary
                    if cur_res.sources:
                        turn.response_text = (
                            f"{turn.response_text}\n\n"
                            f"[Learned autonomously via web research "
                            f"({cur_res.propositions_ingested} facts ingested); "
                            f"source: {cur_res.sources[0]}]"
                        )
            except Exception as exc:
                logger.debug("Autonomous curiosity query fallback skipped: %s", exc)

        if turn.semantic_frame.frame_type == FrameType.COMMAND and turn.semantic_frame.predicate:
            from hbllm.hcir.kernel.command_dispatcher import CommandDispatcher

            dispatcher = self.services.command_dispatcher or self.services.extensions.get(
                "command_dispatcher"
            )
            if dispatcher is None:
                dispatcher = CommandDispatcher(
                    self.services.capability_resolver,
                    workspace=self.workspace,
                )
                self.services.command_dispatcher = dispatcher
                self.services.extensions["command_dispatcher"] = dispatcher

            resolved_cap = dispatcher._resolve_capability(
                turn.semantic_frame.predicate, frame=turn.semantic_frame
            )
            grounded_ents = (
                getattr(turn.grounded_frame, "grounded_entities", {})
                if turn.grounded_frame is not None
                else {}
            )
            if resolved_cap is None and turn.hcir_node_id is not None and grounded_ents:
                return turn

            try:
                receipt = self._run_async(dispatcher.dispatch(turn.semantic_frame))
                if receipt.success:
                    result_summary = _format_capability_result(receipt)
                    turn.response_text = result_summary
                    from hbllm.hcir.language.core.epistemic_policy import CognitiveEpistemicState

                    concept_target = str(receipt.result.get("concept", receipt.verb))
                    has_gap = "do not have" in result_summary.lower() or not receipt.result.get(
                        "ok", True
                    )
                    turn.epistemic_state = CognitiveEpistemicState(
                        target_predicate=str(receipt.capability),
                        target_subject=concept_target,
                        is_known=not has_gap,
                        confidence=0.10 if has_gap else 0.96,
                        uncertainty=0.90 if has_gap else 0.04,
                        support_count=1 if not has_gap else 0,
                    )
                else:
                    turn.response_text = f"Command '{receipt.verb}' failed: {receipt.error}"
                    turn.is_success = False
            except Exception as exc:
                turn.response_text = f"Command dispatch error: {exc}"
                turn.is_success = False

        return turn

    def process_dialogue(
        self,
        dialogue: Any,
        speaker_id: str = "user",
        language: str | None = None,
    ) -> Any:
        """Process natural language dialogue via Pure-HCIR multi-clause & single-turn pipeline."""
        from hbllm.hcir.grounding.contracts.observation import ObservationEnvelope
        from hbllm.hcir.language.core.discourse import (
            DiscourseSegmenter,
            ExpositoryProseProcessor,
        )
        from hbllm.hcir.language.core.semantic_frame import (
            LanguageErrorType,
            LanguageMetadata,
            SemanticFrame,
        )
        from hbllm.hcir.language.runtime import LanguageTurnResult, MultilingualLanguageRuntime

        if isinstance(dialogue, ObservationEnvelope) and self.bridge is not None:
            return self.bridge.process_dialogue(dialogue, speaker_id=speaker_id)

        text = (
            str(dialogue.metadata.get("text", ""))
            if isinstance(dialogue, ObservationEnvelope)
            else str(dialogue)
        )
        speaker = (
            str(dialogue.metadata.get("speaker_id", speaker_id))
            if isinstance(dialogue, ObservationEnvelope)
            else speaker_id
        )

        if language is None:
            if any("\u0d80" <= ch <= "\u0dff" for ch in text):
                language = "si"
            elif any("\u0b80" <= ch <= "\u0bff" for ch in text):
                language = "ta"
            else:
                language = "en"

        lang_rt = self.services.language_runtime or self.services.extensions.get("language_runtime")
        if lang_rt is None:
            ont_store = self.services.ontology_store or self.services.extensions.get(
                "ontology_store"
            )
            if ont_store is None and self.config.memory_dir is None:
                from hbllm.hcir.language.ontology_store import LexicalOntologyStore

                ont_store = LexicalOntologyStore(":memory:")
                self.services.ontology_store = ont_store
                self.services.extensions["ontology_store"] = ont_store

            lang_rt = MultilingualLanguageRuntime(
                self.workspace.graph,
                workspace=self.workspace,
                ontology_store=ont_store,
            )
            if self.tiered_workspace is not None:
                lang_rt._gateway._tiered_workspace = self.tiered_workspace
            self.services.language_runtime = lang_rt
            self.services.extensions["language_runtime"] = lang_rt

        active_parser = (
            lang_rt._parsers[language] if language in lang_rt._parsers else lang_rt._parsers["en"]
        )
        lang_rt._sync_dynamic_vocabulary(active_parser)
        segments = DiscourseSegmenter.segment(text)
        prose_proc = ExpositoryProseProcessor()

        if len(segments) <= 1:
            turn = lang_rt.process_utterance(text, language=language, speaker=speaker)
            has_complex_clause = any(
                kw in text.lower()
                for kw in (" and ", " because ", " that ", " which ", " who ", " to ")
            )
            if (
                not turn.is_success
                and turn.error_type == LanguageErrorType.UNRESOLVED_LANGUAGE
                and len(text.split()) >= 5
            ) or (
                turn.is_success
                and turn.semantic_frame.frame_type == FrameType.ASSERTION
                and has_complex_clause
            ):
                summary = prose_proc.ingest_prose(
                    text,
                    workspace=self.workspace,
                    tiered_workspace=self.tiered_workspace,
                    ontology_store=lang_rt._ontology_store,
                    reference_resolver=lang_rt._ref_resolver,
                    parser=active_parser,
                )
                if summary.propositions and not turn.is_success:
                    turn = LanguageTurnResult(
                        raw_input=text,
                        language="en",
                        semantic_frame=SemanticFrame(
                            frame_type=FrameType.ASSERTION,
                            predicate="expository_prose",
                            metadata=LanguageMetadata(language="en", raw_text=text),
                        ),
                        response_text=summary.summary_text,
                        is_success=True,
                    )
            if turn.is_success:
                turn = self.execute_language_turn(turn, lang_rt, speaker)
        else:
            lang_rt._ref_resolver.next_turn()
            sub_turns: list[LanguageTurnResult] = []
            for seg in segments:
                sub = lang_rt.process_utterance(
                    seg, language=language, speaker=speaker, advance_turn=False
                )
                if sub.is_success:
                    sub = self.execute_language_turn(sub, lang_rt, speaker)
                    sub_turns.append(sub)
                else:
                    summary = prose_proc.ingest_prose(
                        seg,
                        workspace=self.workspace,
                        tiered_workspace=self.tiered_workspace,
                        ontology_store=lang_rt._ontology_store,
                        reference_resolver=lang_rt._ref_resolver,
                        parser=active_parser,
                    )
                    if summary.propositions:
                        from hbllm.hcir.language.core.epistemic_policy import (
                            CognitiveEpistemicState,
                        )

                        sub_turns.append(
                            LanguageTurnResult(
                                raw_input=seg,
                                language="en",
                                semantic_frame=SemanticFrame(
                                    frame_type=FrameType.ASSERTION,
                                    predicate="expository_prose",
                                    metadata=LanguageMetadata(language="en", raw_text=seg),
                                ),
                                epistemic_state=CognitiveEpistemicState(
                                    target_predicate="expository_prose",
                                    target_subject="prose",
                                    confidence=0.96,
                                    uncertainty=0.04,
                                    support_count=len(summary.propositions),
                                    is_known=True,
                                ),
                                response_text=summary.summary_text,
                                is_success=True,
                            )
                        )
                    else:
                        sub_turns.append(sub)

            actionable = [
                t
                for t in sub_turns
                if t.semantic_frame.frame_type in (FrameType.QUERY, FrameType.COMMAND)
            ]
            assertions = [
                t for t in sub_turns if t.semantic_frame.frame_type == FrameType.ASSERTION
            ]
            if assertions and len(actionable) == 0:
                holistic = prose_proc.ingest_prose(
                    text,
                    workspace=self.workspace,
                    tiered_workspace=self.tiered_workspace,
                    ontology_store=lang_rt._ontology_store,
                    reference_resolver=lang_rt._ref_resolver,
                    parser=active_parser,
                )
            else:
                holistic = None

            if actionable:
                if len(actionable) == 1:
                    response_text = actionable[0].response_text
                else:
                    response_text = "\n\n".join(
                        f"[Step {idx}] {t.response_text}" for idx, t in enumerate(actionable, 1)
                    )
                overall_success = all(t.is_success for t in actionable)
                rep = actionable[-1]
            elif holistic is not None and holistic.propositions:
                response_text = holistic.summary_text
                overall_success = True
                rep = next((t for t in reversed(sub_turns) if t.is_success), sub_turns[-1])
            elif any(t.is_success for t in sub_turns):
                ok_resps = [t.response_text for t in sub_turns if t.is_success and t.response_text]
                response_text = ok_resps[-1] if len(ok_resps) == 1 else " ".join(ok_resps)
                overall_success = True
                rep = next(t for t in reversed(sub_turns) if t.is_success)
            else:
                response_text = sub_turns[0].response_text or "I did not understand that input."
                overall_success = False
                rep = sub_turns[0]

            overall_ep = rep.epistemic_state
            if overall_ep is None and overall_success:
                from hbllm.hcir.language.core.epistemic_policy import CognitiveEpistemicState

                overall_ep = CognitiveEpistemicState(
                    target_predicate="composite_utterance",
                    target_subject="composite",
                    confidence=0.96,
                    uncertainty=0.04,
                    support_count=len(sub_turns),
                    is_known=True,
                )

            turn = LanguageTurnResult(
                raw_input=text,
                language=rep.language,
                semantic_frame=rep.semantic_frame,
                grounded_frame=rep.grounded_frame,
                epistemic_state=overall_ep,
                response_text=response_text,
                is_success=overall_success,
                error_type=None if overall_success else rep.error_type,
                error_detail="" if overall_success else rep.error_detail,
                hcir_node_id=rep.hcir_node_id,
                sub_results=sub_turns,
            )

        ext_metrics = self.services.extensions.get("autonomy_metrics")
        metrics = ext_metrics if ext_metrics is not None else self.services.autonomy_metrics
        if metrics is None:
            metrics = {
                "parse_attempts": 0,
                "parse_hcir_success": 0,
                "realize_attempts": 0,
                "realize_hcir_success": 0,
                "shadow_invocations": 0,
                "distilled_lessons": 0,
            }
        self.services.autonomy_metrics = metrics
        self.services.extensions["autonomy_metrics"] = metrics
        if hasattr(metrics, "record_parse"):
            metrics.record_parse(turn.is_success)
            metrics.record_realize(bool(turn.is_success and turn.response_text))
        else:
            metrics["parse_attempts"] += 1
            if turn.is_success:
                metrics["parse_hcir_success"] += 1
            metrics["realize_attempts"] += 1
            if turn.is_success and turn.response_text:
                metrics["realize_hcir_success"] += 1

        neuromod = self.services.neuromodulation or self.services.extensions.get("neuromodulation")
        if turn.is_success:
            if neuromod is not None and hasattr(neuromod, "signal_reward"):
                neuromod.signal_reward(0.2)
            return turn

        if neuromod is not None and hasattr(neuromod, "signal_novelty"):
            neuromod.signal_novelty(0.3)

        scaffold = self.services.extensions.get("llm_scaffold")
        if (
            scaffold is not None
            and not getattr(scaffold, "retired", False)
            and self.config.enable_brain_llm
        ):
            scaffold_res = scaffold.advise_and_distill(text, self.workspace, speaker_id=speaker)
            if not hasattr(metrics, "record_parse"):
                metrics["shadow_invocations"] += 1
                if scaffold_res.get("distilled"):
                    metrics["distilled_lessons"] += 1
            return scaffold_res

        if self.bridge is not None:
            return self.bridge.process_dialogue(dialogue, speaker_id=speaker_id)
        return turn
