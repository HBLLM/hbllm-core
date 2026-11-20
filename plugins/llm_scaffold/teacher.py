"""Parallel LLM Teacher Scaffold & Self-Detaching Autonomy Tracker.

Strictly isolated in ``plugins/llm_scaffold/`` (zero imports from here inside
``hbllm/hcir/``). Defaults offline to ``checkpoints/Llama-3.2-1B``.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hbllm.hcir.kernel.autonomy_tracker import AutonomyTracker
from hbllm.hcir.language.ontology_bootstrap import LexicalOntologyRecord
from hbllm.hcir.nodes import BeliefNode, PhysicalEntityNode
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_CHECKPOINT = Path("checkpoints/Llama-3.2-1B")


class LLMTeacherScaffold:
    """Offline-first teacher scaffold that distills verified lessons into HCIR."""

    def __init__(
        self,
        checkpoint_path: Path | str = DEFAULT_LOCAL_CHECKPOINT,
        tracker: AutonomyTracker | None = None,
        teacher_fn: Callable[[str], dict[str, Any]] | None = None,
        ontology_store: Any | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.tracker = tracker or AutonomyTracker()
        self._teacher_fn = teacher_fn
        self._ontology_store = ontology_store
        self.retired = False

    @classmethod
    def from_local_checkpoint(
        cls,
        checkpoint_path: Path | str = DEFAULT_LOCAL_CHECKPOINT,
        *,
        tracker: AutonomyTracker | None = None,
        ontology_store: Any | None = None,
        max_new_tokens: int = 160,
    ) -> LLMTeacherScaffold:
        """Create a scaffold backed by a local HuggingFace checkpoint (offline)."""
        from plugins.llm_scaffold.local_teacher import LocalHFTeacher

        teacher = LocalHFTeacher(checkpoint_path=checkpoint_path, max_new_tokens=max_new_tokens)
        return cls(
            checkpoint_path=checkpoint_path,
            tracker=tracker,
            teacher_fn=teacher,
            ontology_store=ontology_store,
        )

    @classmethod
    def from_nvidia_endpoint(
        cls,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        tracker: AutonomyTracker | None = None,
        ontology_store: Any | None = None,
        timeout_s: float = 30.0,
    ) -> LLMTeacherScaffold:
        """Create a scaffold backed by an NVIDIA NIM OpenAI-compatible endpoint."""
        from plugins.llm_scaffold.nvidia_teacher import NvidiaNIMTeacher

        teacher = NvidiaNIMTeacher(
            api_key=api_key,
            model=model,
            base_url=base_url,
            timeout_s=timeout_s,
        )
        return cls(
            checkpoint_path=f"nvidia:{teacher.model}",
            tracker=tracker,
            teacher_fn=teacher,
            ontology_store=ontology_store,
        )

    @property
    def checkpoint_available(self) -> bool:
        return self.checkpoint_path.exists()

    def advise_and_distill(
        self,
        utterance: str,
        workspace: Any,
        *,
        speaker_id: str = "user",
        confidence_floor: float = 0.65,
    ) -> dict[str, Any]:
        """Consult the teacher scaffold (if not retired) and distill verified lessons."""
        if self.retired or self.tracker.can_retire_llm:
            self.retired = True
            return {"retired": True, "response_text": "", "distilled": False}

        self.tracker.shadow_invocations += 1
        if self._teacher_fn is not None:
            suggestion = self._teacher_fn(utterance)
        else:
            suggestion = {
                "response_text": "",
                "claim": "",
                "entities": [],
                "lexicon": [],
                "confidence": 0.0,
                "verified": False,
            }

        claim = str(suggestion.get("claim", "")).strip()
        entities = suggestion.get("entities") or []
        lexicon_items = suggestion.get("lexicon") or []
        confidence = float(suggestion.get("confidence", 0.0))
        verified = bool(suggestion.get("verified", False))
        distilled = False

        if not verified or confidence < confidence_floor:
            if claim or entities or lexicon_items:
                self.tracker.rejected_lessons += 1
            return {
                "retired": False,
                "response_text": str(suggestion.get("response_text", "")),
                "distilled": False,
                "checkpoint": str(self.checkpoint_path),
            }

        # 1. Distill verified open-vocabulary lexicon entries into LexicalOntologyStore
        if lexicon_items and self._ontology_store is not None:
            records = [
                LexicalOntologyRecord(
                    lemma=str(item["lemma"]).strip().lower(),
                    pos=str(item.get("pos", "noun")).strip().lower(),
                    language="en",
                    definition=str(item.get("definition", "")),
                    hypernyms=tuple(item.get("hypernyms") or ()),
                    domain="t3_shadow_distilled",
                )
                for item in lexicon_items
                if isinstance(item, dict) and item.get("lemma")
            ]
            if records:
                upserted, _ = self._ontology_store.upsert_many(records, source="t3_llm_scaffold")
                self.tracker.distilled_lexicon_entries += upserted

        # 2. Distill verified PhysicalEntityNodes + BeliefNode atomically via TransactionManager
        ops: list[TransactionOperation] = []
        entity_count = 0
        for ent in entities:
            if not isinstance(ent, dict) or not ent.get("name"):
                continue
            ename = str(ent["name"]).strip().lower()
            etype = str(ent.get("type", ename)).strip().lower()
            eid = f"scaffold_ent_{ename}"
            ent_props = dict(ent.get("properties") or {})
            if etype and etype != ename:
                ent_props.setdefault("semantic_type", etype)
            ent_node = PhysicalEntityNode(
                id=eid,
                entity_name=ename,
                entity_type=ename,
                properties=ent_props,
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=ent_node.id,
                    node_data=ent_node.model_dump(),
                )
            )
            entity_count += 1

        if claim:
            belief = BeliefNode(
                id=f"scaffold_lesson_{uuid.uuid4().hex[:10]}",
                statement=claim,
                confidence=min(1.0, confidence),
                provenance=Provenance(
                    source="llm_scaffold",
                    confidence=min(1.0, confidence),
                    evidence=[f"speaker:{speaker_id}", f"utterance:{utterance[:80]}"],
                ),
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=belief.id,
                    node_data=belief.model_dump(),
                )
            )

        if ops and workspace is not None:
            commit_fn = getattr(workspace, "commit_operations", None)
            if callable(commit_fn):
                receipt = commit_fn(ops, author="llm_scaffold")
                distilled = bool(getattr(receipt, "is_committed", False))
            if distilled:
                self.tracker.distilled_lessons += 1
                self.tracker.distilled_entities += entity_count
            else:
                self.tracker.rejected_lessons += 1

        return {
            "retired": False,
            "response_text": str(suggestion.get("response_text", "")),
            "distilled": distilled,
            "checkpoint": str(self.checkpoint_path),
        }
