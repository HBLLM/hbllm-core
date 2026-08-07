"""Epistemic Workspace — runtime container for long-lived research programs.

Provides the ``ResearchProgram`` (the cognitive object) and
``DiscoveryWorkspace`` (the runtime container that manages its lifecycle).

Architecture::

    ResearchProgram (cognitive object)
        │
        ├── research_question
        ├── hypotheses
        ├── evidence
        ├── unknowns
        ├── experiments
        ├── predictions
        ├── contradictions
        ├── findings
        └── journal entries
                │
                ▼
    DiscoveryWorkspace (runtime container)
        │
        ├── owns CognitiveGraph scoped view
        ├── manages hypothesis lifecycle
        ├── coordinates experiment pipeline
        ├── tracks confidence timeline
        └── produces discovery journal

The ``DiscoveryWorkspace`` integrates with the existing ``TieredWorkspace``
by living in the Persistent tier.  It does NOT create a separate database
or memory system — it uses HCIR nodes in the shared graph.

Usage::

    from hbllm.brain.epistemics.workspace import DiscoveryWorkspace

    workspace = DiscoveryWorkspace(data_dir=Path("./research"))
    program = workspace.create_program(
        title="Understanding Alzheimer's progression",
        research_question="What mechanisms drive Alzheimer's progression?",
    )
    workspace.add_unknown(program.program_id, "Why does tau accumulate?")
    workspace.add_hypothesis(program.program_id, hypothesis_node)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hbllm.hcir.graph import (
    CognitiveGraph,
    ContradictionNode,
    EvidenceNode,
    ExperimentNode,
    HCIREdge,
    HCIREdgeType,
    HypothesisLifecycle,
    HypothesisNode,
    PredictionNode,
    ResearchObjectiveNode,
    ResearchProgramNode,
    UnknownNode,
)
from hbllm.hcir.types import CognitiveMode, FalsificationStatus, ResearchStrategyType

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Research Program — the cognitive object
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ConfidenceSnapshot:
    """A point-in-time snapshot of research confidence."""

    timestamp: float = field(default_factory=time.time)
    overall_confidence: float = 0.0
    hypothesis_confidences: dict[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass
class JournalEntry:
    """A single entry in the discovery journal."""

    entry_id: str = field(default_factory=lambda: f"dje_{uuid.uuid4().hex[:10]}")
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""
    # "hypothesis_generated", "prediction_made", "prediction_verified",
    # "experiment_completed", "contradiction_found", "belief_revised",
    # "unknown_resolved", "evidence_added"
    description: str = ""
    related_node_ids: list[str] = field(default_factory=list)
    confidence_before: float | None = None
    confidence_after: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "description": self.description,
            "related_node_ids": self.related_node_ids,
            "confidence_before": self.confidence_before,
            "confidence_after": self.confidence_after,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> JournalEntry:
        return cls(
            entry_id=d.get("entry_id", f"dje_{uuid.uuid4().hex[:10]}"),
            timestamp=d.get("timestamp", time.time()),
            event_type=d.get("event_type", ""),
            description=d.get("description", ""),
            related_node_ids=d.get("related_node_ids", []),
            confidence_before=d.get("confidence_before"),
            confidence_after=d.get("confidence_after"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class ResearchProgram:
    """The cognitive object for a long-lived research program.

    This is the high-level research context that owns questions,
    hypotheses, evidence, experiments, and findings.  It may persist
    for months or years.

    The ``DiscoveryWorkspace`` manages its runtime lifecycle.
    The ``ResearchProgramNode`` is its HCIR graph representation.
    """

    program_id: str = field(default_factory=lambda: f"rp_{uuid.uuid4().hex[:12]}")
    title: str = ""
    research_question: str = ""
    description: str = ""
    status: str = "active"  # "active", "paused", "completed", "abandoned"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # References to HCIR node IDs
    hypothesis_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    experiment_ids: list[str] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)
    finding_ids: list[str] = field(default_factory=list)
    contradiction_ids: list[str] = field(default_factory=list)
    prediction_ids: list[str] = field(default_factory=list)
    objective_ids: list[str] = field(default_factory=list)

    # Strategy
    current_strategy: ResearchStrategyType = ResearchStrategyType.EXPLORATION

    # Timeline
    confidence_timeline: list[ConfidenceSnapshot] = field(default_factory=list)
    journal: list[JournalEntry] = field(default_factory=list)

    # Cognitive mode
    cognitive_mode: CognitiveMode = CognitiveMode.DISCOVERY

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "title": self.title,
            "research_question": self.research_question,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "hypothesis_ids": self.hypothesis_ids,
            "evidence_ids": self.evidence_ids,
            "experiment_ids": self.experiment_ids,
            "unknown_ids": self.unknown_ids,
            "finding_ids": self.finding_ids,
            "contradiction_ids": self.contradiction_ids,
            "prediction_ids": self.prediction_ids,
            "objective_ids": self.objective_ids,
            "current_strategy": self.current_strategy.value,
            "confidence_timeline": [
                {
                    "timestamp": s.timestamp,
                    "overall_confidence": s.overall_confidence,
                    "hypothesis_confidences": s.hypothesis_confidences,
                    "reason": s.reason,
                }
                for s in self.confidence_timeline
            ],
            "journal": [e.to_dict() for e in self.journal],
            "cognitive_mode": self.cognitive_mode.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResearchProgram:
        timeline = [
            ConfidenceSnapshot(
                timestamp=s.get("timestamp", 0.0),
                overall_confidence=s.get("overall_confidence", 0.0),
                hypothesis_confidences=s.get("hypothesis_confidences", {}),
                reason=s.get("reason", ""),
            )
            for s in d.get("confidence_timeline", [])
        ]
        journal = [JournalEntry.from_dict(e) for e in d.get("journal", [])]
        return cls(
            program_id=d.get("program_id", f"rp_{uuid.uuid4().hex[:12]}"),
            title=d.get("title", ""),
            research_question=d.get("research_question", ""),
            description=d.get("description", ""),
            status=d.get("status", "active"),
            created_at=d.get("created_at", time.time()),
            updated_at=d.get("updated_at", time.time()),
            hypothesis_ids=d.get("hypothesis_ids", []),
            evidence_ids=d.get("evidence_ids", []),
            experiment_ids=d.get("experiment_ids", []),
            unknown_ids=d.get("unknown_ids", []),
            finding_ids=d.get("finding_ids", []),
            contradiction_ids=d.get("contradiction_ids", []),
            prediction_ids=d.get("prediction_ids", []),
            objective_ids=d.get("objective_ids", []),
            current_strategy=ResearchStrategyType(d.get("current_strategy", "exploration")),
            confidence_timeline=timeline,
            journal=journal,
            cognitive_mode=CognitiveMode(d.get("cognitive_mode", "discovery")),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Discovery Workspace — runtime container
# ═══════════════════════════════════════════════════════════════════════════


class DiscoveryWorkspace:
    """Runtime container for managing research programs.

    The DiscoveryWorkspace is the operational layer that manages
    research program lifecycles, coordinates discovery activities,
    and maintains the epistemic state.

    It persists programs to SQLite for long-term durability (research
    may span months) while using the shared HCIR CognitiveGraph for
    all epistemic nodes.

    Architecture::

        DiscoveryWorkspace
            ├── programs: dict[str, ResearchProgram]
            ├── graph: CognitiveGraph (shared, not owned)
            └── persistence: SQLite (for program metadata only)

    The graph nodes (hypotheses, evidence, etc.) live in the shared
    HCIR graph and are discoverable by any subsystem.  The workspace
    only owns the program-level metadata and journal.
    """

    def __init__(
        self,
        data_dir: str | Path,
        graph: CognitiveGraph | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "discovery_workspace.db"
        self._graph = graph or CognitiveGraph()
        self._programs: dict[str, ResearchProgram] = {}
        self._init_db()
        self._load_programs()

    @property
    def graph(self) -> CognitiveGraph:
        """The shared cognitive graph."""
        return self._graph

    # ── Program Lifecycle ─────────────────────────────────────────────

    def create_program(
        self,
        title: str,
        research_question: str,
        description: str = "",
    ) -> ResearchProgram:
        """Create a new research program."""
        program = ResearchProgram(
            title=title,
            research_question=research_question,
            description=description,
        )
        self._programs[program.program_id] = program

        # Create the HCIR node representation
        node = ResearchProgramNode(
            id=program.program_id,
            title=title,
            research_question=research_question,
            description=description,
        )
        self._graph.upsert_node(node)

        # Journal entry
        program.journal.append(
            JournalEntry(
                event_type="program_created",
                description=f"Research program created: {title}",
            )
        )

        self._persist_program(program)
        logger.info("Created research program: %s (%s)", title, program.program_id)
        return program

    def get_program(self, program_id: str) -> ResearchProgram | None:
        """Retrieve a research program by ID."""
        return self._programs.get(program_id)

    def list_programs(self, status: str = "") -> list[ResearchProgram]:
        """List all programs, optionally filtered by status."""
        if status:
            return [p for p in self._programs.values() if p.status == status]
        return list(self._programs.values())

    def update_program_status(self, program_id: str, status: str) -> None:
        """Update a program's status."""
        program = self._programs.get(program_id)
        if program is None:
            return
        old_status = program.status
        program.status = status
        program.updated_at = time.time()
        program.journal.append(
            JournalEntry(
                event_type="status_changed",
                description=f"Status changed: {old_status} → {status}",
            )
        )
        self._persist_program(program)

    # ── Research Objectives ────────────────────────────────────────────

    def add_objective(
        self,
        program_id: str,
        objective: str,
        success_criteria: str = "",
        priority: float = 0.5,
    ) -> str:
        """Add a research objective to a program.

        Research programs decompose into objectives, which decompose
        into questions, which produce unknowns::

            Program → Objectives → Questions → Unknowns → Hypotheses

        Args:
            program_id: The research program.
            objective: The measurable objective statement.
            success_criteria: How to determine if objective is met.
            priority: Relative priority within program [0.0, 1.0].

        Returns:
            The ResearchObjectiveNode ID.
        """
        program = self._programs.get(program_id)
        if program is None:
            raise ValueError(f"Program not found: {program_id}")

        obj_node = ResearchObjectiveNode(
            objective=objective,
            program_id=program_id,
            success_criteria=success_criteria,
            priority=priority,
        )
        self._graph.upsert_node(obj_node)
        program.objective_ids.append(obj_node.id)
        program.updated_at = time.time()

        # Link program → objective in graph
        self._graph.add_edge(
            HCIREdge(
                sources=[program_id],
                targets=[obj_node.id],
                edge_type=HCIREdgeType.PART_OF,
            )
        )

        program.journal.append(
            JournalEntry(
                event_type="objective_added",
                description=f"Research objective added: {objective}",
                related_node_ids=[obj_node.id],
            )
        )

        self._persist_program(program)
        logger.info(
            "Added objective to program %s: %s",
            program_id,
            objective[:80],
        )
        return obj_node.id

    def add_question(
        self,
        program_id: str,
        objective_id: str,
        question: str,
        context: str = "",
        domain: str = "",
        importance: float = 0.5,
    ) -> str:
        """Add a research question under an objective.

        This is a convenience method that creates an UnknownNode
        linked to the specified objective.

        Args:
            program_id: The research program.
            objective_id: The parent research objective.
            question: The research question.
            context: What we know around this gap.
            domain: Knowledge domain.
            importance: How important is this question? [0.0, 1.0]

        Returns:
            The UnknownNode ID.
        """
        return self.add_unknown(
            program_id=program_id,
            question=question,
            context=context,
            domain=domain,
            importance=importance,
            objective_id=objective_id,
        )

    # ── Hypothesis Management ─────────────────────────────────────────

    def add_hypothesis(
        self,
        program_id: str,
        hypothesis: HypothesisNode,
    ) -> str:
        """Add a hypothesis to a research program."""
        program = self._programs.get(program_id)
        if program is None:
            raise ValueError(f"Program not found: {program_id}")

        hypothesis.research_program_id = program_id
        self._graph.upsert_node(hypothesis)
        program.hypothesis_ids.append(hypothesis.id)
        program.updated_at = time.time()

        # Journal entry
        program.journal.append(
            JournalEntry(
                event_type="hypothesis_generated",
                description=f"Hypothesis added: {hypothesis.claim}",
                related_node_ids=[hypothesis.id],
            )
        )

        self._persist_program(program)
        logger.info(
            "Added hypothesis to program %s: %s",
            program_id,
            hypothesis.claim[:80],
        )
        return hypothesis.id

    def update_hypothesis_lifecycle(
        self,
        program_id: str,
        hypothesis_id: str,
        new_lifecycle: HypothesisLifecycle,
        reason: str = "",
    ) -> None:
        """Update a hypothesis lifecycle state."""
        program = self._programs.get(program_id)
        if program is None:
            return

        node = self._graph.get_node(hypothesis_id)
        if not isinstance(node, HypothesisNode):
            return

        old_lifecycle = node.hypothesis_lifecycle
        node.hypothesis_lifecycle = new_lifecycle

        # Map lifecycle to falsification status
        if new_lifecycle == HypothesisLifecycle.FALSIFIED:
            node.falsification_status = FalsificationStatus.FALSIFIED
        elif new_lifecycle in (HypothesisLifecycle.SUPPORTED, HypothesisLifecycle.STRENGTHENED):
            node.falsification_status = FalsificationStatus.CORROBORATED
        elif new_lifecycle == HypothesisLifecycle.WEAKENED:
            node.falsification_status = FalsificationStatus.WEAKENED
        elif new_lifecycle == HypothesisLifecycle.SUPERSEDED:
            node.falsification_status = FalsificationStatus.SUPERSEDED

        self._graph.upsert_node(node)

        program.journal.append(
            JournalEntry(
                event_type="hypothesis_lifecycle_changed",
                description=f"Hypothesis lifecycle: {old_lifecycle} → {new_lifecycle}. {reason}",
                related_node_ids=[hypothesis_id],
            )
        )
        self._persist_program(program)

    # ── Unknown Management ────────────────────────────────────────────

    def add_unknown(
        self,
        program_id: str,
        question: str,
        context: str = "",
        domain: str = "",
        importance: float = 0.5,
        objective_id: str = "",
    ) -> str:
        """Register a knowledge gap in a research program.

        Args:
            program_id: The research program to add the unknown to.
            question: What don't we know?
            context: What do we know around this gap?
            domain: Knowledge domain.
            importance: How important is filling this gap? [0.0, 1.0]
            objective_id: Optional parent research objective.
        """
        program = self._programs.get(program_id)
        if program is None:
            raise ValueError(f"Program not found: {program_id}")

        unknown = UnknownNode(
            question=question,
            context=context,
            domain=domain,
            importance=importance,
            research_program_id=program_id,
            objective_id=objective_id,
        )
        self._graph.upsert_node(unknown)
        program.unknown_ids.append(unknown.id)
        program.updated_at = time.time()

        # Link to objective if specified
        if objective_id:
            obj_node = self._graph.get_node(objective_id)
            if isinstance(obj_node, ResearchObjectiveNode):
                obj_node.question_ids.append(unknown.id)
                self._graph.upsert_node(obj_node)

        program.journal.append(
            JournalEntry(
                event_type="unknown_registered",
                description=f"Knowledge gap identified: {question}",
                related_node_ids=[unknown.id],
            )
        )

        self._persist_program(program)
        logger.info("Registered unknown in program %s: %s", program_id, question[:80])
        return unknown.id

    def resolve_unknown(
        self,
        program_id: str,
        unknown_id: str,
        resolution: str = "",
        finding_id: str = "",
    ) -> None:
        """Mark a knowledge gap as resolved."""
        program = self._programs.get(program_id)
        if program is None:
            return

        program.journal.append(
            JournalEntry(
                event_type="unknown_resolved",
                description=f"Knowledge gap resolved: {resolution}",
                related_node_ids=[unknown_id, finding_id] if finding_id else [unknown_id],
            )
        )
        self._persist_program(program)

    # ── Evidence Management ───────────────────────────────────────────

    def add_evidence(
        self,
        program_id: str,
        evidence: EvidenceNode,
    ) -> str:
        """Add evidence to a research program."""
        program = self._programs.get(program_id)
        if program is None:
            raise ValueError(f"Program not found: {program_id}")

        self._graph.upsert_node(evidence)
        program.evidence_ids.append(evidence.id)
        program.updated_at = time.time()

        program.journal.append(
            JournalEntry(
                event_type="evidence_added",
                description=f"Evidence added (strength={evidence.strength:.2f})",
                related_node_ids=[evidence.id],
            )
        )

        self._persist_program(program)
        return evidence.id

    # ── Experiment Management ─────────────────────────────────────────

    def add_experiment(
        self,
        program_id: str,
        experiment: ExperimentNode,
    ) -> str:
        """Add an experiment to a research program."""
        program = self._programs.get(program_id)
        if program is None:
            raise ValueError(f"Program not found: {program_id}")

        experiment.research_program_id = program_id
        self._graph.upsert_node(experiment)
        program.experiment_ids.append(experiment.id)
        program.updated_at = time.time()

        program.journal.append(
            JournalEntry(
                event_type="experiment_designed",
                description=f"Experiment designed: {experiment.design[:80]}",
                related_node_ids=[experiment.id] + experiment.hypothesis_ids,
            )
        )

        self._persist_program(program)
        return experiment.id

    # ── Contradiction Management ──────────────────────────────────────

    def add_contradiction(
        self,
        program_id: str,
        contradiction: ContradictionNode,
    ) -> str:
        """Record a discovered contradiction."""
        program = self._programs.get(program_id)
        if program is None:
            raise ValueError(f"Program not found: {program_id}")

        self._graph.upsert_node(contradiction)
        program.contradiction_ids.append(contradiction.id)
        program.updated_at = time.time()

        program.journal.append(
            JournalEntry(
                event_type="contradiction_found",
                description=f"Contradiction discovered ({contradiction.contradiction_type})",
                related_node_ids=[
                    contradiction.id,
                    contradiction.claim_a_id,
                    contradiction.claim_b_id,
                ],
            )
        )

        self._persist_program(program)
        logger.info("Contradiction found in program %s", program_id)
        return contradiction.id

    # ── Prediction Management ─────────────────────────────────────────

    def add_prediction(
        self,
        program_id: str,
        prediction: PredictionNode,
    ) -> str:
        """Register a prediction for tracking."""
        program = self._programs.get(program_id)
        if program is None:
            raise ValueError(f"Program not found: {program_id}")

        self._graph.upsert_node(prediction)
        program.prediction_ids.append(prediction.id)
        program.updated_at = time.time()

        # Link prediction to hypothesis
        if prediction.hypothesis_id:
            hyp = self._graph.get_node(prediction.hypothesis_id)
            if isinstance(hyp, HypothesisNode):
                hyp.linked_predictions.append(prediction.id)
                self._graph.upsert_node(hyp)

            # Create PREDICTS edge
            edge = HCIREdge(
                edge_type=HCIREdgeType.PREDICTS,
                sources=[prediction.hypothesis_id],
                targets=[prediction.id],
            )
            try:
                self._graph.add_edge(edge)
            except ValueError:
                pass  # Edge already exists or dangling ref

        program.journal.append(
            JournalEntry(
                event_type="prediction_made",
                description=f"Prediction: {prediction.predicted_outcome[:80]}",
                related_node_ids=[prediction.id, prediction.hypothesis_id],
            )
        )

        self._persist_program(program)
        return prediction.id

    # ── Confidence Tracking ───────────────────────────────────────────

    def snapshot_confidence(
        self,
        program_id: str,
        reason: str = "",
    ) -> ConfidenceSnapshot | None:
        """Take a confidence snapshot across all hypotheses in a program."""
        program = self._programs.get(program_id)
        if program is None:
            return None

        hyp_confidences: dict[str, float] = {}
        for hid in program.hypothesis_ids:
            node = self._graph.get_node(hid)
            if isinstance(node, HypothesisNode):
                hyp_confidences[hid] = node.uncertainty.confidence

        overall = sum(hyp_confidences.values()) / len(hyp_confidences) if hyp_confidences else 0.0

        snapshot = ConfidenceSnapshot(
            overall_confidence=overall,
            hypothesis_confidences=hyp_confidences,
            reason=reason,
        )
        program.confidence_timeline.append(snapshot)
        program.updated_at = time.time()
        self._persist_program(program)
        return snapshot

    # ── Discovery View ────────────────────────────────────────────────

    def get_program_graph_view(self, program_id: str) -> list[Any]:
        """Get all HCIR nodes belonging to a research program."""
        program = self._programs.get(program_id)
        if program is None:
            return []

        all_ids = (
            program.hypothesis_ids
            + program.evidence_ids
            + program.experiment_ids
            + program.unknown_ids
            + program.finding_ids
            + program.contradiction_ids
            + program.prediction_ids
        )

        nodes = []
        for nid in all_ids:
            node = self._graph.get_node(nid)
            if node is not None:
                nodes.append(node)
        return nodes

    def generate_report(self, program_id: str) -> str:
        """Generate a human-readable research report."""
        program = self._programs.get(program_id)
        if program is None:
            return "Program not found."

        lines = [
            f"# Research Program: {program.title}",
            "",
            f"**Question:** {program.research_question}",
            f"**Status:** {program.status}",
            f"**Created:** {time.strftime('%Y-%m-%d', time.localtime(program.created_at))}",
            "",
            f"## Hypotheses ({len(program.hypothesis_ids)})",
        ]

        for hid in program.hypothesis_ids:
            node = self._graph.get_node(hid)
            if isinstance(node, HypothesisNode):
                lines.append(
                    f"- [{node.hypothesis_lifecycle.value}] {node.claim} "
                    f"(confidence={node.uncertainty.confidence:.2f})"
                )

        lines.append("")
        lines.append(f"## Unknowns ({len(program.unknown_ids)})")
        for uid in program.unknown_ids:
            node = self._graph.get_node(uid)
            if isinstance(node, UnknownNode):
                lines.append(f"- {node.question}")

        lines.append("")
        lines.append(f"## Evidence ({len(program.evidence_ids)})")
        lines.append(f"## Experiments ({len(program.experiment_ids)})")
        lines.append(f"## Contradictions ({len(program.contradiction_ids)})")

        lines.append("")
        lines.append("## Journal (last 10 entries)")
        for entry in program.journal[-10:]:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry.timestamp))
            lines.append(f"- [{ts}] {entry.event_type}: {entry.description}")

        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_programs (
                    program_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rp_status
                ON research_programs(json_extract(data, '$.status'))
            """)

    def _persist_program(self, program: ResearchProgram) -> None:
        data = json.dumps(program.to_dict())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO research_programs
                   (program_id, data, created_at, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (program.program_id, data, program.created_at, program.updated_at),
            )

    def _load_programs(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute("SELECT data FROM research_programs").fetchall()
            for (data_json,) in rows:
                d = json.loads(data_json)
                program = ResearchProgram.from_dict(d)
                self._programs[program.program_id] = program
            if self._programs:
                logger.info("Loaded %d research programs", len(self._programs))
        except Exception as e:
            logger.warning("Failed to load research programs: %s", e)
