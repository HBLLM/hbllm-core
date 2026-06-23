"""
ProjectGraph — persistent, graph-based project cognition.

Tracks ongoing projects across conversations. Projects, goals, milestones,
questions, and blockers are nodes in a directed graph with typed edges.

Graph structure:
    HBLLM (project)
     ├─ has_goal ──→ "Reduce LLM dependency" (goal)
     ├─ has_goal ──→ "Build architecture" (goal)
     │                  ├─ has_milestone ──→ "Curiosity system" (milestone, completed)
     │                  ├─ has_question ──→ "Human model design?" (question)
     │                  └─ blocked_by ────→ "UserModel design" (blocker)
     ├─ has_decision ─→ "Use KG as canonical store" (decision)
     └─ depends_on ──→ "UserModel" (project)

Auto-detection:
    On every query, matches text against project tags/topics/files.
    If confidence > 0.6, auto-associates conversation with project.

Reactivation:
    Generates a context summary via graph traversal for LLM injection.
    "What are we working on? What's blocked? What's next?"

Bus Topics:
    project.reactivated   → Context summary for active project
    project.milestone     → Milestone status change
    project.detected      → Auto-detection matched a project
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Entity Types ─────────────────────────────────────────────────────────────

ENTITY_TYPES = {"project", "goal", "milestone", "question", "decision", "blocker"}

RELATION_TYPES = {
    "has_goal",
    "has_milestone",
    "has_question",
    "has_decision",
    "has_blocker",
    "blocked_by",
    "depends_on",
    "resolved_by",
    "supersedes",
}

STATUS_VALUES = {"active", "completed", "blocked", "pending", "resolved", "paused", "archived"}


# ── Data Models ──────────────────────────────────────────────────────────────


@dataclass
class ProjectEntity:
    """A node in the project graph."""

    entity_id: str
    entity_type: str  # "project" | "goal" | "milestone" | "question" | "decision" | "blocker"
    name: str
    description: str = ""
    status: str = "active"
    tenant_id: str = "default"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


@dataclass
class ProjectRelation:
    """An edge in the project graph."""

    source_id: str
    target_id: str
    relation_type: str  # "has_goal" | "blocked_by" | "depends_on" etc.
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


# ── Project Context (for reactivation) ───────────────────────────────────────


@dataclass
class ProjectContext:
    """Contextual metadata for project association and reactivation."""

    project_id: str
    tags: list[str] = field(default_factory=list)
    last_topics: list[str] = field(default_factory=list)
    last_files_touched: list[str] = field(default_factory=list)
    last_decisions: list[str] = field(default_factory=list)
    last_active: float = field(default_factory=time.time)
    conversation_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "tags": self.tags,
            "last_topics": self.last_topics,
            "last_files_touched": self.last_files_touched,
            "last_decisions": self.last_decisions,
            "last_active": self.last_active,
            "conversation_count": self.conversation_count,
        }


# ── ProjectGraph Engine ──────────────────────────────────────────────────────


class ProjectGraph:
    """Graph-based persistent project state engine.

    All entities (projects, goals, milestones, questions, blockers)
    are nodes. All relationships (has_goal, blocked_by, depends_on)
    are directed edges. Enables powerful graph traversal for
    dependency tracking and context reactivation.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self._db_path = Path(data_dir) / "project_graph.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    status TEXT DEFAULT 'active',
                    tenant_id TEXT DEFAULT 'default',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    PRIMARY KEY (source_id, target_id, relation_type)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS project_context (
                    project_id TEXT PRIMARY KEY,
                    tags TEXT DEFAULT '[]',
                    last_topics TEXT DEFAULT '[]',
                    last_files_touched TEXT DEFAULT '[]',
                    last_decisions TEXT DEFAULT '[]',
                    last_active REAL NOT NULL,
                    conversation_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_entity_tenant ON entities(tenant_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON relations(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON relations(target_id)")

    # ── Entity CRUD ──────────────────────────────────────────────────

    def create_project(
        self,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
        tenant_id: str = "default",
    ) -> ProjectEntity:
        """Create a new project node."""
        entity_id = f"proj_{int(time.time())}_{hash(name) % 10000}"
        entity = ProjectEntity(
            entity_id=entity_id,
            entity_type="project",
            name=name,
            description=description,
            tenant_id=tenant_id,
        )
        self._save_entity(entity)

        # Create context record
        ctx = ProjectContext(
            project_id=entity_id,
            tags=tags or [],
        )
        self._save_context(ctx)

        logger.info("Created project: %s (%s)", name, entity_id)
        return entity

    def add_entity(
        self,
        project_id: str,
        entity_type: str,
        name: str,
        description: str = "",
        tenant_id: str = "default",
    ) -> ProjectEntity:
        """Add a goal, milestone, question, decision, or blocker to a project.

        Automatically creates the appropriate relation edge.
        """
        entity_id = f"{entity_type[:4]}_{int(time.time())}_{hash(name) % 10000}"
        entity = ProjectEntity(
            entity_id=entity_id,
            entity_type=entity_type,
            name=name,
            description=description,
            tenant_id=tenant_id,
        )
        self._save_entity(entity)

        # Auto-create relation to parent project
        type_to_relation = {
            "goal": "has_goal",
            "milestone": "has_milestone",
            "question": "has_question",
            "decision": "has_decision",
            "blocker": "has_blocker",
        }
        rel_type = type_to_relation.get(entity_type, f"has_{entity_type}")
        self.add_relation(project_id, entity_id, rel_type)

        logger.info("Added %s to project %s: %s", entity_type, project_id, name)
        return entity

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        metadata: dict[str, Any] | None = None,
    ) -> ProjectRelation:
        """Add a directed edge between two entities."""
        rel = ProjectRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            metadata=metadata or {},
        )
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO relations "
                "(source_id, target_id, relation_type, created_at, metadata) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    rel.source_id, rel.target_id, rel.relation_type,
                    rel.created_at, json.dumps(rel.metadata),
                ),
            )
        return rel

    def get_entity(self, entity_id: str) -> ProjectEntity | None:
        """Retrieve a single entity by ID."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM entities WHERE entity_id = ?", (entity_id,)
            ).fetchone()
        if row:
            return self._row_to_entity(row)
        return None

    def update_status(self, entity_id: str, status: str) -> bool:
        """Update entity status."""
        now = time.time()
        with sqlite3.connect(str(self._db_path)) as conn:
            cursor = conn.execute(
                "UPDATE entities SET status = ?, updated_at = ? WHERE entity_id = ?",
                (status, now, entity_id),
            )
            return cursor.rowcount > 0

    def resolve(
        self, entity_id: str, resolution: str = "", resolved_by: str = ""
    ) -> bool:
        """Mark a question or blocker as resolved."""
        entity = self.get_entity(entity_id)
        if not entity:
            return False

        now = time.time()
        with sqlite3.connect(str(self._db_path)) as conn:
            meta = entity.metadata
            meta["resolution"] = resolution
            meta["resolved_at"] = now
            conn.execute(
                "UPDATE entities SET status = 'resolved', updated_at = ?, metadata = ? "
                "WHERE entity_id = ?",
                (now, json.dumps(meta), entity_id),
            )

        if resolved_by:
            self.add_relation(entity_id, resolved_by, "resolved_by")

        logger.info("Resolved %s: %s", entity.entity_type, entity.name)
        return True

    # ── Graph Traversal ──────────────────────────────────────────────

    def get_children(
        self,
        entity_id: str,
        relation_type: str | None = None,
        entity_type: str | None = None,
    ) -> list[ProjectEntity]:
        """Get entities connected from source via outgoing edges."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            if relation_type:
                rel_rows = conn.execute(
                    "SELECT target_id FROM relations WHERE source_id = ? AND relation_type = ?",
                    (entity_id, relation_type),
                ).fetchall()
            else:
                rel_rows = conn.execute(
                    "SELECT target_id FROM relations WHERE source_id = ?",
                    (entity_id,),
                ).fetchall()

            target_ids = [r["target_id"] for r in rel_rows]
            if not target_ids:
                return []

            placeholders = ",".join("?" * len(target_ids))
            query = f"SELECT * FROM entities WHERE entity_id IN ({placeholders})"
            if entity_type:
                query += " AND entity_type = ?"
                target_ids.append(entity_type)

            rows = conn.execute(query, target_ids).fetchall()
            return [self._row_to_entity(r) for r in rows]

    def get_blockers(self, project_id: str) -> list[ProjectEntity]:
        """Get all active blockers for a project (transitive)."""
        blockers: list[ProjectEntity] = []
        # Direct blockers on the project
        direct = self.get_children(project_id, relation_type="has_blocker")
        blockers.extend(e for e in direct if e.status != "resolved")

        # Also check blocked_by on goals
        goals = self.get_children(project_id, relation_type="has_goal")
        for goal in goals:
            goal_blockers = self.get_children(goal.entity_id, relation_type="blocked_by")
            blockers.extend(e for e in goal_blockers if e.status != "resolved")

        return blockers

    def get_dependencies(self, project_id: str) -> list[ProjectEntity]:
        """Get projects that this project depends on."""
        return self.get_children(project_id, relation_type="depends_on")

    def get_open_questions(self, project_id: str) -> list[ProjectEntity]:
        """Get unresolved questions for a project."""
        questions = self.get_children(project_id, relation_type="has_question")
        return [q for q in questions if q.status != "resolved"]

    def get_active_goals(self, project_id: str) -> list[ProjectEntity]:
        """Get active goals for a project."""
        goals = self.get_children(project_id, relation_type="has_goal")
        return [g for g in goals if g.status in ("active", "blocked")]

    def get_milestones(self, project_id: str) -> list[ProjectEntity]:
        """Get all milestones for a project, across all goals."""
        milestones: list[ProjectEntity] = []
        # Direct milestones
        milestones.extend(self.get_children(project_id, relation_type="has_milestone"))
        # Milestones under goals
        for goal in self.get_children(project_id, relation_type="has_goal"):
            milestones.extend(self.get_children(goal.entity_id, relation_type="has_milestone"))
        return milestones

    # ── Project Detection & Association ──────────────────────────────

    def get_active_projects(self, tenant_id: str = "default") -> list[ProjectEntity]:
        """Get all active projects for a tenant."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM entities WHERE entity_type = 'project' "
                "AND status = 'active' AND tenant_id = ? "
                "ORDER BY updated_at DESC",
                (tenant_id,),
            ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def auto_detect_project(
        self,
        query: str,
        files: list[str] | None = None,
        tenant_id: str = "default",
    ) -> ProjectEntity | None:
        """Try to match a query to an existing project.

        Matching strategy:
        1. Keyword overlap with project tags + last_topics
        2. File path overlap with last_files_touched
        3. Name/description substring match

        Returns the best match if confidence > threshold.
        """
        projects = self.get_active_projects(tenant_id)
        if not projects:
            return None

        query_words = set(query.lower().split())
        files = files or []
        best_score = 0.0
        best_project = None

        for proj in projects:
            score = 0.0
            ctx = self._load_context(proj.entity_id)

            # Tag overlap
            if ctx and ctx.tags:
                tag_words = set(t.lower() for t in ctx.tags)
                tag_overlap = len(query_words & tag_words)
                score += tag_overlap * 0.3

            # Topic overlap
            if ctx and ctx.last_topics:
                topic_words = set()
                for topic in ctx.last_topics:
                    topic_words.update(topic.lower().split())
                topic_overlap = len(query_words & topic_words)
                score += topic_overlap * 0.2

            # File overlap
            if ctx and ctx.last_files_touched and files:
                file_overlap = len(set(files) & set(ctx.last_files_touched))
                score += file_overlap * 0.4

            # Name/description match
            name_words = set(proj.name.lower().split())
            name_overlap = len(query_words & name_words)
            score += name_overlap * 0.5

            if score > best_score:
                best_score = score
                best_project = proj

        if best_score >= 0.6 and best_project:
            return best_project
        return None

    def associate_conversation(
        self,
        project_id: str,
        topic: str = "",
        files: list[str] | None = None,
    ) -> None:
        """Link current conversation to a project."""
        ctx = self._load_context(project_id)
        if not ctx:
            ctx = ProjectContext(project_id=project_id)

        ctx.last_active = time.time()
        ctx.conversation_count += 1

        if topic:
            ctx.last_topics.append(topic)
            ctx.last_topics = ctx.last_topics[-20:]  # Keep last 20

        if files:
            ctx.last_files_touched.extend(files)
            ctx.last_files_touched = list(set(ctx.last_files_touched))[-50:]

        self._save_context(ctx)

        # Update entity timestamp
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "UPDATE entities SET updated_at = ? WHERE entity_id = ?",
                (time.time(), project_id),
            )

    def record_decision(self, project_id: str, decision: str) -> ProjectEntity:
        """Record a key decision made in the project."""
        entity = self.add_entity(project_id, "decision", decision)

        ctx = self._load_context(project_id)
        if ctx:
            ctx.last_decisions.append(decision)
            ctx.last_decisions = ctx.last_decisions[-20:]
            self._save_context(ctx)

        return entity

    # ── Reactivation (Context Generation) ────────────────────────────

    def reactivate(self, project_id: str) -> str:
        """Generate a project context summary for prompt injection.

        Traverses the project graph and produces a structured NL summary.
        """
        proj = self.get_entity(project_id)
        if not proj:
            return ""

        ctx = self._load_context(project_id)
        parts: list[str] = []

        # Header
        time_ago = self._time_ago(ctx.last_active if ctx else proj.updated_at)
        conv_count = ctx.conversation_count if ctx else 0
        parts.append(f"**{proj.name}** ({proj.status}) — last active {time_ago}, {conv_count} conversations")

        if proj.description:
            parts.append(proj.description)

        # Goals
        goals = self.get_active_goals(project_id)
        if goals:
            goal_lines = []
            for g in goals:
                status_icon = {"active": "🔄", "blocked": "🚫", "completed": "✅"}.get(
                    g.status, "⏳"
                )
                # Check milestones under this goal
                milestones = self.get_children(g.entity_id, relation_type="has_milestone")
                done = sum(1 for m in milestones if m.status == "completed")
                total = len(milestones)
                progress = f" ({done}/{total} milestones)" if total else ""
                goal_lines.append(f"  - {status_icon} {g.name}{progress}")
            parts.append("Goals:\n" + "\n".join(goal_lines))

        # Open questions
        questions = self.get_open_questions(project_id)
        if questions:
            q_lines = [f"  - {q.name}" for q in questions[:5]]
            parts.append("Open Questions:\n" + "\n".join(q_lines))

        # Blockers
        blockers = self.get_blockers(project_id)
        if blockers:
            b_lines = [f"  - {b.name}" for b in blockers]
            parts.append("Blocked By:\n" + "\n".join(b_lines))

        # Recent decisions
        if ctx and ctx.last_decisions:
            d_lines = [f"  - {d}" for d in ctx.last_decisions[-3:]]
            parts.append("Recent Decisions:\n" + "\n".join(d_lines))

        # Dependencies
        deps = self.get_dependencies(project_id)
        if deps:
            dep_lines = [f"  - {d.name} ({d.status})" for d in deps]
            parts.append("Depends On:\n" + "\n".join(dep_lines))

        return "\n\n".join(parts)

    async def get_context(
        self, query: str, tenant_id: str, budget: int
    ) -> str:
        """ContextFusion-compatible provider.

        Auto-detects the relevant project and returns its reactivation summary.
        """
        project = self.auto_detect_project(query, tenant_id=tenant_id)
        if project:
            return self.reactivate(project.entity_id)
        return ""

    # ── Persistence Helpers ──────────────────────────────────────────

    def _save_entity(self, entity: ProjectEntity) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO entities "
                "(entity_id, entity_type, name, description, status, tenant_id, "
                "created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entity.entity_id, entity.entity_type, entity.name,
                    entity.description, entity.status, entity.tenant_id,
                    entity.created_at, entity.updated_at, json.dumps(entity.metadata),
                ),
            )

    def _save_context(self, ctx: ProjectContext) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO project_context "
                "(project_id, tags, last_topics, last_files_touched, last_decisions, "
                "last_active, conversation_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    ctx.project_id,
                    json.dumps(ctx.tags),
                    json.dumps(ctx.last_topics),
                    json.dumps(ctx.last_files_touched),
                    json.dumps(ctx.last_decisions),
                    ctx.last_active,
                    ctx.conversation_count,
                ),
            )

    def _load_context(self, project_id: str) -> ProjectContext | None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM project_context WHERE project_id = ?",
                (project_id,),
            ).fetchone()
        if not row:
            return None
        return ProjectContext(
            project_id=row["project_id"],
            tags=json.loads(row["tags"]),
            last_topics=json.loads(row["last_topics"]),
            last_files_touched=json.loads(row["last_files_touched"]),
            last_decisions=json.loads(row["last_decisions"]),
            last_active=row["last_active"],
            conversation_count=row["conversation_count"],
        )

    def _row_to_entity(self, row: sqlite3.Row) -> ProjectEntity:
        return ProjectEntity(
            entity_id=row["entity_id"],
            entity_type=row["entity_type"],
            name=row["name"],
            description=row["description"],
            status=row["status"],
            tenant_id=row["tenant_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"]),
        )

    @staticmethod
    def _time_ago(ts: float) -> str:
        """Human-readable relative time."""
        delta = time.time() - ts
        if delta < 60:
            return "just now"
        if delta < 3600:
            return f"{int(delta / 60)}m ago"
        if delta < 86400:
            return f"{int(delta / 3600)}h ago"
        return f"{int(delta / 86400)}d ago"

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Summary statistics."""
        with sqlite3.connect(str(self._db_path)) as conn:
            total_entities = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            total_relations = conn.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
            by_type = conn.execute(
                "SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type"
            ).fetchall()
            active_projects = conn.execute(
                "SELECT COUNT(*) FROM entities WHERE entity_type='project' AND status='active'"
            ).fetchone()[0]
        return {
            "total_entities": total_entities,
            "total_relations": total_relations,
            "active_projects": active_projects,
            "by_type": {t: c for t, c in by_type},
        }
