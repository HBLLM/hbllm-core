"""Initial schema — captures all existing tables.

Revision ID: 001
Revises: None
Create Date: 2026-06-14
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── pgvector extension ──────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── Episodic Memory ─────────────────────────────────────────────
    op.create_table(
        "episodic_memories",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("tenant_id", sa.Text, nullable=False, index=True),
        sa.Column("timestamp", sa.Float, nullable=False),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("response", sa.Text, nullable=False),
        sa.Column("emotion", sa.Text, server_default="neutral"),
        sa.Column("importance", sa.Float, server_default="0.5"),
        sa.Column("metadata", sa.JSON, server_default="{}"),
    )
    op.create_index("idx_episodic_tenant_time", "episodic_memories",
                     ["tenant_id", "timestamp"])

    # ── Semantic Memory ─────────────────────────────────────────────
    op.create_table(
        "semantic_facts",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("tenant_id", sa.Text, nullable=False, index=True),
        sa.Column("fact", sa.Text, nullable=False),
        sa.Column("source", sa.Text, server_default=""),
        sa.Column("confidence", sa.Float, server_default="1.0"),
        sa.Column("created_at", sa.Float, nullable=False),
        sa.Column("embedding", sa.LargeBinary, nullable=True),
    )

    # ── Audit Log ───────────────────────────────────────────────────
    op.create_table(
        "audit_log",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("timestamp", sa.Float, nullable=False),
        sa.Column("tenant_id", sa.Text, nullable=False),
        sa.Column("user_id", sa.Text, server_default=""),
        sa.Column("device_id", sa.Text, server_default=""),
        sa.Column("actor", sa.Text, nullable=False, server_default="system"),
        sa.Column("action", sa.Text, nullable=False),
        sa.Column("resource", sa.Text, server_default=""),
        sa.Column("ip_address", sa.Text, server_default=""),
        sa.Column("user_agent", sa.Text, server_default=""),
        sa.Column("severity", sa.Text, server_default="info"),
        sa.Column("details", sa.JSON, server_default="{}"),
        sa.Column("success", sa.Boolean, server_default="true"),
    )
    op.create_index("idx_audit_tenant", "audit_log",
                     ["tenant_id", sa.text("timestamp DESC")])
    op.create_index("idx_audit_action", "audit_log",
                     ["action", sa.text("timestamp DESC")])
    op.create_index("idx_audit_severity", "audit_log",
                     ["severity", sa.text("timestamp DESC")])

    # ── Tenant Registry ─────────────────────────────────────────────
    op.create_table(
        "tenants",
        sa.Column("tenant_id", sa.Text, primary_key=True),
        sa.Column("display_name", sa.Text, server_default=""),
        sa.Column("plan", sa.Text, server_default="free"),
        sa.Column("max_conversations", sa.Integer, server_default="100"),
        sa.Column("max_turns_per_convo", sa.Integer, server_default="200"),
        sa.Column("created_at", sa.Float, nullable=False),
        sa.Column("is_active", sa.Boolean, server_default="true"),
    )

    # ── Conversations ───────────────────────────────────────────────
    op.create_table(
        "conversations",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("tenant_id", sa.Text, nullable=False, index=True),
        sa.Column("title", sa.Text, server_default=""),
        sa.Column("created_at", sa.Float, nullable=False),
        sa.Column("updated_at", sa.Float, nullable=False),
        sa.Column("turn_count", sa.Integer, server_default="0"),
        sa.Column("metadata", sa.JSON, server_default="{}"),
    )

    # ── Conversation Turns ──────────────────────────────────────────
    op.create_table(
        "conversation_turns",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("conversation_id", sa.Text, sa.ForeignKey("conversations.id"),
                   nullable=False, index=True),
        sa.Column("tenant_id", sa.Text, nullable=False, index=True),
        sa.Column("role", sa.Text, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("timestamp", sa.Float, nullable=False),
        sa.Column("metadata", sa.JSON, server_default="{}"),
    )

    # ── Tool Memory ─────────────────────────────────────────────────
    op.create_table(
        "tool_executions",
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("tenant_id", sa.Text, nullable=False, index=True),
        sa.Column("tool_name", sa.Text, nullable=False),
        sa.Column("input_hash", sa.Text, nullable=False),
        sa.Column("result", sa.JSON),
        sa.Column("success", sa.Boolean, server_default="true"),
        sa.Column("duration_ms", sa.Float),
        sa.Column("created_at", sa.Float, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("tool_executions")
    op.drop_table("conversation_turns")
    op.drop_table("conversations")
    op.drop_table("tenants")
    op.drop_table("audit_log")
    op.drop_table("semantic_facts")
    op.drop_table("episodic_memories")
    op.execute("DROP EXTENSION IF EXISTS vector")
