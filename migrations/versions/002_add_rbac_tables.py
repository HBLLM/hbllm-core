"""Add RBAC tables.

Revision ID: 002
Revises: 001
Create Date: 2026-06-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "role_assignments",
        sa.Column("tenant_id", sa.Text, nullable=False),
        sa.Column("user_id", sa.Text, nullable=False),
        sa.Column("role", sa.Text, nullable=False, server_default="member"),
        sa.Column("assigned_by", sa.Text, server_default="system"),
        sa.Column("assigned_at", sa.Float, nullable=False),
        sa.PrimaryKeyConstraint("tenant_id", "user_id"),
    )
    op.create_index("idx_rbac_tenant", "role_assignments", ["tenant_id"])


def downgrade() -> None:
    op.drop_table("role_assignments")
