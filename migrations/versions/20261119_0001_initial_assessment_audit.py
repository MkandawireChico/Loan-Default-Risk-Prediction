"""Create the metadata-only assessment audit table."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20261119_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "assessment_audit",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("model_version", sa.String(length=80), nullable=False),
        sa.Column("probability_of_default", sa.Float(), nullable=False),
        sa.Column("risk_band", sa.String(length=20), nullable=False),
        sa.Column("review_flag", sa.Boolean(), nullable=False),
        sa.Column("source", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_assessment_audit_model_version", "assessment_audit", ["model_version"])
    op.create_index("ix_assessment_audit_risk_band", "assessment_audit", ["risk_band"])
    op.create_index("ix_assessment_audit_created_at", "assessment_audit", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_assessment_audit_created_at", table_name="assessment_audit")
    op.drop_index("ix_assessment_audit_risk_band", table_name="assessment_audit")
    op.drop_index("ix_assessment_audit_model_version", table_name="assessment_audit")
    op.drop_table("assessment_audit")
