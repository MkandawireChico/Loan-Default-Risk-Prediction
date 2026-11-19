"""Add versioned model registry metadata."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20261119_0002"
down_revision = "20261119_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_registry",
        sa.Column("model_version", sa.String(length=80), nullable=False),
        sa.Column("model_type", sa.String(length=120), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("artifact_path", sa.String(length=255), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("dataset_sha256", sa.String(length=64), nullable=False),
        sa.Column("training_rows", sa.Integer(), nullable=True),
        sa.Column("training_seed", sa.Integer(), nullable=True),
        sa.Column("decision_threshold", sa.Float(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=False),
        sa.Column("evaluation_json", sa.Text(), nullable=False),
        sa.Column("registered_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("model_version"),
    )
    op.create_index("ix_model_registry_status", "model_registry", ["status"])
    op.create_index("ix_model_registry_registered_at", "model_registry", ["registered_at"])


def downgrade() -> None:
    op.drop_index("ix_model_registry_registered_at", table_name="model_registry")
    op.drop_index("ix_model_registry_status", table_name="model_registry")
    op.drop_table("model_registry")
