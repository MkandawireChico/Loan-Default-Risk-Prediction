from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


def test_initial_audit_migration_is_reversible(tmp_path, monkeypatch):
    database_path = tmp_path / "migration-test.sqlite"
    database_url = f"sqlite:///{database_path}"
    monkeypatch.setenv("DATABASE_URL", database_url)

    config = Config("alembic.ini")
    command.upgrade(config, "head")

    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        assert inspector.has_table("alembic_version")
        assert inspector.has_table("assessment_audit")
        columns = {column["name"] for column in inspector.get_columns("assessment_audit")}
        assert columns == {
            "id",
            "model_version",
            "probability_of_default",
            "risk_band",
            "review_flag",
            "source",
            "created_at",
        }
        indexes = {index["name"] for index in inspector.get_indexes("assessment_audit")}
        assert {
            "ix_assessment_audit_model_version",
            "ix_assessment_audit_risk_band",
            "ix_assessment_audit_created_at",
        }.issubset(indexes)
    finally:
        engine.dispose()

    command.downgrade(config, "base")

    engine = create_engine(database_url)
    try:
        inspector = inspect(engine)
        assert not inspector.has_table("assessment_audit")
        assert not inspector.has_table("alembic_version")
    finally:
        engine.dispose()
