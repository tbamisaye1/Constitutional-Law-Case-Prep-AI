"""Migration runner behaviour."""

from __future__ import annotations

from tests.conftest import requires_database


@requires_database
def test_apply_pending_is_idempotent(migrated_database):
    """
    A second run must be a no-op.

    This is the property that lets the command be safe to run on every deploy
    without anyone checking first.
    """
    from app.db.migrate import apply_pending

    assert apply_pending() == []


@requires_database
def test_status_reports_nothing_pending_after_apply(migrated_database):
    from app.db.migrate import migration_status

    status = migration_status()

    assert status["pending"] == []
    assert "001_workspace_persistence.sql" in status["applied"]


@requires_database
def test_every_expected_table_exists(cursor):
    """Guards against a migration that parses but forgets a table."""
    cursor.execute(
        """
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        """
    )
    tables = {row["table_name"] for row in cursor.fetchall()}

    expected = {
        "workspaces",
        "matters",
        "cases",
        "documents",
        "annotations",
        "notes",
        "library_records",
        "schema_migrations",
    }
    assert expected <= tables


def test_migration_files_are_ordered_and_numbered():
    """
    Filenames must sort into apply order.

    Runs without a database because it only reads the directory. A file named
    without a numeric prefix would still sort, but not predictably, so catch it
    here rather than after it has been applied somewhere.
    """
    from app.db.migrate import migration_files

    names = [path.name for path in migration_files()]

    assert names, "no migration files found"
    assert names == sorted(names)
    for name in names:
        prefix = name.split("_", 1)[0]
        assert prefix.isdigit(), f"{name} does not start with a number"
