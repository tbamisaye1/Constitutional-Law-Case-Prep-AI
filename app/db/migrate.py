"""
Forward-only SQL migrations.

Run them yourself rather than on application start:

    python -m app.db.migrate            # apply anything pending
    python -m app.db.migrate --status   # list applied and pending, change nothing

Applying on start looked tempting, but on Vercel every cold start would race
every other cold start for the same DDL, and a failed migration would take the
whole API down instead of one deploy step. Keeping it a command means a broken
migration is visible before traffic sees it.

There is no down-migration on purpose. Reverting schema on a database that
holds the only copy of someone's case notes is more dangerous than writing a
new forward migration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.db.connection import DatabaseNotConfigured, db_connection

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"

# Any 64-bit constant works; it only has to be the same in every process that
# migrates this database. Two concurrent runners then queue instead of both
# issuing the same CREATE TABLE.
_MIGRATION_LOCK_KEY = 8_531_204_770_113_499

_TRACKING_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def migration_files() -> list[Path]:
    """Migration files in lexical order, which is also apply order."""
    if not MIGRATIONS_DIR.is_dir():
        return []
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


def _applied_filenames(cursor) -> set[str]:
    cursor.execute("SELECT filename FROM schema_migrations")
    return {row["filename"] for row in cursor.fetchall()}


def migration_status() -> dict:
    """
    Report which migrations are applied and which are pending.

    Returns:
        dict with 'applied' and 'pending' filename lists.

    Raises:
        DatabaseNotConfigured: when DATABASE_URL is empty.
    """
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(_TRACKING_TABLE_SQL)
            applied = _applied_filenames(cursor)

    available = [path.name for path in migration_files()]
    return {
        "applied": [name for name in available if name in applied],
        "pending": [name for name in available if name not in applied],
    }


def apply_pending() -> list[str]:
    """
    Apply every migration that has not run yet.

    Each migration runs in the same transaction as the row that records it, so
    a failure halfway through leaves neither the DDL nor the bookkeeping behind.

    Returns:
        Filenames applied, in order. Empty when the schema was already current.
    """
    applied_now: list[str] = []

    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(_TRACKING_TABLE_SQL)
            cursor.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_KEY,))
            already_applied = _applied_filenames(cursor)

            for path in migration_files():
                if path.name in already_applied:
                    continue
                cursor.execute(path.read_text(encoding="utf-8"))
                cursor.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)",
                    (path.name,),
                )
                applied_now.append(path.name)

    return applied_now


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply Neon Postgres migrations.")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show applied and pending migrations without changing anything.",
    )
    args = parser.parse_args()

    try:
        if args.status:
            status = migration_status()
            print("applied:", ", ".join(status["applied"]) or "(none)")
            print("pending:", ", ".join(status["pending"]) or "(none)")
            return 0

        applied = apply_pending()
    except DatabaseNotConfigured as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if applied:
        for name in applied:
            print(f"applied {name}")
    else:
        print("schema already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
