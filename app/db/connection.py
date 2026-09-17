"""
Connections to Neon Postgres.

We deliberately do not keep a process-wide connection pool. Every request on
Vercel may land on a different function instance, and an instance can be frozen
between requests, which leaves pooled sockets half-dead. Opening one short
connection per request and letting Neon's PgBouncer absorb the churn is both
simpler and more reliable here. If this ever runs on a long-lived host where
connection setup cost matters, swap this module for psycopg_pool and keep the
same `db_connection()` signature.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings


class DatabaseNotConfigured(RuntimeError):
    """Raised when DATABASE_URL is missing, so callers can return a clear 503."""


def _conninfo() -> str:
    settings = get_settings()
    url = settings.database_url.strip()
    if not url:
        raise DatabaseNotConfigured(
            "DATABASE_URL is not set. Add the Neon integration to this Vercel "
            "project, or run `vercel env pull .env.local` for local development."
        )
    # Neon requires TLS. The integration's URL normally carries sslmode already,
    # but a hand-pasted URL often does not and then psycopg silently tries plain
    # TCP, which Neon closes with a confusing error.
    if "sslmode=" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}sslmode=require"
    return url


@contextmanager
def db_connection() -> Iterator[psycopg.Connection]:
    """
    Yield a dict-row connection inside a transaction.

    The transaction commits when the block exits cleanly and rolls back if the
    block raises, which is psycopg's own context-manager behaviour.

    Raises:
        DatabaseNotConfigured: when DATABASE_URL is empty.
        psycopg.OperationalError: when Neon is unreachable or rejects the login.
    """
    settings = get_settings()
    connection = psycopg.connect(
        _conninfo(),
        connect_timeout=settings.database_connect_timeout,
        row_factory=dict_row,
        # PgBouncer in transaction-pooling mode hands each transaction a
        # different server connection, so a prepared statement created on one
        # may not exist on the next. Disabling psycopg's automatic prepare keeps
        # us safe on Neon's pooled endpoint.
        prepare_threshold=None,
    )
    with connection:
        yield connection
