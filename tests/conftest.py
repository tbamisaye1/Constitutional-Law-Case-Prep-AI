"""
Test fixtures for the persistence layer.

These tests talk to a real Postgres rather than a mocked cursor, because what
is worth testing here is the SQL itself: the last-write-wins guard, the
tombstones, and the workspace isolation all live in ON CONFLICT clauses that a
mock would happily accept while being wrong.

Point TEST_DATABASE_URL at a scratch database and the tests run. Leave it
unset and they skip, so the suite stays runnable without a database.

    createdb case_law_test
    export TEST_DATABASE_URL='postgresql://localhost/case_law_test?sslmode=disable'
    pytest

Every test truncates the tables it touches, and each one works in its own
randomly generated workspace, so a leftover row cannot make a later test pass.
"""

from __future__ import annotations

import os
import uuid

import pytest

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "").strip()

requires_database = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason="Set TEST_DATABASE_URL to a scratch Postgres database to run these.",
)


@pytest.fixture(scope="session", autouse=True)
def _point_settings_at_test_database():
    """
    Make the app read the scratch database instead of the real one.

    Settings are cached with lru_cache, so the environment has to be set and
    the cache cleared before anything calls get_settings().
    """
    if not TEST_DATABASE_URL:
        yield
        return

    from app.config import get_settings

    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    get_settings.cache_clear()

    yield

    if previous is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = previous
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def migrated_database(_point_settings_at_test_database):
    """Apply migrations once for the whole session."""
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL not set")

    from app.db.migrate import apply_pending

    apply_pending()
    return TEST_DATABASE_URL


@pytest.fixture
def workspace_id() -> str:
    """A fresh workspace per test, so tests cannot see each other's rows."""
    return str(uuid.uuid4())


@pytest.fixture
def cursor(migrated_database):
    """
    An open cursor that rolls back when the test finishes.

    Rolling back rather than truncating keeps the tests independent without
    needing to know which tables a given test wrote to.
    """
    from app.db.connection import db_connection

    with db_connection() as connection:
        with connection.cursor() as open_cursor:
            yield open_cursor
        connection.rollback()


@pytest.fixture
def client(migrated_database):
    """FastAPI test client with the app wired to the scratch database."""
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
