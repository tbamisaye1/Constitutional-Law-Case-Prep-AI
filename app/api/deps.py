"""
Shared request dependencies: workspace identity and a database cursor.

Identity today is a key the browser generates once and keeps in localStorage,
sent as X-Workspace-Id. That is not authentication. It stops two people's prep
from colliding and it lets you open the same notes on a second device by
copying the key across, but anyone holding a key can read that workspace. The
next step here is real accounts (see docs/TODO.md), at which point the key
becomes a claimable row rather than the whole security model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
from uuid import UUID

import psycopg
from fastapi import Depends, Header, HTTPException

from app.db.connection import DatabaseNotConfigured, db_connection
from app.db.repository import touch_workspace

WORKSPACE_HEADER = "X-Workspace-Id"


def require_workspace_id(x_workspace_id: str = Header(..., alias=WORKSPACE_HEADER)) -> str:
    """
    Validate and normalise the caller's workspace key.

    Parsing it as a UUID is what keeps the value safe to hand to Postgres as a
    uuid parameter, and it rejects junk before we open a connection.

    Raises:
        HTTPException: 400 when the header is not a UUID.
    """
    try:
        return str(UUID(x_workspace_id.strip()))
    except (ValueError, AttributeError):
        raise HTTPException(
            status_code=400,
            detail=f"{WORKSPACE_HEADER} must be a UUID.",
        ) from None


def optional_workspace_id(
    x_workspace_id: str | None = Header(default=None, alias=WORKSPACE_HEADER),
) -> str | None:
    """
    Same validation as require_workspace_id, but a missing header is allowed.

    Used by endpoints that have to keep answering older clients which predate
    workspace keys. A malformed header is still an error: silently ignoring it
    would hand back the wrong workspace's data.
    """
    if x_workspace_id is None or not x_workspace_id.strip():
        return None
    return require_workspace_id(x_workspace_id)


@dataclass
class WorkspaceSession:
    """A validated workspace plus an open cursor inside one transaction."""

    workspace_id: str
    cursor: psycopg.Cursor


def workspace_session(
    workspace_id: str = Depends(require_workspace_id),
) -> Iterator[WorkspaceSession]:
    """
    Open a transaction for this workspace and make sure its row exists.

    The transaction commits when the route returns and rolls back if it raises,
    so a handler that fails halfway cannot leave a partial sync behind.

    Raises:
        HTTPException: 503 when the database is unconfigured or unreachable,
            which the client shows as "working offline" rather than an error.
    """
    try:
        with db_connection() as connection:
            with connection.cursor() as cursor:
                touch_workspace(cursor, workspace_id)
                yield WorkspaceSession(workspace_id=workspace_id, cursor=cursor)
    except DatabaseNotConfigured as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except psycopg.OperationalError as error:
        raise HTTPException(
            status_code=503,
            detail=f"Could not reach the database: {error}",
        ) from error
