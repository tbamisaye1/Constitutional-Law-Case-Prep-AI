"""
Sync endpoints for the local-first client.

The client stays the fast path: it writes to localStorage and IndexedDB first,
then reconciles with Postgres here. One POST does push-then-pull in a single
round trip, because the client has nothing useful to do between the two.

Cursor handling matters. The client stores `serverTime` from the response and
sends it back as `since` next time, so the cursor is always server-generated.
If the client used its own clock, a device running a few minutes fast would set
a cursor in the future and silently stop receiving changes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import WorkspaceSession, workspace_session
from app.db.repository import (
    ENTITY_BY_NAME,
    MAX_ROWS_PER_ENTITY,
    pull_changes,
    push_changes,
    to_epoch_ms,
    workspace_counts,
)

router = APIRouter(prefix="/sync", tags=["sync"])


class SyncPushRequest(BaseModel):
    """
    Rows the client changed, plus where it last read up to.

    Attributes:
        since: Server timestamp from the client's previous sync, epoch
            milliseconds. 0 asks for a full download.
        changes: Collection name to rows in the client's own shape. Collections
            this backend does not know are ignored rather than rejected, so a
            newer frontend can deploy before the API catches up.
    """

    since: int = Field(default=0, ge=0)
    changes: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class SyncResponse(BaseModel):
    server_time: int = Field(alias="serverTime")
    changes: dict[str, list[dict[str, Any]]]
    written: dict[str, int] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


def _reject_oversized(changes: dict[str, list[dict[str, Any]]]) -> None:
    for name, rows in changes.items():
        if len(rows) > MAX_ROWS_PER_ENTITY:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Too many {name} rows in one push ({len(rows)}); "
                    f"the limit is {MAX_ROWS_PER_ENTITY}. Sync in smaller batches."
                ),
            )


@router.get("", response_model=SyncResponse, response_model_by_alias=True)
def pull(
    since: int = Query(default=0, ge=0, description="Epoch ms cursor from a previous sync."),
    session: WorkspaceSession = Depends(workspace_session),
) -> SyncResponse:
    """Download everything in this workspace that changed after `since`."""
    now = datetime.now(timezone.utc)
    changes = pull_changes(session.cursor, session.workspace_id, since)
    return SyncResponse(serverTime=to_epoch_ms(now), changes=changes)


@router.post("", response_model=SyncResponse, response_model_by_alias=True)
def push_then_pull(
    body: SyncPushRequest,
    session: WorkspaceSession = Depends(workspace_session),
) -> SyncResponse:
    """
    Upload local changes, then download whatever else moved.

    Rows the client just sent can come back in the response. That is harmless
    and deliberate: the client applies a pulled row only when its updatedAt is
    strictly newer than the local copy, so an echo is a no-op instead of a loop.
    """
    _reject_oversized(body.changes)

    unknown = sorted(set(body.changes) - set(ENTITY_BY_NAME))
    now = datetime.now(timezone.utc)

    written = push_changes(session.cursor, session.workspace_id, body.changes, now)
    changes = pull_changes(session.cursor, session.workspace_id, body.since)

    if unknown:
        # Visible in the response rather than raising, so one unrecognised
        # collection never blocks a sync that is otherwise fine.
        written["ignored:" + ",".join(unknown)] = 0

    return SyncResponse(
        serverTime=to_epoch_ms(now),
        changes=changes,
        written=written,
    )


@router.get("/status")
def status(session: WorkspaceSession = Depends(workspace_session)) -> dict:
    """Live row counts for this workspace, for the sync line in the UI."""
    return {
        "workspaceId": session.workspace_id,
        "counts": workspace_counts(session.cursor, session.workspace_id),
        "serverTime": to_epoch_ms(datetime.now(timezone.utc)),
    }
