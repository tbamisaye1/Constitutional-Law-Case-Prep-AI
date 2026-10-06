"""
Durable workspace backups: list, create, download, restore.

These are Postgres snapshots of notes / arguments / annotations / cases /
document metadata. They survive hard refresh. PDF bytes stay in Blob; restore
rebrings the metadata that points at them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.api.deps import WorkspaceSession, workspace_session
from app.db.repository import (
    create_workspace_backup,
    get_workspace_backup,
    list_workspace_backups,
    restore_workspace_backup,
)

router = APIRouter(prefix="/sync/backups", tags=["backups"])


class CreateBackupBody(BaseModel):
    label: str = Field(default="Manual backup", max_length=200)


@router.get("")
def list_backups(
    session: WorkspaceSession = Depends(workspace_session),
) -> dict[str, Any]:
    return {
        "workspaceId": session.workspace_id,
        "backups": list_workspace_backups(session.cursor, session.workspace_id),
    }


@router.post("")
def create_backup(
    body: CreateBackupBody,
    session: WorkspaceSession = Depends(workspace_session),
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    label = (body.label or "Manual backup").strip() or "Manual backup"
    try:
        created = create_workspace_backup(
            session.cursor,
            session.workspace_id,
            label=label,
            source="manual",
            now=now,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not write backup. Is migration 007 applied?",
        ) from exc
    return created


@router.get("/{backup_id}/download")
def download_backup(
    backup_id: int,
    session: WorkspaceSession = Depends(workspace_session),
) -> JSONResponse:
    backup = get_workspace_backup(session.cursor, session.workspace_id, backup_id)
    if not backup:
        raise HTTPException(status_code=404, detail="Backup not found.")
    filename = f"case-prep-backup-{backup_id}.json"
    return JSONResponse(
        content={
            "id": backup["id"],
            "label": backup["label"],
            "source": backup["source"],
            "createdAt": backup["createdAt"],
            "payload": backup["payload"],
        },
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{backup_id}/restore")
def restore_backup(
    backup_id: int,
    session: WorkspaceSession = Depends(workspace_session),
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    try:
        result = restore_workspace_backup(
            session.cursor, session.workspace_id, backup_id, now
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="Backup not found.") from None
    return result
