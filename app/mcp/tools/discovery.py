"""Discovery and workspace tools."""

from __future__ import annotations

from typing import Any

from mcp.types import ToolAnnotations

from app.api.health import _database_health
from app.db.repository import to_epoch_ms, workspace_counts
from app.mcp.dbutil import clamp_limit, run_db_async, tool_guard
from app.mcp.server import mcp
from app.mcp.workspace import (
    label_workspace as label_workspace_row,
    resolve_workspace_id,
    set_default_workspace as store_default_workspace,
)
from app.rag.store import list_index_sources
from app.storage.blob_client import blob_configured


def _list_workspaces_sync(cursor, limit: int) -> list[dict[str, Any]]:
    cursor.execute(
        """
        SELECT id, label, created_at, last_seen_at
        FROM workspaces
        ORDER BY last_seen_at DESC
        LIMIT %s
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    results: list[dict[str, Any]] = []
    for row in rows:
        wid = str(row["id"])
        counts = workspace_counts(cursor, wid)
        cursor.execute(
            """
            SELECT count(*) AS n FROM workspace_backups WHERE workspace_id = %s
            """,
            (wid,),
        )
        backup_count = cursor.fetchone()["n"]
        cursor.execute(
            """
            SELECT updated_at
            FROM library_records
            WHERE workspace_id = %s AND kind = 'arguments' AND deleted_at IS NULL
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (wid,),
        )
        args_row = cursor.fetchone()
        # Recency for sorting: newest content across key tables.
        cursor.execute(
            """
            SELECT GREATEST(
                COALESCE((SELECT max(updated_at) FROM matters WHERE workspace_id = %s), '-infinity'),
                COALESCE((SELECT max(updated_at) FROM cases WHERE workspace_id = %s), '-infinity'),
                COALESCE((SELECT max(updated_at) FROM notes WHERE workspace_id = %s), '-infinity'),
                COALESCE((SELECT max(updated_at) FROM annotations WHERE workspace_id = %s), '-infinity'),
                COALESCE((SELECT max(updated_at) FROM library_records WHERE workspace_id = %s), '-infinity'),
                COALESCE((SELECT max(updated_at) FROM documents WHERE workspace_id = %s), '-infinity')
            ) AS content_at
            """,
            (wid, wid, wid, wid, wid, wid),
        )
        content_at = cursor.fetchone()["content_at"]
        results.append(
            {
                "id": wid,
                "label": row["label"],
                "created_at": to_epoch_ms(row["created_at"]),
                "last_seen_at": to_epoch_ms(row["last_seen_at"]),
                "counts": counts,
                "backups": backup_count,
                "arguments_updated_at": (
                    to_epoch_ms(args_row["updated_at"]) if args_row else None
                ),
                "_content_at": content_at,
            }
        )
    results.sort(
        key=lambda item: item.get("_content_at") or item["last_seen_at"],
        reverse=True,
    )
    for item in results:
        item.pop("_content_at", None)
    return results


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def health() -> dict[str, Any]:
    """Database, Blob, FAISS, and resolved default workspace status."""

    def _sync(cursor) -> dict[str, Any]:
        default_ws = None
        try:
            default_ws = resolve_workspace_id(cursor, None)
        except Exception:
            default_ws = None
        sources = list_index_sources()
        return {
            "database": _database_health(),
            "blobStorage": {"configured": blob_configured()},
            "faiss_sources": len(sources),
            "default_workspace_id": default_ws,
        }

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def list_workspaces(limit: int = 50) -> dict[str, Any]:
    """List every workspace with counts, sorted by most recent content."""
    lim = clamp_limit(limit)
    rows = await run_db_async(_list_workspaces_sync, lim)
    return {"workspaces": rows, "count": len(rows)}


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def set_default_workspace(workspace_id: str) -> dict[str, Any]:
    """Persist the default workspace in app_settings for later tool calls."""
    return await run_db_async(store_default_workspace, workspace_id)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def label_workspace(workspace_id: str, label: str) -> dict[str, Any]:
    """Set workspaces.label for easier recovery."""
    return await run_db_async(label_workspace_row, workspace_id, label)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def workspace_status(workspace_id: str | None = None) -> dict[str, Any]:
    """Counts, last sync audit, and last backup for a workspace."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        counts = workspace_counts(cursor, wid)
        cursor.execute(
            """
            SELECT created_at, event, note
            FROM sync_audit_log
            WHERE workspace_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (wid,),
        )
        last_sync = cursor.fetchone()
        cursor.execute(
            """
            SELECT id, label, source, created_at
            FROM workspace_backups
            WHERE workspace_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (wid,),
        )
        last_backup = cursor.fetchone()
        return {
            "workspace_id": wid,
            "counts": counts,
            "last_sync": (
                {
                    "created_at": to_epoch_ms(last_sync["created_at"]),
                    "event": last_sync["event"],
                    "note": last_sync["note"],
                }
                if last_sync
                else None
            ),
            "last_backup": (
                {
                    "id": last_backup["id"],
                    "label": last_backup["label"],
                    "source": last_backup["source"],
                    "created_at": to_epoch_ms(last_backup["created_at"]),
                }
                if last_backup
                else None
            ),
        }

    return await run_db_async(_sync)
