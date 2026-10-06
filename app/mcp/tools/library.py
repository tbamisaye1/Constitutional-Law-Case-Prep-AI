"""Generic library_records tools (kinds discovered dynamically)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import jsonpatch
from mcp.types import ToolAnnotations

from app.db.repository import to_epoch_ms
from app.mcp.dbutil import clamp_limit, run_db_async, tool_guard
from app.mcp.errors import McpToolError
from app.mcp.server import mcp
from app.mcp.workspace import resolve_workspace_id
from app.mcp.writes import require_confirm, validate_library_kind, write_library_data, write_entity


def _preview_data(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    for key in ("title", "name"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:120]
    # Arguments board preview
    drafts = data.get("draftsBySide")
    if isinstance(drafts, dict):
        for side in ("petitioner", "respondent"):
            side_drafts = drafts.get(side)
            if isinstance(side_drafts, list) and side_drafts:
                draft = side_drafts[0]
                if isinstance(draft, dict):
                    name = draft.get("name") or draft.get("id")
                    sections = draft.get("sections") or []
                    sec_title = ""
                    if sections and isinstance(sections[0], dict):
                        sec_title = sections[0].get("title") or ""
                    return f"{side}: {name}" + (f" / {sec_title}" if sec_title else "")
    return None


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def list_library_kinds(workspace_id: str | None = None) -> dict[str, Any]:
    """Discover library kinds present in this workspace."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        cursor.execute(
            """
            SELECT kind, count(*) AS n, max(updated_at) AS latest
            FROM library_records
            WHERE workspace_id = %s AND deleted_at IS NULL
            GROUP BY kind
            ORDER BY latest DESC NULLS LAST
            """,
            (wid,),
        )
        return {
            "kinds": [
                {
                    "kind": r["kind"],
                    "count": r["n"],
                    "updated_at": to_epoch_ms(r["latest"]) if r["latest"] else None,
                }
                for r in cursor.fetchall()
            ]
        }

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def list_library_records(
    kind: str,
    limit: int = 50,
    cursor: int = 0,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """List records for one kind with short previews."""

    def _sync(cur) -> dict[str, Any]:
        wid = resolve_workspace_id(cur, workspace_id)
        validate_library_kind(kind)
        lim = clamp_limit(limit)
        offset = max(0, int(cursor or 0))
        cur.execute(
            """
            SELECT id, data, updated_at
            FROM library_records
            WHERE workspace_id = %s AND kind = %s AND deleted_at IS NULL
            ORDER BY updated_at DESC
            LIMIT %s OFFSET %s
            """,
            (wid, kind, lim, offset),
        )
        rows = [
            {
                "id": r["id"],
                "updated_at": to_epoch_ms(r["updated_at"]),
                "preview": _preview_data(r["data"]),
            }
            for r in cur.fetchall()
        ]
        return {
            "kind": kind,
            "records": rows,
            "next_cursor": offset + len(rows) if len(rows) == lim else None,
        }

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def get_library_record(
    kind: str,
    id: str,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Return one library record's data and updated_at."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        validate_library_kind(kind)
        cursor.execute(
            """
            SELECT data, updated_at
            FROM library_records
            WHERE workspace_id = %s AND kind = %s AND id = %s AND deleted_at IS NULL
            """,
            (wid, kind, id),
        )
        row = cursor.fetchone()
        if not row:
            raise McpToolError(
                "not_found",
                "Library record not found.",
                {"kind": kind, "id": id},
            )
        return {
            "kind": kind,
            "id": id,
            "data": row["data"],
            "updated_at": to_epoch_ms(row["updated_at"]),
        }

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def upsert_library_record(
    kind: str,
    id: str,
    data: dict[str, Any],
    expected_updated_at: int | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Create or replace a library record."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        return write_library_data(
            cursor,
            wid,
            kind=kind,
            record_id=id,
            data=data,
            tool_name="upsert_library_record",
            expected_updated_at=expected_updated_at,
        )

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def patch_library_record(
    kind: str,
    id: str,
    json_patch: list[dict[str, Any]],
    expected_updated_at: int,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Apply RFC 6902 JSON Patch under the row lock."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        validate_library_kind(kind)
        cursor.execute(
            """
            SELECT data, updated_at
            FROM library_records
            WHERE workspace_id = %s AND kind = %s AND id = %s AND deleted_at IS NULL
            FOR UPDATE
            """,
            (wid, kind, id),
        )
        row = cursor.fetchone()
        if not row:
            raise McpToolError(
                "not_found",
                "Library record not found.",
                {"kind": kind, "id": id},
            )
        if to_epoch_ms(row["updated_at"]) != int(expected_updated_at):
            raise McpToolError(
                "conflict",
                "Row was modified.",
                {
                    "current": {
                        "kind": kind,
                        "id": id,
                        "data": row["data"],
                        "updatedAt": to_epoch_ms(row["updated_at"]),
                    }
                },
            )
        try:
            patched = jsonpatch.apply_patch(
                deepcopy(row["data"]), json_patch, in_place=False
            )
        except Exception as exc:
            raise McpToolError("invalid", f"JSON patch failed: {exc}") from exc
        return write_library_data(
            cursor,
            wid,
            kind=kind,
            record_id=id,
            data=patched if isinstance(patched, dict) else {"value": patched},
            tool_name="patch_library_record",
            expected_updated_at=expected_updated_at,
        )

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False)
)
@tool_guard
async def delete_library_record(
    kind: str,
    id: str,
    dry_run: bool = True,
    confirm: bool = False,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Soft-delete a library record."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        validate_library_kind(kind)
        cursor.execute(
            """
            SELECT data, updated_at
            FROM library_records
            WHERE workspace_id = %s AND kind = %s AND id = %s AND deleted_at IS NULL
            """,
            (wid, kind, id),
        )
        row = cursor.fetchone()
        if not row:
            raise McpToolError(
                "not_found",
                "Library record not found.",
                {"kind": kind, "id": id},
            )
        summary = {
            "kind": kind,
            "id": id,
            "preview": _preview_data(row["data"]),
            "bytes": len(str(row["data"])),
        }
        if dry_run:
            return {"dry_run": True, **summary}
        require_confirm(dry_run=dry_run, confirm=confirm, action="delete_library_record")
        result = write_entity(
            cursor,
            wid,
            entity="library_records",
            key={"kind": kind, "id": id},
            values={"data": row["data"]},
            tool_name="delete_library_record",
            expected_updated_at=to_epoch_ms(row["updated_at"]),
            soft_delete=True,
        )
        return {"dry_run": False, **summary, **result}

    return await run_db_async(_sync)
