"""Case note layer tools."""

from __future__ import annotations

from typing import Any

from mcp.types import ToolAnnotations

from app.db.repository import to_epoch_ms
from app.mcp.convert import format_note_payload, resolve_write_body
from app.mcp.dbutil import run_db_async, tool_guard
from app.mcp.errors import McpToolError
from app.mcp.server import mcp
from app.mcp.workspace import resolve_workspace_id
from app.mcp.writes import write_entity


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def list_note_layers(
    case_id: str,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """List note layers for a case with updated_at and character counts."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        cursor.execute(
            """
            SELECT layer_id, html, updated_at
            FROM notes
            WHERE workspace_id = %s AND case_id = %s AND deleted_at IS NULL
            ORDER BY layer_id
            """,
            (wid, case_id),
        )
        layers = [
            {
                "layer_id": r["layer_id"],
                "updated_at": to_epoch_ms(r["updated_at"]),
                "chars": len(r["html"] or ""),
            }
            for r in cursor.fetchall()
        ]
        return {"case_id": case_id, "layers": layers}

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def get_note(
    case_id: str,
    layer_id: str,
    format: str = "markdown",
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Read one note layer."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        cursor.execute(
            """
            SELECT html, updated_at
            FROM notes
            WHERE workspace_id = %s AND case_id = %s AND layer_id = %s
              AND deleted_at IS NULL
            """,
            (wid, case_id, layer_id),
        )
        row = cursor.fetchone()
        if not row:
            raise McpToolError(
                "not_found",
                "Note not found.",
                {"case_id": case_id, "layer_id": layer_id},
            )
        payload = format_note_payload(row["html"] or "", format=format)  # type: ignore[arg-type]
        return {
            "case_id": case_id,
            "layer_id": layer_id,
            "updated_at": to_epoch_ms(row["updated_at"]),
            **payload,
        }

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def set_note(
    case_id: str,
    layer_id: str,
    markdown: str | None = None,
    html: str | None = None,
    expected_updated_at: int | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Replace a note layer body (exactly one of markdown or html)."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        try:
            body = resolve_write_body(markdown=markdown, html=html)
        except ValueError as exc:
            raise McpToolError("invalid", str(exc)) from exc
        return write_entity(
            cursor,
            wid,
            entity="notes",
            key={"caseId": case_id, "layerId": layer_id},
            values={"html": body},
            tool_name="set_note",
            expected_updated_at=expected_updated_at,
        )

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def append_to_note(
    case_id: str,
    layer_id: str,
    expected_updated_at: int,
    markdown: str | None = None,
    html: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Append content to a note under the row lock (avoids read-modify-write races)."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        try:
            addition = resolve_write_body(markdown=markdown, html=html)
        except ValueError as exc:
            raise McpToolError("invalid", str(exc)) from exc
        cursor.execute(
            """
            SELECT html, updated_at
            FROM notes
            WHERE workspace_id = %s AND case_id = %s AND layer_id = %s
            FOR UPDATE
            """,
            (wid, case_id, layer_id),
        )
        row = cursor.fetchone()
        if row is None:
            combined = addition
            expected = None
        else:
            if to_epoch_ms(row["updated_at"]) != int(expected_updated_at):
                raise McpToolError(
                    "conflict",
                    "Row was modified.",
                    {
                        "current": {
                            "caseId": case_id,
                            "layerId": layer_id,
                            "html": row["html"],
                            "updatedAt": to_epoch_ms(row["updated_at"]),
                        }
                    },
                )
            combined = (row["html"] or "") + addition
            expected = expected_updated_at
        return write_entity(
            cursor,
            wid,
            entity="notes",
            key={"caseId": case_id, "layerId": layer_id},
            values={"html": combined},
            tool_name="append_to_note",
            expected_updated_at=expected,
        )

    return await run_db_async(_sync)
