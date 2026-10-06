"""Annotation tools."""

from __future__ import annotations

from typing import Any

from mcp.types import ToolAnnotations

from app.db.repository import to_epoch_ms
from app.mcp.convert import resolve_write_body
from app.mcp.dbutil import clamp_limit, run_db_async, tool_guard
from app.mcp.errors import McpToolError
from app.mcp.ids import new_annotation_id
from app.mcp.server import mcp
from app.mcp.workspace import resolve_workspace_id
from app.mcp.writes import require_confirm, write_entity


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def list_annotations(
    case_id: str | None = None,
    document_id: str | None = None,
    page: int | None = None,
    topic: str | None = None,
    pinned: bool | None = None,
    kind: str | None = None,
    query: str | None = None,
    limit: int = 50,
    cursor: int = 0,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Filter annotations in the workspace."""

    def _sync(cur) -> dict[str, Any]:
        wid = resolve_workspace_id(cur, workspace_id)
        lim = clamp_limit(limit)
        offset = max(0, int(cursor or 0))
        clauses = ["workspace_id = %s", "deleted_at IS NULL"]
        params: list[Any] = [wid]
        if case_id:
            clauses.append("case_id = %s")
            params.append(case_id)
        if document_id:
            clauses.append("document_id = %s")
            params.append(document_id)
        if page is not None:
            clauses.append("page = %s")
            params.append(page)
        if pinned is not None:
            clauses.append("pinned = %s")
            params.append(pinned)
        if kind:
            clauses.append("kind = %s")
            params.append(kind)
        if topic:
            clauses.append("topics::text ILIKE %s")
            params.append(f"%{topic}%")
        if query:
            clauses.append("(body ILIKE %s OR quote ILIKE %s)")
            like = f"%{query}%"
            params.extend([like, like])
        where = " AND ".join(clauses)
        params.extend([lim, offset])
        cur.execute(
            f"""
            SELECT id, case_id, document_id, page, kind, quote, body, rects,
                   pinned, color, topics, updated_at
            FROM annotations
            WHERE {where}
            ORDER BY updated_at DESC
            LIMIT %s OFFSET %s
            """,
            tuple(params),
        )
        rows = [
            {
                "id": r["id"],
                "caseId": r["case_id"],
                "fileId": r["document_id"],
                "page": r["page"],
                "kind": r["kind"],
                "quote": r["quote"],
                "text": r["body"],
                "rects": r["rects"],
                "pinned": r["pinned"],
                "color": r["color"],
                "topics": r["topics"],
                "updated_at": to_epoch_ms(r["updated_at"]),
            }
            for r in cur.fetchall()
        ]
        return {
            "annotations": rows,
            "next_cursor": offset + len(rows) if len(rows) == lim else None,
        }

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def create_annotation(
    case_id: str,
    page: int,
    kind: str,
    body_markdown: str | None = None,
    body_html: str | None = None,
    quote: str | None = None,
    document_id: str | None = None,
    topics: list[Any] | None = None,
    color: str | None = None,
    pinned: bool = False,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """
    Create an annotation. rects stay null (page note; UI must render null-rect
    annotations as page notes).
    """

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        try:
            body = resolve_write_body(markdown=body_markdown, html=body_html)
        except ValueError as exc:
            raise McpToolError("invalid", str(exc)) from exc
        if kind not in ("highlight", "page"):
            raise McpToolError("invalid", "kind must be highlight or page.")
        anno_id = new_annotation_id()
        return write_entity(
            cursor,
            wid,
            entity="annotations",
            key={"id": anno_id},
            values={
                "caseId": case_id,
                "fileId": document_id,
                "page": page,
                "kind": kind,
                "quote": quote or "",
                "text": body,
                "rects": None,
                "pinned": pinned,
                "color": color,
                "topics": topics if topics is not None else [],
            },
            tool_name="create_annotation",
            expected_updated_at=None,
        )

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def update_annotation(
    id: str,
    expected_updated_at: int,
    case_id: str | None = None,
    page: int | None = None,
    kind: str | None = None,
    quote: str | None = None,
    body_markdown: str | None = None,
    body_html: str | None = None,
    document_id: str | None = None,
    topics: list[Any] | None = None,
    color: str | None = None,
    pinned: bool | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Update annotation fields under optimistic concurrency."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        cursor.execute(
            """
            SELECT * FROM annotations
            WHERE workspace_id = %s AND id = %s AND deleted_at IS NULL
            FOR UPDATE
            """,
            (wid, id),
        )
        existing = cursor.fetchone()
        if not existing:
            raise McpToolError("not_found", "Annotation not found.", {"id": id})
        values: dict[str, Any] = {
            "caseId": case_id if case_id is not None else existing["case_id"],
            "fileId": document_id if document_id is not None else existing["document_id"],
            "page": page if page is not None else existing["page"],
            "kind": kind if kind is not None else existing["kind"],
            "quote": quote if quote is not None else existing["quote"],
            "text": existing["body"],
            "rects": existing["rects"],
            "pinned": pinned if pinned is not None else existing["pinned"],
            "color": color if color is not None else existing["color"],
            "topics": topics if topics is not None else existing["topics"],
        }
        if body_markdown is not None or body_html is not None:
            try:
                values["text"] = resolve_write_body(
                    markdown=body_markdown, html=body_html
                )
            except ValueError as exc:
                raise McpToolError("invalid", str(exc)) from exc
        return write_entity(
            cursor,
            wid,
            entity="annotations",
            key={"id": id},
            values=values,
            tool_name="update_annotation",
            expected_updated_at=expected_updated_at,
        )

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False)
)
@tool_guard
async def delete_annotation(
    id: str,
    dry_run: bool = True,
    confirm: bool = False,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Soft-delete an annotation."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        cursor.execute(
            """
            SELECT id, case_id, page, body, updated_at
            FROM annotations
            WHERE workspace_id = %s AND id = %s AND deleted_at IS NULL
            """,
            (wid, id),
        )
        row = cursor.fetchone()
        if not row:
            raise McpToolError("not_found", "Annotation not found.", {"id": id})
        summary = {
            "id": id,
            "caseId": row["case_id"],
            "page": row["page"],
            "preview": (row["body"] or "")[:120],
        }
        if dry_run:
            return {"dry_run": True, **summary}
        require_confirm(dry_run=dry_run, confirm=confirm, action="delete_annotation")
        result = write_entity(
            cursor,
            wid,
            entity="annotations",
            key={"id": id},
            values={},
            tool_name="delete_annotation",
            expected_updated_at=to_epoch_ms(row["updated_at"]),
            soft_delete=True,
        )
        return {"dry_run": False, **summary, **result}

    return await run_db_async(_sync)
