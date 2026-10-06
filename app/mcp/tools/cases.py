"""Matters and cases tools."""

from __future__ import annotations

from typing import Any

from mcp.types import ToolAnnotations

from app.db.repository import to_epoch_ms
from app.mcp.dbutil import clamp_limit, run_db_async, tool_guard
from app.mcp.errors import McpToolError
from app.mcp.server import mcp
from app.mcp.workspace import resolve_workspace_id
from app.mcp.writes import require_confirm, write_entity


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def list_matters(workspace_id: str | None = None) -> dict[str, Any]:
    """List matters from Postgres only (no seed merge)."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        cursor.execute(
            """
            SELECT id, title, season, issues, updated_at
            FROM matters
            WHERE workspace_id = %s AND deleted_at IS NULL
            ORDER BY updated_at DESC
            """,
            (wid,),
        )
        rows = [
            {
                "id": r["id"],
                "title": r["title"],
                "season": r["season"],
                "issues": r["issues"],
                "updated_at": to_epoch_ms(r["updated_at"]),
            }
            for r in cursor.fetchall()
        ]
        if not rows:
            return {
                "matters": [],
                "hint": "No matters in this workspace. Call upsert_matter to create one.",
            }
        return {"matters": rows}

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def upsert_matter(
    id: str,
    title: str = "",
    season: str = "",
    issues: list[Any] | None = None,
    expected_updated_at: int | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Create or update a matter row."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        return write_entity(
            cursor,
            wid,
            entity="matters",
            key={"id": id},
            values={
                "title": title,
                "season": season,
                "issues": issues if issues is not None else [],
            },
            tool_name="upsert_matter",
            expected_updated_at=expected_updated_at,
        )

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def list_cases(
    issue: int | None = None,
    tag: str | None = None,
    usefulness: str | None = None,
    query: str | None = None,
    limit: int = 50,
    cursor: int = 0,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """List cases with optional filters and ILIKE query."""

    def _sync(cur) -> dict[str, Any]:
        wid = resolve_workspace_id(cur, workspace_id)
        lim = clamp_limit(limit)
        offset = max(0, int(cursor or 0))
        clauses = ["workspace_id = %s", "deleted_at IS NULL"]
        params: list[Any] = [wid]
        if issue is not None:
            clauses.append("issue = %s")
            params.append(issue)
        if tag:
            clauses.append("tag = %s")
            params.append(tag)
        if usefulness:
            clauses.append("usefulness = %s")
            params.append(usefulness)
        if query:
            clauses.append(
                "(name ILIKE %s OR cite ILIKE %s OR holding ILIKE %s "
                "OR rule ILIKE %s OR headline_note ILIKE %s)"
            )
            like = f"%{query}%"
            params.extend([like, like, like, like, like])
        where = " AND ".join(clauses)
        params.extend([lim, offset])
        cur.execute(
            f"""
            SELECT id, name, cite, year, issue, tag, usefulness, headline_note,
                   holding, rule, use_petitioner, use_respondent, suggested_file,
                   updated_at
            FROM cases
            WHERE {where}
            ORDER BY updated_at DESC
            LIMIT %s OFFSET %s
            """,
            tuple(params),
        )
        rows = []
        for r in cur.fetchall():
            rows.append(
                {
                    "id": r["id"],
                    "name": r["name"],
                    "cite": r["cite"],
                    "year": r["year"],
                    "issue": r["issue"],
                    "tag": r["tag"],
                    "usefulness": r["usefulness"],
                    "headlineNote": r["headline_note"],
                    "holding": r["holding"],
                    "rule": r["rule"],
                    "usePetitioner": r["use_petitioner"],
                    "useRespondent": r["use_respondent"],
                    "suggestedFile": r["suggested_file"],
                    "updated_at": to_epoch_ms(r["updated_at"]),
                }
            )
        return {
            "cases": rows,
            "next_cursor": offset + len(rows) if len(rows) == lim else None,
        }

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def get_case(
    case_id: str,
    format: str = "markdown",
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Case row plus note layers, annotation count, and documents."""
    from app.mcp.convert import format_note_payload

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        cursor.execute(
            """
            SELECT *
            FROM cases
            WHERE workspace_id = %s AND id = %s AND deleted_at IS NULL
            """,
            (wid, case_id),
        )
        row = cursor.fetchone()
        if not row:
            raise McpToolError("not_found", "Case not found.", {"case_id": case_id})
        cursor.execute(
            """
            SELECT layer_id, html, updated_at
            FROM notes
            WHERE workspace_id = %s AND case_id = %s AND deleted_at IS NULL
            ORDER BY layer_id
            """,
            (wid, case_id),
        )
        notes = []
        for n in cursor.fetchall():
            payload = format_note_payload(n["html"] or "", format=format)  # type: ignore[arg-type]
            notes.append(
                {
                    "layer_id": n["layer_id"],
                    "updated_at": to_epoch_ms(n["updated_at"]),
                    **payload,
                }
            )
        cursor.execute(
            """
            SELECT count(*) AS n FROM annotations
            WHERE workspace_id = %s AND case_id = %s AND deleted_at IS NULL
            """,
            (wid, case_id),
        )
        anno_count = cursor.fetchone()["n"]
        cursor.execute(
            """
            SELECT id, name, size_bytes, content_type, blob_pathname, updated_at
            FROM documents
            WHERE workspace_id = %s AND case_id = %s AND deleted_at IS NULL
            """,
            (wid, case_id),
        )
        docs = [
            {
                "id": d["id"],
                "name": d["name"],
                "size": d["size_bytes"],
                "contentType": d["content_type"],
                "stored": d["blob_pathname"] is not None,
                "updated_at": to_epoch_ms(d["updated_at"]),
            }
            for d in cursor.fetchall()
        ]
        return {
            "id": row["id"],
            "name": row["name"],
            "cite": row["cite"],
            "year": row["year"],
            "issue": row["issue"],
            "tag": row["tag"],
            "usefulness": row["usefulness"],
            "headlineNote": row["headline_note"],
            "holding": row["holding"],
            "rule": row["rule"],
            "usePetitioner": row["use_petitioner"],
            "useRespondent": row["use_respondent"],
            "suggestedFile": row["suggested_file"],
            "updated_at": to_epoch_ms(row["updated_at"]),
            "notes": notes,
            "annotation_count": anno_count,
            "documents": docs,
        }

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def upsert_case(
    id: str,
    name: str = "",
    cite: str = "",
    year: str = "",
    issue: int | None = None,
    tag: str | None = None,
    usefulness: str = "background",
    headlineNote: str = "",
    holding: str = "",
    rule: str = "",
    usePetitioner: str = "",
    useRespondent: str = "",
    suggestedFile: str = "",
    expected_updated_at: int | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Create or update a case (camelCase fields match the sync wire format)."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        return write_entity(
            cursor,
            wid,
            entity="cases",
            key={"id": id},
            values={
                "name": name,
                "cite": cite,
                "year": year,
                "issue": issue,
                "tag": tag,
                "usefulness": usefulness,
                "headlineNote": headlineNote,
                "holding": holding,
                "rule": rule,
                "usePetitioner": usePetitioner,
                "useRespondent": useRespondent,
                "suggestedFile": suggestedFile,
            },
            tool_name="upsert_case",
            expected_updated_at=expected_updated_at,
        )

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False)
)
@tool_guard
async def delete_case(
    case_id: str,
    dry_run: bool = True,
    confirm: bool = False,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Soft-delete a case. Dry run reports notes and annotations that would orphan."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        cursor.execute(
            """
            SELECT id, name, updated_at FROM cases
            WHERE workspace_id = %s AND id = %s AND deleted_at IS NULL
            """,
            (wid, case_id),
        )
        row = cursor.fetchone()
        if not row:
            raise McpToolError("not_found", "Case not found.", {"case_id": case_id})
        cursor.execute(
            """
            SELECT count(*) AS n FROM notes
            WHERE workspace_id = %s AND case_id = %s AND deleted_at IS NULL
            """,
            (wid, case_id),
        )
        note_count = cursor.fetchone()["n"]
        cursor.execute(
            """
            SELECT count(*) AS n FROM annotations
            WHERE workspace_id = %s AND case_id = %s AND deleted_at IS NULL
            """,
            (wid, case_id),
        )
        anno_count = cursor.fetchone()["n"]
        summary = {
            "case_id": case_id,
            "name": row["name"],
            "notes_that_would_orphan": note_count,
            "annotations_that_would_orphan": anno_count,
        }
        if dry_run:
            return {"dry_run": True, **summary}
        require_confirm(dry_run=dry_run, confirm=confirm, action="delete_case")
        result = write_entity(
            cursor,
            wid,
            entity="cases",
            key={"id": case_id},
            values={},
            tool_name="delete_case",
            expected_updated_at=to_epoch_ms(row["updated_at"]),
            soft_delete=True,
        )
        return {"dry_run": False, **summary, **result}

    return await run_db_async(_sync)
