"""Cross-collection workspace search."""

from __future__ import annotations

from typing import Any

from mcp.types import ToolAnnotations

from app.db.repository import to_epoch_ms
from app.mcp.dbutil import clamp_limit, run_db_async, tool_guard
from app.mcp.server import mcp
from app.mcp.workspace import resolve_workspace_id


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def search_workspace(
    query: str,
    kinds: list[str] | None = None,
    limit: int = 50,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """
    ILIKE pass over notes, annotations, library_records, and case fields.

    Results include a where-to-open pointer for the web app.
    """

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        lim = clamp_limit(limit)
        like = f"%{query}%"
        results: list[dict[str, Any]] = []
        want = set(kinds) if kinds else None

        if want is None or "notes" in want:
            cursor.execute(
                """
                SELECT case_id, layer_id, html, updated_at
                FROM notes
                WHERE workspace_id = %s AND deleted_at IS NULL AND html ILIKE %s
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (wid, like, lim),
            )
            for row in cursor.fetchall():
                html = row["html"] or ""
                idx = html.lower().find(query.lower())
                snippet = html[max(0, idx - 40) : idx + 80] if idx >= 0 else html[:120]
                results.append(
                    {
                        "type": "note",
                        "id": f"{row['case_id']}:{row['layer_id']}",
                        "case_id": row["case_id"],
                        "layer_id": row["layer_id"],
                        "snippet": snippet,
                        "updated_at": to_epoch_ms(row["updated_at"]),
                        "open": {
                            "view": "case_notes",
                            "case_id": row["case_id"],
                            "layer_id": row["layer_id"],
                        },
                    }
                )

        if want is None or "annotations" in want:
            cursor.execute(
                """
                SELECT id, case_id, page, body, quote, updated_at
                FROM annotations
                WHERE workspace_id = %s AND deleted_at IS NULL
                  AND (body ILIKE %s OR quote ILIKE %s)
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (wid, like, like, lim),
            )
            for row in cursor.fetchall():
                text = row["body"] or row["quote"] or ""
                results.append(
                    {
                        "type": "annotation",
                        "id": row["id"],
                        "case_id": row["case_id"],
                        "page": row["page"],
                        "snippet": text[:160],
                        "updated_at": to_epoch_ms(row["updated_at"]),
                        "open": {
                            "view": "annotation",
                            "case_id": row["case_id"],
                            "annotation_id": row["id"],
                            "page": row["page"],
                        },
                    }
                )

        if want is None or "cases" in want:
            cursor.execute(
                """
                SELECT id, name, cite, holding, rule, headline_note, updated_at
                FROM cases
                WHERE workspace_id = %s AND deleted_at IS NULL
                  AND (
                    name ILIKE %s OR cite ILIKE %s OR holding ILIKE %s
                    OR rule ILIKE %s OR headline_note ILIKE %s
                  )
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                (wid, like, like, like, like, like, lim),
            )
            for row in cursor.fetchall():
                results.append(
                    {
                        "type": "case",
                        "id": row["id"],
                        "snippet": f"{row['name']} — {row['cite']}"[:160],
                        "updated_at": to_epoch_ms(row["updated_at"]),
                        "open": {"view": "case", "case_id": row["id"]},
                    }
                )

        if want is None or "library" in want or (
            want and any(k not in ("notes", "annotations", "cases") for k in want)
        ):
            kind_filter = ""
            params: list[Any] = [wid, like]
            if want:
                lib_kinds = [
                    k
                    for k in want
                    if k not in ("notes", "annotations", "cases", "library")
                ]
                if lib_kinds:
                    kind_filter = " AND kind = ANY(%s)"
                    params.append(lib_kinds)
                elif "library" not in want and want is not None:
                    lib_kinds = []
            params.append(lim)
            cursor.execute(
                f"""
                SELECT kind, id, data, updated_at
                FROM library_records
                WHERE workspace_id = %s AND deleted_at IS NULL
                  AND data::text ILIKE %s
                  {kind_filter}
                ORDER BY updated_at DESC
                LIMIT %s
                """,
                tuple(params),
            )
            for row in cursor.fetchall():
                results.append(
                    {
                        "type": "library",
                        "kind": row["kind"],
                        "id": row["id"],
                        "snippet": str(row["data"])[:160],
                        "updated_at": to_epoch_ms(row["updated_at"]),
                        "open": {
                            "view": "library",
                            "kind": row["kind"],
                            "id": row["id"],
                        },
                    }
                )

        results.sort(key=lambda r: r.get("updated_at") or 0, reverse=True)
        return {"results": results[:lim], "query": query}

    return await run_db_async(_sync)
