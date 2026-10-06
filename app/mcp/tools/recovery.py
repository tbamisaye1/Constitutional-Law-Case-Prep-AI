"""Recovery tools: find orphaned workspaces, backups, revisions, merge."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mcp.types import ToolAnnotations
from psycopg.types.json import Jsonb

from app.db.repository import (
    create_workspace_backup,
    get_workspace_backup,
    list_workspace_backups,
    restore_workspace_backup,
    to_epoch_ms,
    touch_workspace,
)
from app.mcp.dbutil import clamp_limit, run_db_async, tool_guard
from app.mcp.errors import McpToolError
from app.mcp.server import mcp
from app.mcp.workspace import resolve_workspace_id
from app.mcp.writes import require_confirm, write_library_data


def _args_preview(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    drafts = (data.get("draftsBySide") or {}).get("petitioner") or []
    if not drafts or not isinstance(drafts[0], dict):
        return ""
    draft = drafts[0]
    name = draft.get("name") or draft.get("id") or "?"
    sections = draft.get("sections") or []
    sec = ""
    if sections and isinstance(sections[0], dict):
        sec = sections[0].get("title") or ""
    return f"{name}" + (f" · {sec}" if sec else "")


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def find_my_data(query: str | None = None) -> dict[str, Any]:
    """
    Rank every workspace by how much real prep it holds.
    Answers "where did my data go" without knowing a workspace id.
    """

    def _sync(cursor) -> dict[str, Any]:
        cursor.execute(
            "SELECT id, label, created_at, last_seen_at FROM workspaces"
        )
        workspaces = cursor.fetchall()
        ranked: list[dict[str, Any]] = []
        for ws in workspaces:
            wid = str(ws["id"])
            cursor.execute(
                """
                SELECT
                  COALESCE(sum(length(coalesce(html, ''))), 0) AS note_chars
                FROM notes
                WHERE workspace_id = %s AND deleted_at IS NULL
                """,
                (wid,),
            )
            note_chars = int(cursor.fetchone()["note_chars"] or 0)
            cursor.execute(
                """
                SELECT count(*) AS n FROM annotations
                WHERE workspace_id = %s AND deleted_at IS NULL
                """,
                (wid,),
            )
            anno_count = int(cursor.fetchone()["n"])
            cursor.execute(
                """
                SELECT count(*) AS n FROM cases
                WHERE workspace_id = %s AND deleted_at IS NULL
                """,
                (wid,),
            )
            case_count = int(cursor.fetchone()["n"])
            cursor.execute(
                """
                SELECT data, length(data::text) AS chars
                FROM library_records
                WHERE workspace_id = %s AND kind = 'arguments'
                  AND deleted_at IS NULL
                LIMIT 1
                """,
                (wid,),
            )
            args_row = cursor.fetchone()
            arg_chars = int(args_row["chars"]) if args_row else 0
            preview = _args_preview(args_row["data"]) if args_row else ""
            if query:
                q = query.lower()
                blob = f"{ws['label']} {preview}".lower()
                if q not in blob and q not in wid.lower():
                    # Still include if query matches case names.
                    cursor.execute(
                        """
                        SELECT 1 FROM cases
                        WHERE workspace_id = %s AND deleted_at IS NULL
                          AND (name ILIKE %s OR cite ILIKE %s)
                        LIMIT 1
                        """,
                        (wid, f"%{query}%", f"%{query}%"),
                    )
                    if not cursor.fetchone():
                        continue
            score = arg_chars + note_chars + (anno_count * 50) + (case_count * 20)
            ranked.append(
                {
                    "id": wid,
                    "label": ws["label"],
                    "created_at": to_epoch_ms(ws["created_at"]),
                    "last_seen_at": to_epoch_ms(ws["last_seen_at"]),
                    "score": score,
                    "argument_chars": arg_chars,
                    "note_chars": note_chars,
                    "annotations": anno_count,
                    "cases": case_count,
                    "arguments_preview": preview,
                }
            )
        ranked.sort(key=lambda row: row["score"], reverse=True)
        return {"workspaces": ranked}

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def list_backups(
    workspace_id: str | None = None,
    source: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List workspace backups with byte sizes."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        rows = list_workspace_backups(cursor, wid, limit=clamp_limit(limit))
        if source:
            rows = [r for r in rows if str(r.get("source", "")).startswith(source)]
        return {"workspace_id": wid, "backups": rows}

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def get_backup(
    backup_id: int,
    include: str = "summary",
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Fetch a backup. include: summary | arguments | notebook | full."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        backup = get_workspace_backup(cursor, wid, int(backup_id))
        if not backup:
            raise McpToolError("not_found", "Backup not found.", {"backup_id": backup_id})
        payload = backup.get("payload") or {}
        base = {
            "id": backup["id"],
            "label": backup["label"],
            "source": backup["source"],
            "created_at": backup["createdAt"],
        }
        if include == "summary":
            includes = payload.get("includes") if isinstance(payload, dict) else []
            return {
                **base,
                "includes": includes,
                "has_arguments": bool(
                    isinstance(payload, dict) and payload.get("arguments")
                ),
            }
        if include == "arguments":
            return {**base, "arguments": payload.get("arguments") if isinstance(payload, dict) else None}
        if include == "notebook":
            return {**base, "notebook": payload.get("notebook") if isinstance(payload, dict) else None}
        if include == "full":
            return {**base, "payload": payload}
        raise McpToolError(
            "invalid",
            "include must be summary, arguments, notebook, or full.",
        )

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def create_backup(
    label: str,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Create a manual workspace backup."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        now = datetime.now(timezone.utc)
        return create_workspace_backup(
            cursor, wid, label=label, source="manual", now=now
        )

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False)
)
@tool_guard
async def restore_backup(
    backup_id: int,
    dry_run: bool = True,
    confirm: bool = False,
    into_workspace_id: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Restore a backup. Dry run diffs kinds; confirm writes after a safety backup."""

    def _sync(cursor) -> dict[str, Any]:
        source_ws = resolve_workspace_id(cursor, workspace_id)
        target_ws = into_workspace_id or source_ws
        touch_workspace(cursor, target_ws)
        backup = get_workspace_backup(cursor, source_ws, int(backup_id))
        if not backup:
            # Allow restoring a backup that lives on another workspace id key.
            cursor.execute(
                """
                SELECT id, label, source, created_at, payload, workspace_id
                FROM workspace_backups WHERE id = %s
                """,
                (int(backup_id),),
            )
            row = cursor.fetchone()
            if not row:
                raise McpToolError("not_found", "Backup not found.")
            backup = {
                "id": row["id"],
                "label": row["label"],
                "source": row["source"],
                "createdAt": to_epoch_ms(row["created_at"]),
                "payload": row["payload"],
                "workspace_id": str(row["workspace_id"]),
            }
            source_ws = str(row["workspace_id"])

        payload = backup.get("payload") or {}
        # Diff library kinds present in backup vs live.
        from app.db.repository import _library_records_from_backup_payload

        records = _library_records_from_backup_payload(payload)
        diffs = []
        for record in records:
            kind = record["kind"]
            rid = record["id"]
            cursor.execute(
                """
                SELECT updated_at, length(data::text) AS bytes
                FROM library_records
                WHERE workspace_id = %s AND kind = %s AND id = %s AND deleted_at IS NULL
                """,
                (target_ws, kind, rid),
            )
            live = cursor.fetchone()
            diffs.append(
                {
                    "kind": kind,
                    "id": rid,
                    "backup_bytes": len(str(record.get("data"))),
                    "live_bytes": int(live["bytes"]) if live else 0,
                    "live_updated_at": to_epoch_ms(live["updated_at"]) if live else None,
                }
            )
        if dry_run:
            return {
                "dry_run": True,
                "source_workspace_id": source_ws,
                "target_workspace_id": target_ws,
                "backup_id": backup_id,
                "diffs": diffs,
            }
        require_confirm(dry_run=dry_run, confirm=confirm, action="restore_backup")
        now = datetime.now(timezone.utc)
        # restore_workspace_backup already takes a before_restore snapshot.
        if target_ws == source_ws:
            result = restore_workspace_backup(cursor, target_ws, int(backup_id), now)
        else:
            # Copy payload records into target via library upserts.
            create_workspace_backup(
                cursor,
                target_ws,
                label=f"Before restore of #{backup_id}",
                source="before_restore",
                now=now,
            )
            restored = 0
            for record in records:
                cursor.execute(
                    """
                    INSERT INTO library_records (workspace_id, kind, id, data, updated_at, deleted_at)
                    VALUES (%s, %s, %s, %s, %s, NULL)
                    ON CONFLICT (workspace_id, kind, id) DO UPDATE SET
                        data = EXCLUDED.data,
                        updated_at = EXCLUDED.updated_at,
                        deleted_at = NULL
                    """,
                    (
                        target_ws,
                        record["kind"],
                        record["id"],
                        Jsonb(record["data"]),
                        now,
                    ),
                )
                restored += 1
            result = {
                "restoredBackupId": backup_id,
                "libraryRecords": restored,
                "target_workspace_id": target_ws,
            }
        return {"dry_run": False, **result}

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def list_revisions(
    kind: str,
    id: str,
    limit: int = 50,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """List library_record_revisions for one row."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        lim = clamp_limit(limit)
        cursor.execute(
            """
            SELECT id, revised_at, source, pg_column_size(data) AS bytes,
                   left(data::text, 160) AS preview
            FROM library_record_revisions
            WHERE workspace_id = %s AND kind = %s AND record_id = %s
            ORDER BY revised_at DESC
            LIMIT %s
            """,
            (wid, kind, id, lim),
        )
        return {
            "kind": kind,
            "id": id,
            "revisions": [
                {
                    "revision_id": r["id"],
                    "revised_at": to_epoch_ms(r["revised_at"]),
                    "source": r["source"],
                    "bytes": r["bytes"] or 0,
                    "preview": r["preview"],
                }
                for r in cursor.fetchall()
            ],
        }

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False)
)
@tool_guard
async def restore_revision(
    revision_id: int,
    dry_run: bool = True,
    confirm: bool = False,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Restore a library revision through the MCP write path (itself revisioned)."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        cursor.execute(
            """
            SELECT id, workspace_id, kind, record_id, data, revised_at, source
            FROM library_record_revisions
            WHERE id = %s
            """,
            (int(revision_id),),
        )
        rev = cursor.fetchone()
        if not rev:
            raise McpToolError("not_found", "Revision not found.")
        if str(rev["workspace_id"]) != wid:
            raise McpToolError(
                "forbidden",
                "Revision belongs to another workspace. Pass that workspace_id.",
                {"workspace_id": str(rev["workspace_id"])},
            )
        cursor.execute(
            """
            SELECT data, updated_at FROM library_records
            WHERE workspace_id = %s AND kind = %s AND id = %s AND deleted_at IS NULL
            """,
            (wid, rev["kind"], rev["record_id"]),
        )
        live = cursor.fetchone()
        summary = {
            "revision_id": revision_id,
            "kind": rev["kind"],
            "id": rev["record_id"],
            "revised_at": to_epoch_ms(rev["revised_at"]),
            "source": rev["source"],
            "live_updated_at": to_epoch_ms(live["updated_at"]) if live else None,
        }
        if dry_run:
            return {"dry_run": True, **summary}
        require_confirm(dry_run=dry_run, confirm=confirm, action="restore_revision")
        result = write_library_data(
            cursor,
            wid,
            kind=rev["kind"],
            record_id=rev["record_id"],
            data=rev["data"] if isinstance(rev["data"], dict) else {"value": rev["data"]},
            tool_name="restore_revision",
            expected_updated_at=to_epoch_ms(live["updated_at"]) if live else None,
        )
        return {"dry_run": False, **summary, **result}

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def list_arguments_snapshots(
    workspace_id: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """List sync_audit_log rows that carried an arguments_snapshot."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        lim = clamp_limit(limit)
        cursor.execute(
            """
            SELECT id, created_at, event, arguments_snapshot
            FROM sync_audit_log
            WHERE workspace_id = %s AND arguments_snapshot IS NOT NULL
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (wid, lim),
        )
        rows = []
        for r in cursor.fetchall():
            snap = r["arguments_snapshot"] or {}
            drafts = (snap.get("draftsBySide") or {}).get("petitioner") or []
            draft_count = sum(
                len((snap.get("draftsBySide") or {}).get(s) or [])
                for s in ("petitioner", "respondent")
            )
            section_count = 0
            char_count = len(str(snap))
            if drafts and isinstance(drafts[0], dict):
                section_count = len(drafts[0].get("sections") or [])
            rows.append(
                {
                    "audit_id": r["id"],
                    "created_at": to_epoch_ms(r["created_at"]),
                    "event": r["event"],
                    "draft_count": draft_count,
                    "section_count": section_count,
                    "character_count": char_count,
                    "preview": _args_preview(snap),
                }
            )
        return {"snapshots": rows}

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False)
)
@tool_guard
async def restore_arguments_snapshot(
    audit_id: int,
    dry_run: bool = True,
    confirm: bool = False,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Restore Arguments board from a sync_audit_log snapshot."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        cursor.execute(
            """
            SELECT id, workspace_id, arguments_snapshot, created_at
            FROM sync_audit_log
            WHERE id = %s
            """,
            (int(audit_id),),
        )
        row = cursor.fetchone()
        if not row or not isinstance(row["arguments_snapshot"], dict):
            raise McpToolError("not_found", "Arguments snapshot not found.")
        if str(row["workspace_id"]) != wid:
            raise McpToolError(
                "forbidden",
                "Snapshot belongs to another workspace.",
                {"workspace_id": str(row["workspace_id"])},
            )
        snap = row["arguments_snapshot"]
        summary = {
            "audit_id": audit_id,
            "created_at": to_epoch_ms(row["created_at"]),
            "preview": _args_preview(snap),
            "character_count": len(str(snap)),
        }
        if dry_run:
            return {"dry_run": True, **summary}
        require_confirm(
            dry_run=dry_run, confirm=confirm, action="restore_arguments_snapshot"
        )
        cursor.execute(
            """
            SELECT updated_at FROM library_records
            WHERE workspace_id = %s AND kind = 'arguments' AND id = 'main'
              AND deleted_at IS NULL
            """,
            (wid,),
        )
        live = cursor.fetchone()
        result = write_library_data(
            cursor,
            wid,
            kind="arguments",
            record_id="main",
            data=snap,
            tool_name="restore_arguments_snapshot",
            expected_updated_at=to_epoch_ms(live["updated_at"]) if live else None,
        )
        return {"dry_run": False, **summary, **result}

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False)
)
@tool_guard
async def copy_workspace(
    from_workspace_id: str,
    to_workspace_id: str,
    dry_run: bool = True,
    confirm: bool = False,
    kinds: list[str] | None = None,
) -> dict[str, Any]:
    """
    Copy rows from an orphaned workspace into another.
    Default conflict policy: newer updated_at wins. Backs up the target first.
    """

    def _sync(cursor) -> dict[str, Any]:
        touch_workspace(cursor, to_workspace_id)
        tables = [
            ("matters", "id"),
            ("cases", "id"),
            ("annotations", "id"),
            ("notes", None),
            ("library_records", None),
            ("documents", "id"),
        ]
        conflicts = []
        would_copy = []
        for table, _ in tables:
            if kinds and table == "library_records":
                cursor.execute(
                    f"""
                    SELECT * FROM {table}
                    WHERE workspace_id = %s AND deleted_at IS NULL
                      AND kind = ANY(%s)
                    """,
                    (from_workspace_id, kinds),
                )
            else:
                cursor.execute(
                    f"""
                    SELECT * FROM {table}
                    WHERE workspace_id = %s AND deleted_at IS NULL
                    """,
                    (from_workspace_id,),
                )
            for row in cursor.fetchall():
                if table == "library_records":
                    key_sql = "kind = %s AND id = %s"
                    key_vals = (row["kind"], row["id"])
                    label = f"{row['kind']}/{row['id']}"
                elif table == "notes":
                    key_sql = "case_id = %s AND layer_id = %s"
                    key_vals = (row["case_id"], row["layer_id"])
                    label = f"notes/{row['case_id']}/{row['layer_id']}"
                else:
                    key_sql = "id = %s"
                    key_vals = (row["id"],)
                    label = f"{table}/{row['id']}"
                cursor.execute(
                    f"""
                    SELECT updated_at FROM {table}
                    WHERE workspace_id = %s AND {key_sql} AND deleted_at IS NULL
                    """,
                    (to_workspace_id, *key_vals),
                )
                live = cursor.fetchone()
                action = "insert"
                if live:
                    if live["updated_at"] >= row["updated_at"]:
                        action = "skip_newer_target"
                        conflicts.append(
                            {
                                "row": label,
                                "action": action,
                                "source_updated_at": to_epoch_ms(row["updated_at"]),
                                "target_updated_at": to_epoch_ms(live["updated_at"]),
                            }
                        )
                    else:
                        action = "overwrite"
                        conflicts.append(
                            {
                                "row": label,
                                "action": action,
                                "source_updated_at": to_epoch_ms(row["updated_at"]),
                                "target_updated_at": to_epoch_ms(live["updated_at"]),
                            }
                        )
                would_copy.append({"table": table, "row": label, "action": action})

        if dry_run:
            return {
                "dry_run": True,
                "from_workspace_id": from_workspace_id,
                "to_workspace_id": to_workspace_id,
                "would_copy": would_copy,
                "conflicts": conflicts,
            }
        require_confirm(dry_run=dry_run, confirm=confirm, action="copy_workspace")
        now = datetime.now(timezone.utc)
        create_workspace_backup(
            cursor,
            to_workspace_id,
            label=f"Before copy from {from_workspace_id[:8]}",
            source="before_restore",
            now=now,
        )
        copied = 0
        for item in would_copy:
            if item["action"] == "skip_newer_target":
                continue
            table = item["table"]
            # Re-fetch source row and upsert.
            # Simplified: re-query by scanning would_copy labels is awkward;
            # redo select for this table batch instead.
            pass

        # Perform copy table by table.
        for table, _ in tables:
            if kinds and table == "library_records":
                cursor.execute(
                    f"SELECT * FROM {table} WHERE workspace_id = %s AND deleted_at IS NULL AND kind = ANY(%s)",
                    (from_workspace_id, kinds),
                )
            else:
                cursor.execute(
                    f"SELECT * FROM {table} WHERE workspace_id = %s AND deleted_at IS NULL",
                    (from_workspace_id,),
                )
            for row in cursor.fetchall():
                if table == "library_records":
                    cursor.execute(
                        """
                        SELECT updated_at FROM library_records
                        WHERE workspace_id = %s AND kind = %s AND id = %s AND deleted_at IS NULL
                        """,
                        (to_workspace_id, row["kind"], row["id"]),
                    )
                    live = cursor.fetchone()
                    if live and live["updated_at"] >= row["updated_at"]:
                        continue
                    cursor.execute(
                        """
                        INSERT INTO library_records (workspace_id, kind, id, data, updated_at, deleted_at)
                        VALUES (%s, %s, %s, %s, %s, NULL)
                        ON CONFLICT (workspace_id, kind, id) DO UPDATE SET
                            data = EXCLUDED.data,
                            updated_at = EXCLUDED.updated_at,
                            deleted_at = NULL
                        """,
                        (
                            to_workspace_id,
                            row["kind"],
                            row["id"],
                            Jsonb(row["data"]),
                            now,
                        ),
                    )
                    copied += 1
                elif table == "notes":
                    cursor.execute(
                        """
                        SELECT updated_at FROM notes
                        WHERE workspace_id = %s AND case_id = %s AND layer_id = %s
                          AND deleted_at IS NULL
                        """,
                        (to_workspace_id, row["case_id"], row["layer_id"]),
                    )
                    live = cursor.fetchone()
                    if live and live["updated_at"] >= row["updated_at"]:
                        continue
                    cursor.execute(
                        """
                        INSERT INTO notes (workspace_id, case_id, layer_id, html, updated_at, deleted_at)
                        VALUES (%s, %s, %s, %s, %s, NULL)
                        ON CONFLICT (workspace_id, case_id, layer_id) DO UPDATE SET
                            html = EXCLUDED.html,
                            updated_at = EXCLUDED.updated_at,
                            deleted_at = NULL
                        """,
                        (
                            to_workspace_id,
                            row["case_id"],
                            row["layer_id"],
                            row["html"],
                            now,
                        ),
                    )
                    copied += 1
                elif table == "matters":
                    cursor.execute(
                        """
                        SELECT updated_at FROM matters
                        WHERE workspace_id = %s AND id = %s AND deleted_at IS NULL
                        """,
                        (to_workspace_id, row["id"]),
                    )
                    live = cursor.fetchone()
                    if live and live["updated_at"] >= row["updated_at"]:
                        continue
                    cursor.execute(
                        """
                        INSERT INTO matters (workspace_id, id, title, season, issues, updated_at, deleted_at)
                        VALUES (%s, %s, %s, %s, %s, %s, NULL)
                        ON CONFLICT (workspace_id, id) DO UPDATE SET
                            title = EXCLUDED.title,
                            season = EXCLUDED.season,
                            issues = EXCLUDED.issues,
                            updated_at = EXCLUDED.updated_at,
                            deleted_at = NULL
                        """,
                        (
                            to_workspace_id,
                            row["id"],
                            row["title"],
                            row["season"],
                            Jsonb(row["issues"]),
                            now,
                        ),
                    )
                    copied += 1
                elif table == "cases":
                    cursor.execute(
                        """
                        SELECT updated_at FROM cases
                        WHERE workspace_id = %s AND id = %s AND deleted_at IS NULL
                        """,
                        (to_workspace_id, row["id"]),
                    )
                    live = cursor.fetchone()
                    if live and live["updated_at"] >= row["updated_at"]:
                        continue
                    cursor.execute(
                        """
                        INSERT INTO cases (
                            workspace_id, id, name, cite, year, issue, tag, usefulness,
                            headline_note, holding, rule, use_petitioner, use_respondent,
                            suggested_file, updated_at, deleted_at
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s,
                            %s, %s, NULL
                        )
                        ON CONFLICT (workspace_id, id) DO UPDATE SET
                            name = EXCLUDED.name,
                            cite = EXCLUDED.cite,
                            year = EXCLUDED.year,
                            issue = EXCLUDED.issue,
                            tag = EXCLUDED.tag,
                            usefulness = EXCLUDED.usefulness,
                            headline_note = EXCLUDED.headline_note,
                            holding = EXCLUDED.holding,
                            rule = EXCLUDED.rule,
                            use_petitioner = EXCLUDED.use_petitioner,
                            use_respondent = EXCLUDED.use_respondent,
                            suggested_file = EXCLUDED.suggested_file,
                            updated_at = EXCLUDED.updated_at,
                            deleted_at = NULL
                        """,
                        (
                            to_workspace_id,
                            row["id"],
                            row["name"],
                            row["cite"],
                            row["year"],
                            row["issue"],
                            row["tag"],
                            row["usefulness"],
                            row.get("headline_note") or "",
                            row["holding"],
                            row["rule"],
                            row["use_petitioner"],
                            row["use_respondent"],
                            row["suggested_file"],
                            now,
                        ),
                    )
                    copied += 1
                elif table == "annotations":
                    cursor.execute(
                        """
                        SELECT updated_at FROM annotations
                        WHERE workspace_id = %s AND id = %s AND deleted_at IS NULL
                        """,
                        (to_workspace_id, row["id"]),
                    )
                    live = cursor.fetchone()
                    if live and live["updated_at"] >= row["updated_at"]:
                        continue
                    cursor.execute(
                        """
                        INSERT INTO annotations (
                            workspace_id, id, case_id, document_id, page, kind, quote, body,
                            rects, pinned, color, topics, updated_at, deleted_at
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, NULL
                        )
                        ON CONFLICT (workspace_id, id) DO UPDATE SET
                            case_id = EXCLUDED.case_id,
                            document_id = EXCLUDED.document_id,
                            page = EXCLUDED.page,
                            kind = EXCLUDED.kind,
                            quote = EXCLUDED.quote,
                            body = EXCLUDED.body,
                            rects = EXCLUDED.rects,
                            pinned = EXCLUDED.pinned,
                            color = EXCLUDED.color,
                            topics = EXCLUDED.topics,
                            updated_at = EXCLUDED.updated_at,
                            deleted_at = NULL
                        """,
                        (
                            to_workspace_id,
                            row["id"],
                            row["case_id"],
                            row["document_id"],
                            row["page"],
                            row["kind"],
                            row["quote"],
                            row["body"],
                            Jsonb(row["rects"]) if row["rects"] is not None else None,
                            row.get("pinned") or False,
                            row.get("color"),
                            Jsonb(row["topics"]) if row.get("topics") is not None else Jsonb([]),
                            now,
                        ),
                    )
                    copied += 1
                elif table == "documents":
                    cursor.execute(
                        """
                        SELECT updated_at FROM documents
                        WHERE workspace_id = %s AND id = %s AND deleted_at IS NULL
                        """,
                        (to_workspace_id, row["id"]),
                    )
                    live = cursor.fetchone()
                    if live and live["updated_at"] >= row["updated_at"]:
                        continue
                    cursor.execute(
                        """
                        INSERT INTO documents (
                            workspace_id, id, case_id, name, size_bytes, content_type,
                            blob_pathname, blob_url, updated_at, deleted_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NULL)
                        ON CONFLICT (workspace_id, id) DO UPDATE SET
                            case_id = EXCLUDED.case_id,
                            name = EXCLUDED.name,
                            size_bytes = EXCLUDED.size_bytes,
                            content_type = EXCLUDED.content_type,
                            blob_pathname = EXCLUDED.blob_pathname,
                            blob_url = EXCLUDED.blob_url,
                            updated_at = EXCLUDED.updated_at,
                            deleted_at = NULL
                        """,
                        (
                            to_workspace_id,
                            row["id"],
                            row["case_id"],
                            row["name"],
                            row["size_bytes"],
                            row["content_type"],
                            row.get("blob_pathname"),
                            row.get("blob_url"),
                            now,
                        ),
                    )
                    copied += 1

        return {
            "dry_run": False,
            "copied": copied,
            "conflicts": conflicts,
            "from_workspace_id": from_workspace_id,
            "to_workspace_id": to_workspace_id,
        }

    return await run_db_async(_sync)
