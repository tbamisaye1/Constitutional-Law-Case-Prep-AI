"""
The single write path for every MCP mutation.

Rules (from the build spec):
1. Lock the target row with SELECT … FOR UPDATE.
2. Optimistic concurrency via expected_updated_at (epoch ms).
3. Archive library_records into revisions with source='mcp'.
4. Write with updated_at = now() so /sync pulls the change.
5. Insert sync_audit_log event='mcp_write'.
6. maybe_auto_backup_workspace like /sync.
7. Bypass seed/fingerprint heuristics in repository.push_changes.
"""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from psycopg.types.json import Jsonb

from app.db.repository import (
    ENTITY_BY_NAME,
    EntitySpec,
    _archive_library_record_before_overwrite,
    maybe_auto_backup_workspace,
    record_sync_audit,
    to_epoch_ms,
    touch_workspace,
)
from app.mcp.errors import McpToolError

_KIND_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")

# Tables the write path may touch. Allowed constant (spec §2.1).
_TABLES = frozenset(
    {
        "matters",
        "cases",
        "documents",
        "annotations",
        "notes",
        "library_records",
        "workspaces",
    }
)


def validate_library_kind(kind: str) -> str:
    if not isinstance(kind, str) or not _KIND_RE.match(kind):
        raise McpToolError(
            "invalid",
            "kind must match ^[a-z][a-z0-9_]{1,40}$",
            {"kind": kind},
        )
    return kind


def _epoch_equals(row_updated_at: datetime, expected_ms: int | None) -> bool:
    if expected_ms is None:
        return False
    return to_epoch_ms(row_updated_at) == int(expected_ms)


def _conflict(current: dict[str, Any], message: str = "Row was modified.") -> None:
    raise McpToolError(
        "conflict",
        message,
        {"current": current},
    )


def _wire_row(spec: EntitySpec, row: dict[str, Any]) -> dict[str, Any]:
    wire: dict[str, Any] = {}
    for field_name, column in spec.all_fields:
        wire[field_name] = row[column]
    wire["updatedAt"] = to_epoch_ms(row["updated_at"])
    wire["deleted"] = row.get("deleted_at") is not None
    return wire


def _lock_row(
    cursor,
    table: str,
    workspace_id: str,
    key_sql: str,
    key_values: tuple[Any, ...],
) -> dict[str, Any] | None:
    cursor.execute(
        f"""
        SELECT *
        FROM {table}
        WHERE workspace_id = %s AND {key_sql}
        FOR UPDATE
        """,
        (workspace_id, *key_values),
    )
    return cursor.fetchone()


def _audit_mcp_write(
    cursor,
    workspace_id: str,
    *,
    tool_name: str,
    kind: str | None,
    record_id: str | None,
    arguments_snapshot: dict[str, Any] | None,
    now: datetime,
) -> None:
    note = f"tool={tool_name}"
    if kind:
        note += f",kind={kind}"
    if record_id:
        note += f",id={record_id}"
    record_sync_audit(
        cursor,
        workspace_id,
        {
            "event": "mcp_write",
            "client_since_ms": None,
            "inbound": {"tool": tool_name},
            "written": {"mcp": 1},
            "row_summaries": [
                {
                    "collection": "mcp",
                    "tool": tool_name,
                    "kind": kind,
                    "id": record_id,
                }
            ],
            "arguments_snapshot": arguments_snapshot,
            "empty_push": False,
            "arguments_unchanged": None,
            "note": note,
        },
        now,
    )


def write_entity(
    cursor,
    workspace_id: str,
    *,
    entity: str,
    key: dict[str, Any],
    values: dict[str, Any],
    tool_name: str,
    expected_updated_at: int | None = None,
    soft_delete: bool = False,
    create_if_missing: bool = True,
) -> dict[str, Any]:
    """
    Upsert or soft-delete one entity row under optimistic concurrency.

    Returns:
        { id, kind?, updated_at, revision_id? } plus wire fields for the row.
    """
    if entity not in ENTITY_BY_NAME:
        raise McpToolError("invalid", f"Unknown entity {entity!r}.")
    spec = ENTITY_BY_NAME[entity]
    if spec.table not in _TABLES:
        raise McpToolError("forbidden", f"Writes to {spec.table} are not allowed.")

    now = datetime.now(timezone.utc)
    touch_workspace(cursor, workspace_id)

    key_cols = [col for _, col in spec.key_fields]
    key_sql = " AND ".join(f"{col} = %s" for col in key_cols)
    key_values = tuple(key[wire] for wire, _ in spec.key_fields)

    existing = _lock_row(cursor, spec.table, workspace_id, key_sql, key_values)

    if existing is None:
        if soft_delete:
            raise McpToolError("not_found", "Row not found.", {"key": key})
        if expected_updated_at is not None:
            raise McpToolError(
                "not_found",
                "Row not found for update.",
                {"key": key},
            )
        if not create_if_missing:
            raise McpToolError("not_found", "Row not found.", {"key": key})
    else:
        current_wire = _wire_row(spec, existing)
        if expected_updated_at is None and not soft_delete:
            # Spec: omitting expected_updated_at allows creates only.
            if existing.get("deleted_at") is None:
                _conflict(
                    current_wire,
                    "expected_updated_at is required to update an existing row.",
                )
        elif expected_updated_at is not None and not _epoch_equals(
            existing["updated_at"], expected_updated_at
        ):
            _conflict(current_wire)

    revision_id: int | None = None
    arguments_snapshot: dict[str, Any] | None = None
    kind_for_audit: str | None = None
    id_for_audit: str | None = None

    if entity == "library_records":
        kind_for_audit = str(key.get("kind") or values.get("kind") or "")
        id_for_audit = str(key.get("id") or values.get("id") or "")
        validate_library_kind(kind_for_audit)
        data = values.get("data")
        if not soft_delete and data is not None:
            revision_id = _archive_library_record_before_overwrite(
                cursor,
                workspace_id,
                {
                    "kind": kind_for_audit,
                    "id": id_for_audit,
                    "data": data,
                },
                now,
                source="mcp",
            )
            if kind_for_audit == "arguments" and isinstance(data, dict):
                arguments_snapshot = data

    # Build column values.
    row_values: dict[str, Any] = {}
    for wire_name, column in spec.all_fields:
        if wire_name in key:
            row_values[column] = key[wire_name]
        elif wire_name in values:
            value = values[wire_name]
            if wire_name in spec.json_fields and value is not None:
                value = Jsonb(value)
            row_values[column] = value
        elif existing is not None:
            row_values[column] = existing[column]
        else:
            # Defaults for required columns on create.
            defaults = {
                "title": "",
                "season": "",
                "issues": Jsonb([]),
                "name": "",
                "cite": "",
                "year": "",
                "usefulness": "background",
                "headline_note": "",
                "holding": "",
                "rule": "",
                "use_petitioner": "",
                "use_respondent": "",
                "suggested_file": "",
                "case_id": "",
                "document_id": None,
                "page": 1,
                "kind": "page",
                "quote": "",
                "body": "",
                "rects": None,
                "pinned": False,
                "color": None,
                "topics": Jsonb([]),
                "html": "",
                "data": Jsonb({}),
                "size_bytes": 0,
                "content_type": "application/pdf",
            }
            if column in defaults:
                row_values[column] = defaults[column]
            else:
                row_values[column] = None

    deleted_at = now if soft_delete else None

    columns = ["workspace_id"] + [col for _, col in spec.all_fields] + [
        "updated_at",
        "deleted_at",
    ]
    placeholders = ", ".join(["%s"] * len(columns))
    conflict_target = ", ".join(["workspace_id"] + key_cols)
    updatable = [col for _, col in spec.value_fields] + ["updated_at", "deleted_at"]
    # For soft delete of notes/cases etc., also allow clearing via deleted_at.
    assignments = ", ".join(f"{col} = EXCLUDED.{col}" for col in updatable)

    insert_values: list[Any] = [workspace_id]
    for _, col in spec.all_fields:
        insert_values.append(row_values[col])
    insert_values.extend([now, deleted_at])

    cursor.execute(
        f"""
        INSERT INTO {spec.table} ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT ({conflict_target}) DO UPDATE SET {assignments}
        RETURNING *
        """,
        tuple(insert_values),
    )
    saved = cursor.fetchone()

    # Auto-backup for critical library kinds.
    if entity == "library_records" and not soft_delete:
        maybe_auto_backup_workspace(cursor, workspace_id, now, reason="mcp_write")

    _audit_mcp_write(
        cursor,
        workspace_id,
        tool_name=tool_name,
        kind=kind_for_audit or (str(key.get("kind")) if "kind" in key else None),
        record_id=id_for_audit
        or str(key.get("id") or key.get("caseId") or key.get("layerId") or ""),
        arguments_snapshot=arguments_snapshot,
        now=now,
    )

    result: dict[str, Any] = {
        "updated_at": to_epoch_ms(saved["updated_at"]),
        "deleted": saved.get("deleted_at") is not None,
        "row": _wire_row(spec, saved),
    }
    if "id" in saved:
        result["id"] = saved["id"]
    if entity == "library_records":
        result["kind"] = saved["kind"]
        result["id"] = saved["id"]
        if revision_id is not None:
            result["revision_id"] = revision_id
    if entity == "notes":
        result["case_id"] = saved["case_id"]
        result["layer_id"] = saved["layer_id"]
    return result


def write_library_data(
    cursor,
    workspace_id: str,
    *,
    kind: str,
    record_id: str,
    data: dict[str, Any],
    tool_name: str,
    expected_updated_at: int | None,
) -> dict[str, Any]:
    """Convenience wrapper for library_records writes."""
    return write_entity(
        cursor,
        workspace_id,
        entity="library_records",
        key={"kind": validate_library_kind(kind), "id": record_id},
        values={"data": deepcopy(data)},
        tool_name=tool_name,
        expected_updated_at=expected_updated_at,
    )


def require_confirm(*, dry_run: bool, confirm: bool, action: str) -> None:
    if dry_run:
        return
    if not confirm:
        raise McpToolError(
            "invalid",
            f"{action} requires dry_run=false and confirm=true.",
        )
