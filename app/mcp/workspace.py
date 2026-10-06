"""
Workspace resolution for MCP tools.

Order: explicit argument → app_settings → MCP_DEFAULT_WORKSPACE_ID env → error.
"""

from __future__ import annotations

import uuid
from typing import Any

from psycopg.types.json import Jsonb

from app.config import get_settings
from app.db.repository import to_epoch_ms
from app.mcp.errors import McpToolError

DEFAULT_WORKSPACE_KEY = "mcp_default_workspace"


def _as_uuid(value: str) -> str:
    try:
        return str(uuid.UUID(str(value).strip()))
    except (ValueError, AttributeError, TypeError) as exc:
        raise McpToolError(
            "invalid",
            "workspace_id must be a valid UUID.",
            {"workspace_id": value},
        ) from exc


def workspace_exists(cursor, workspace_id: str) -> bool:
    cursor.execute("SELECT 1 FROM workspaces WHERE id = %s", (workspace_id,))
    return cursor.fetchone() is not None


def get_setting(cursor, key: str) -> Any | None:
    cursor.execute(
        "SELECT value FROM app_settings WHERE key = %s",
        (key,),
    )
    row = cursor.fetchone()
    if not row:
        return None
    return row["value"]


def set_setting(cursor, key: str, value: Any) -> dict[str, Any]:
    cursor.execute(
        """
        INSERT INTO app_settings (key, value, updated_at)
        VALUES (%s, %s, now())
        ON CONFLICT (key) DO UPDATE SET
            value = EXCLUDED.value,
            updated_at = now()
        RETURNING key, value, updated_at
        """,
        (key, Jsonb(value)),
    )
    row = cursor.fetchone()
    return {
        "key": row["key"],
        "value": row["value"],
        "updated_at": to_epoch_ms(row["updated_at"]),
    }


def resolve_workspace_id(cursor, workspace_id: str | None = None) -> str:
    """
    Resolve the workspace for a tool call.

    Raises:
        McpToolError: no_workspace / invalid / not_found
    """
    if workspace_id:
        wid = _as_uuid(workspace_id)
        if not workspace_exists(cursor, wid):
            raise McpToolError(
                "not_found",
                "Workspace not found.",
                {"workspace_id": wid},
            )
        return wid

    stored = get_setting(cursor, DEFAULT_WORKSPACE_KEY)
    if isinstance(stored, str) and stored.strip():
        wid = _as_uuid(stored)
        if workspace_exists(cursor, wid):
            return wid
    elif isinstance(stored, dict) and isinstance(stored.get("id"), str):
        wid = _as_uuid(stored["id"])
        if workspace_exists(cursor, wid):
            return wid

    env_default = get_settings().mcp_default_workspace_id.strip()
    if env_default:
        wid = _as_uuid(env_default)
        if workspace_exists(cursor, wid):
            return wid

    raise McpToolError(
        "no_workspace",
        "No workspace selected. Call list_workspaces, then set_default_workspace.",
    )


def set_default_workspace(cursor, workspace_id: str) -> dict[str, Any]:
    wid = _as_uuid(workspace_id)
    if not workspace_exists(cursor, wid):
        raise McpToolError(
            "not_found",
            "Workspace not found.",
            {"workspace_id": wid},
        )
    return set_setting(cursor, DEFAULT_WORKSPACE_KEY, wid)


def label_workspace(cursor, workspace_id: str, label: str) -> dict[str, Any]:
    wid = _as_uuid(workspace_id)
    if not workspace_exists(cursor, wid):
        raise McpToolError(
            "not_found",
            "Workspace not found.",
            {"workspace_id": wid},
        )
    cursor.execute(
        """
        UPDATE workspaces
        SET label = %s
        WHERE id = %s
        RETURNING id, label, created_at, last_seen_at
        """,
        ((label or "")[:200], wid),
    )
    row = cursor.fetchone()
    return {
        "id": str(row["id"]),
        "label": row["label"],
        "created_at": to_epoch_ms(row["created_at"]),
        "last_seen_at": to_epoch_ms(row["last_seen_at"]),
    }
