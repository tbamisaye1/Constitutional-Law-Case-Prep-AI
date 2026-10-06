"""Notebook tools (library kind notebook, id main). Shape: tree + pagesBySection."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from mcp.types import ToolAnnotations

from app.db.repository import to_epoch_ms
from app.mcp.convert import format_note_payload, resolve_write_body
from app.mcp.dbutil import run_db_async, tool_guard
from app.mcp.errors import McpToolError
from app.mcp.ids import new_id
from app.mcp.server import mcp
from app.mcp.workspace import resolve_workspace_id
from app.mcp.writes import write_library_data

_KIND = "notebook"
_ID = "main"


def _load_notebook(cursor, workspace_id: str) -> tuple[dict[str, Any], int | None]:
    cursor.execute(
        """
        SELECT data, updated_at
        FROM library_records
        WHERE workspace_id = %s AND kind = %s AND id = %s AND deleted_at IS NULL
        FOR UPDATE
        """,
        (workspace_id, _KIND, _ID),
    )
    row = cursor.fetchone()
    if not row:
        return {"tree": [], "pagesBySection": {}}, None
    data = row["data"] if isinstance(row["data"], dict) else {}
    if "tree" not in data or "pagesBySection" not in data:
        raise McpToolError(
            "invalid",
            "Notebook row is missing tree or pagesBySection.",
        )
    return data, to_epoch_ms(row["updated_at"])


def _save(
    cursor,
    workspace_id: str,
    data: dict[str, Any],
    *,
    tool_name: str,
    expected_updated_at: int | None,
) -> dict[str, Any]:
    if not isinstance(data.get("tree"), list) or not isinstance(
        data.get("pagesBySection"), dict
    ):
        raise McpToolError(
            "invalid",
            "Notebook must have tree (list) and pagesBySection (object).",
        )
    return write_library_data(
        cursor,
        workspace_id,
        kind=_KIND,
        record_id=_ID,
        data=data,
        tool_name=tool_name,
        expected_updated_at=expected_updated_at,
    )


def _find_page(
    pages_by_section: dict[str, Any], section_id: str, page_id: str
) -> dict[str, Any] | None:
    pages = pages_by_section.get(section_id) or []
    if not isinstance(pages, list):
        return None

    def walk(nodes: list) -> dict[str, Any] | None:
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if node.get("id") == page_id:
                return node
            children = node.get("children")
            if isinstance(children, list):
                found = walk(children)
                if found:
                    return found
        return None

    return walk(pages)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def get_notebook_tree(workspace_id: str | None = None) -> dict[str, Any]:
    """Return the notebook section-group tree and page id lists per section."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        data, updated_at = _load_notebook(cursor, wid)
        pages_by_section = data.get("pagesBySection") or {}
        page_index = {
            sec_id: [
                {"id": p.get("id"), "title": p.get("title")}
                for p in (pages or [])
                if isinstance(p, dict)
            ]
            for sec_id, pages in pages_by_section.items()
        }
        return {
            "updated_at": updated_at,
            "tree": data.get("tree") or [],
            "pages": page_index,
        }

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def get_notebook_page(
    section_id: str,
    page_id: str,
    format: str = "markdown",
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Read one notebook page."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        data, updated_at = _load_notebook(cursor, wid)
        page = _find_page(data.get("pagesBySection") or {}, section_id, page_id)
        if not page:
            raise McpToolError(
                "not_found",
                "Notebook page not found.",
                {"section_id": section_id, "page_id": page_id},
            )
        payload = format_note_payload(page.get("html") or "", format=format)  # type: ignore[arg-type]
        return {
            "section_id": section_id,
            "page_id": page_id,
            "title": page.get("title"),
            "updated_at": updated_at,
            **payload,
        }

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def upsert_notebook_page(
    section_id: str,
    expected_updated_at: int | None = None,
    page_id: str | None = None,
    title: str = "Untitled",
    markdown: str | None = None,
    html: str | None = None,
    parent_page_id: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Create or update a notebook page."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        data, current = _load_notebook(cursor, wid)
        if current is not None and expected_updated_at is None:
            raise McpToolError(
                "conflict",
                "expected_updated_at is required to update the notebook.",
                {"current": {"updatedAt": current}},
            )
        if current is not None and int(expected_updated_at) != int(current):
            raise McpToolError(
                "conflict",
                "Row was modified.",
                {"current": {"data": data, "updatedAt": current}},
            )
        body = None
        if markdown is not None or html is not None:
            try:
                body = resolve_write_body(markdown=markdown, html=html)
            except ValueError as exc:
                raise McpToolError("invalid", str(exc)) from exc
        next_data = deepcopy(data)
        pages_by_section = next_data.setdefault("pagesBySection", {})
        pages = pages_by_section.setdefault(section_id, [])
        if not isinstance(pages, list):
            pages = []
            pages_by_section[section_id] = pages
        if page_id:
            page = _find_page(pages_by_section, section_id, page_id)
            if not page:
                raise McpToolError(
                    "not_found",
                    "Notebook page not found.",
                    {"page_id": page_id},
                )
            page["title"] = title
            if body is not None:
                page["html"] = body
            pid = page_id
        else:
            page = {
                "id": new_id("pg"),
                "title": title,
                "html": body or "",
            }
            if parent_page_id:
                parent = _find_page(pages_by_section, section_id, parent_page_id)
                if not parent:
                    raise McpToolError(
                        "not_found",
                        "Parent page not found.",
                        {"parent_page_id": parent_page_id},
                    )
                parent.setdefault("children", []).append(page)
            else:
                pages.append(page)
            pid = page["id"]
        result = _save(
            cursor,
            wid,
            next_data,
            tool_name="upsert_notebook_page",
            expected_updated_at=expected_updated_at if current is not None else None,
        )
        result["page_id"] = pid
        return result

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def upsert_notebook_section(
    name: str,
    expected_updated_at: int | None = None,
    section_id: str | None = None,
    parent_group_id: str | None = None,
    color: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Create or rename a notebook section in the tree."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        data, current = _load_notebook(cursor, wid)
        if current is not None and expected_updated_at is None:
            raise McpToolError(
                "conflict",
                "expected_updated_at is required to update the notebook.",
            )
        if current is not None and int(expected_updated_at) != int(current):
            raise McpToolError(
                "conflict",
                "Row was modified.",
                {"current": {"updatedAt": current}},
            )
        next_data = deepcopy(data)
        tree = next_data.setdefault("tree", [])

        def find_node(nodes: list, node_id: str) -> dict[str, Any] | None:
            for node in nodes:
                if not isinstance(node, dict):
                    continue
                if node.get("id") == node_id:
                    return node
                children = node.get("children")
                if isinstance(children, list):
                    found = find_node(children, node_id)
                    if found:
                        return found
            return None

        if section_id:
            node = find_node(tree, section_id)
            if not node:
                raise McpToolError(
                    "not_found",
                    "Section not found.",
                    {"section_id": section_id},
                )
            node["name"] = name
            if color:
                node["color"] = color
            sid = section_id
        else:
            section = {
                "id": new_id("sec"),
                "name": name,
                "kind": "section",
            }
            if color:
                section["color"] = color
            if parent_group_id:
                parent = find_node(tree, parent_group_id)
                if not parent:
                    raise McpToolError(
                        "not_found",
                        "Parent group not found.",
                        {"parent_group_id": parent_group_id},
                    )
                parent.setdefault("children", []).append(section)
            else:
                tree.append(section)
            next_data.setdefault("pagesBySection", {}).setdefault(section["id"], [])
            sid = section["id"]
        result = _save(
            cursor,
            wid,
            next_data,
            tool_name="upsert_notebook_section",
            expected_updated_at=expected_updated_at if current is not None else None,
        )
        result["section_id"] = sid
        return result

    return await run_db_async(_sync)
