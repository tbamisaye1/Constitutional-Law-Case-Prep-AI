"""Structured Arguments board tools (library kind arguments, id main)."""

from __future__ import annotations

from typing import Any

from mcp.types import ToolAnnotations

from app.db.repository import to_epoch_ms
from app.mcp.arguments_shape import (
    deep_copy_board,
    empty_main_draft,
    find_draft,
    find_prong,
    find_section,
    joined_markdown,
    validate_arguments_board,
    append_scratch_html,
)
from app.mcp.convert import format_note_payload, resolve_write_body
from app.mcp.dbutil import run_db_async, tool_guard
from app.mcp.errors import McpToolError
from app.mcp.ids import new_id
from app.mcp.server import mcp
from app.mcp.workspace import resolve_workspace_id
from app.mcp.writes import require_confirm, write_library_data

_BOARD_KIND = "arguments"
_BOARD_ID = "main"


def _load_board(cursor, workspace_id: str) -> tuple[dict[str, Any], int | None]:
    cursor.execute(
        """
        SELECT data, updated_at
        FROM library_records
        WHERE workspace_id = %s AND kind = %s AND id = %s AND deleted_at IS NULL
        FOR UPDATE
        """,
        (workspace_id, _BOARD_KIND, _BOARD_ID),
    )
    row = cursor.fetchone()
    if not row:
        board = {
            "draftsBySide": {
                "petitioner": [empty_main_draft("petitioner")],
                "respondent": [empty_main_draft("respondent")],
            },
            "activeDraftBySide": {
                "petitioner": "petitioner-main",
                "respondent": "respondent-main",
            },
            "activeSectionBySide": {"petitioner": None, "respondent": None},
            "activeFocusBySide": {
                "petitioner": {"type": "side"},
                "respondent": {"type": "side"},
            },
        }
        return board, None
    data = row["data"] if isinstance(row["data"], dict) else {}
    return data, to_epoch_ms(row["updated_at"])


def _save_board(
    cursor,
    workspace_id: str,
    board: dict[str, Any],
    *,
    tool_name: str,
    expected_updated_at: int | None,
) -> dict[str, Any]:
    validate_arguments_board(board)
    return write_library_data(
        cursor,
        workspace_id,
        kind=_BOARD_KIND,
        record_id=_BOARD_ID,
        data=board,
        tool_name=tool_name,
        expected_updated_at=expected_updated_at,
    )


def _require_version(current: int | None, expected: int | None, board: dict) -> None:
    if current is None:
        # Creating the board: expected must be omitted.
        if expected is not None:
            raise McpToolError(
                "not_found",
                "Arguments board did not exist; omit expected_updated_at to create it.",
            )
        return
    if expected is None:
        raise McpToolError(
            "conflict",
            "expected_updated_at is required to update the Arguments board.",
            {"current": {"data": board, "updatedAt": current}},
        )
    if int(expected) != int(current):
        raise McpToolError(
            "conflict",
            "Row was modified.",
            {"current": {"data": board, "updatedAt": current}},
        )


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)
)
@tool_guard
async def get_arguments(
    side: str | None = None,
    draft_id: str | None = None,
    format: str = "markdown",
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Nested outline with IDs, plus joined_markdown for the selected draft."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        board, updated_at = _load_board(cursor, wid)
        if updated_at is None:
            return {
                "exists": False,
                "updated_at": None,
                "hint": "No Arguments board yet. create_draft or upsert_prong will create one.",
            }
        drafts_by_side = board.get("draftsBySide") or {}
        sides = [side] if side else ["petitioner", "respondent"]
        outline: dict[str, Any] = {}
        joined = None
        for s in sides:
            drafts = drafts_by_side.get(s) or []
            if draft_id:
                drafts = [d for d in drafts if isinstance(d, dict) and d.get("id") == draft_id]
            rendered = []
            for draft in drafts:
                if not isinstance(draft, dict):
                    continue
                entry = {
                    "id": draft.get("id"),
                    "name": draft.get("name"),
                    "notes": format_note_payload(
                        draft.get("notes") or "", format=format  # type: ignore[arg-type]
                    ),
                    # Free-form side notes from the Arguments page scratch pane.
                    "scratch": format_note_payload(
                        draft.get("scratch") or "", format=format  # type: ignore[arg-type]
                    ),
                    "sections": [],
                }
                for section in draft.get("sections") or []:
                    if not isinstance(section, dict):
                        continue
                    sec = {
                        "id": section.get("id"),
                        "title": section.get("title"),
                        "notes": format_note_payload(
                            section.get("notes") or "", format=format  # type: ignore[arg-type]
                        ),
                        "prongs": [],
                    }
                    for prong in section.get("prongs") or []:
                        if not isinstance(prong, dict):
                            continue
                        sec["prongs"].append(
                            {
                                "id": prong.get("id"),
                                "title": prong.get("title"),
                                "notes": format_note_payload(
                                    prong.get("notes") or "",
                                    format=format,  # type: ignore[arg-type]
                                ),
                            }
                        )
                    entry["sections"].append(sec)
                rendered.append(entry)
                if joined is None and (not draft_id or draft.get("id") == draft_id):
                    joined = joined_markdown(draft)
            outline[s] = rendered
        return {
            "exists": True,
            "updated_at": updated_at,
            "activeDraftBySide": board.get("activeDraftBySide"),
            "outline": outline,
            "joined_markdown": joined,
        }

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def create_draft(
    side: str,
    name: str,
    expected_updated_at: int | None = None,
    copy_from_draft_id: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Create a new draft on one side; optionally copy outline from another draft."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        board, current = _load_board(cursor, wid)
        _require_version(current, expected_updated_at, board)
        next_board = deep_copy_board(board)
        drafts = next_board.setdefault("draftsBySide", {}).setdefault(side, [])
        if copy_from_draft_id:
            source = find_draft(next_board, side, copy_from_draft_id)
            new_draft = deep_copy_board(source)
            new_draft["id"] = new_id("draft")
            new_draft["name"] = name
        else:
            new_draft = {
                "id": new_id("draft"),
                "name": name,
                "notes": "",
                "sections": [
                    {
                        "id": new_id("sec"),
                        "title": "New section",
                        "notes": "",
                        "prongs": [],
                    }
                ],
            }
        drafts.append(new_draft)
        result = _save_board(
            cursor,
            wid,
            next_board,
            tool_name="create_draft",
            expected_updated_at=expected_updated_at if current is not None else None,
        )
        result["draft_id"] = new_draft["id"]
        return result

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def set_draft_notes(
    side: str,
    draft_id: str,
    expected_updated_at: int,
    markdown: str | None = None,
    html: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Set whole-argument notes on a draft."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        try:
            body = resolve_write_body(markdown=markdown, html=html)
        except ValueError as exc:
            raise McpToolError("invalid", str(exc)) from exc
        board, current = _load_board(cursor, wid)
        _require_version(current, expected_updated_at, board)
        next_board = deep_copy_board(board)
        draft = find_draft(next_board, side, draft_id)
        draft["notes"] = body
        return _save_board(
            cursor,
            wid,
            next_board,
            tool_name="set_draft_notes",
            expected_updated_at=expected_updated_at,
        )

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def append_to_draft_scratch(
    side: str,
    draft_id: str,
    markdown: str | None = None,
    html: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """
    Append a note to the end of a draft's scratch pane (the messy-notes column
    beside the Arguments page).

    No expected_updated_at: appending is safe to replay on the latest board
    (the row is locked FOR UPDATE), so an open browser tab cannot make this
    conflict. Use set_draft_scratch to replace the whole scratch instead.
    """

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        try:
            body = resolve_write_body(markdown=markdown, html=html)
        except ValueError as exc:
            raise McpToolError("invalid", str(exc)) from exc
        board, current = _load_board(cursor, wid)
        if current is None:
            raise McpToolError("not_found", "No Arguments board yet.")
        next_board = deep_copy_board(board)
        draft = find_draft(next_board, side, draft_id)
        append_scratch_html(draft, body)
        return _save_board(
            cursor,
            wid,
            next_board,
            tool_name="append_to_draft_scratch",
            expected_updated_at=current,
        )

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def set_draft_scratch(
    side: str,
    draft_id: str,
    expected_updated_at: int,
    markdown: str | None = None,
    html: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Replace a draft's scratch notes (read with get_arguments first)."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        try:
            body = resolve_write_body(markdown=markdown, html=html)
        except ValueError as exc:
            raise McpToolError("invalid", str(exc)) from exc
        board, current = _load_board(cursor, wid)
        _require_version(current, expected_updated_at, board)
        next_board = deep_copy_board(board)
        draft = find_draft(next_board, side, draft_id)
        draft["scratch"] = body
        return _save_board(
            cursor,
            wid,
            next_board,
            tool_name="set_draft_scratch",
            expected_updated_at=expected_updated_at,
        )

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def upsert_section(
    side: str,
    draft_id: str,
    expected_updated_at: int | None = None,
    section_id: str | None = None,
    title: str = "Untitled section",
    notes_markdown: str | None = None,
    notes_html: str | None = None,
    position: int | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Create or update a section. Omit section_id to create."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        board, current = _load_board(cursor, wid)
        _require_version(current, expected_updated_at, board)
        next_board = deep_copy_board(board)
        draft = find_draft(next_board, side, draft_id)
        sections = draft.setdefault("sections", [])
        notes_body = None
        if notes_markdown is not None or notes_html is not None:
            try:
                notes_body = resolve_write_body(
                    markdown=notes_markdown, html=notes_html
                )
            except ValueError as exc:
                raise McpToolError("invalid", str(exc)) from exc
        if section_id:
            section = find_section(draft, section_id)
            section["title"] = title
            if notes_body is not None:
                section["notes"] = notes_body
            sid = section_id
        else:
            section = {
                "id": new_id("sec"),
                "title": title,
                "notes": notes_body or "",
                "prongs": [],
            }
            if position is None or position >= len(sections):
                sections.append(section)
            else:
                sections.insert(max(0, position), section)
            sid = section["id"]
        if position is not None and section_id:
            current_idx = next(
                i for i, s in enumerate(sections) if s.get("id") == section_id
            )
            item = sections.pop(current_idx)
            sections.insert(max(0, min(position, len(sections))), item)
        result = _save_board(
            cursor,
            wid,
            next_board,
            tool_name="upsert_section",
            expected_updated_at=expected_updated_at if current is not None else None,
        )
        result["section_id"] = sid
        return result

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def upsert_prong(
    side: str,
    draft_id: str,
    section_id: str,
    expected_updated_at: int | None = None,
    prong_id: str | None = None,
    title: str = "Untitled prong",
    notes_markdown: str | None = None,
    notes_html: str | None = None,
    position: int | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Create or update a prong. Omit prong_id to create."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        board, current = _load_board(cursor, wid)
        _require_version(current, expected_updated_at, board)
        next_board = deep_copy_board(board)
        draft = find_draft(next_board, side, draft_id)
        section = find_section(draft, section_id)
        prongs = section.setdefault("prongs", [])
        notes_body = None
        if notes_markdown is not None or notes_html is not None:
            try:
                notes_body = resolve_write_body(
                    markdown=notes_markdown, html=notes_html
                )
            except ValueError as exc:
                raise McpToolError("invalid", str(exc)) from exc
        if prong_id:
            prong = find_prong(section, prong_id)
            prong["title"] = title
            if notes_body is not None:
                prong["notes"] = notes_body
            pid = prong_id
        else:
            prong = {
                "id": new_id("pr"),
                "title": title,
                "notes": notes_body or "",
            }
            if position is None or position >= len(prongs):
                prongs.append(prong)
            else:
                prongs.insert(max(0, position), prong)
            pid = prong["id"]
        if position is not None and prong_id:
            current_idx = next(i for i, p in enumerate(prongs) if p.get("id") == prong_id)
            item = prongs.pop(current_idx)
            prongs.insert(max(0, min(position, len(prongs))), item)
        result = _save_board(
            cursor,
            wid,
            next_board,
            tool_name="upsert_prong",
            expected_updated_at=expected_updated_at if current is not None else None,
        )
        result["prong_id"] = pid
        return result

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)
)
@tool_guard
async def move_item(
    side: str,
    draft_id: str,
    section_id: str,
    to_position: int,
    expected_updated_at: int,
    prong_id: str | None = None,
    to_section_id: str | None = None,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Reorder a section or prong; optionally move a prong to another section."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        board, current = _load_board(cursor, wid)
        _require_version(current, expected_updated_at, board)
        next_board = deep_copy_board(board)
        draft = find_draft(next_board, side, draft_id)
        if prong_id:
            source_section = find_section(draft, section_id)
            prong = find_prong(source_section, prong_id)
            source_section["prongs"] = [
                p for p in source_section.get("prongs") or [] if p.get("id") != prong_id
            ]
            dest = find_section(draft, to_section_id or section_id)
            dest_prongs = dest.setdefault("prongs", [])
            dest_prongs.insert(max(0, min(to_position, len(dest_prongs))), prong)
        else:
            sections = draft.setdefault("sections", [])
            idx = next(i for i, s in enumerate(sections) if s.get("id") == section_id)
            item = sections.pop(idx)
            sections.insert(max(0, min(to_position, len(sections))), item)
        return _save_board(
            cursor,
            wid,
            next_board,
            tool_name="move_item",
            expected_updated_at=expected_updated_at,
        )

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False)
)
@tool_guard
async def delete_item(
    side: str,
    draft_id: str,
    section_id: str,
    expected_updated_at: int,
    prong_id: str | None = None,
    dry_run: bool = True,
    confirm: bool = False,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """Delete a section or prong from the board."""

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        board, current = _load_board(cursor, wid)
        _require_version(current, expected_updated_at, board)
        draft = find_draft(board, side, draft_id)
        if prong_id:
            section = find_section(draft, section_id)
            prong = find_prong(section, prong_id)
            summary = {"type": "prong", "id": prong_id, "title": prong.get("title")}
        else:
            section = find_section(draft, section_id)
            summary = {
                "type": "section",
                "id": section_id,
                "title": section.get("title"),
                "prong_count": len(section.get("prongs") or []),
            }
        if dry_run:
            return {"dry_run": True, **summary}
        require_confirm(dry_run=dry_run, confirm=confirm, action="delete_item")
        next_board = deep_copy_board(board)
        draft = find_draft(next_board, side, draft_id)
        if prong_id:
            section = find_section(draft, section_id)
            section["prongs"] = [
                p for p in section.get("prongs") or [] if p.get("id") != prong_id
            ]
        else:
            draft["sections"] = [
                s for s in draft.get("sections") or [] if s.get("id") != section_id
            ]
        result = _save_board(
            cursor,
            wid,
            next_board,
            tool_name="delete_item",
            expected_updated_at=expected_updated_at,
        )
        return {"dry_run": False, **summary, **result}

    return await run_db_async(_sync)


@mcp.tool(
    annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False)
)
@tool_guard
async def replace_draft_outline(
    side: str,
    draft_id: str,
    outline: Any,
    expected_updated_at: int,
    dry_run: bool = True,
    confirm: bool = False,
    workspace_id: str | None = None,
) -> dict[str, Any]:
    """
    Bulk rebuild a draft outline from nested JSON or markdown (# / ## / ###).

    Dry run returns a structural diff.
    """

    def _parse_markdown_outline(text: str) -> list[dict[str, Any]]:
        sections: list[dict[str, Any]] = []
        current_sec: dict[str, Any] | None = None
        current_prong: dict[str, Any] | None = None
        for line in text.splitlines():
            if line.startswith("### "):
                title = line[4:].strip()
                current_prong = {
                    "id": new_id("pr"),
                    "title": title,
                    "notes": "",
                }
                if current_sec is None:
                    current_sec = {
                        "id": new_id("sec"),
                        "title": "Section",
                        "notes": "",
                        "prongs": [],
                    }
                    sections.append(current_sec)
                current_sec["prongs"].append(current_prong)
            elif line.startswith("## "):
                current_sec = {
                    "id": new_id("sec"),
                    "title": line[3:].strip(),
                    "notes": "",
                    "prongs": [],
                }
                sections.append(current_sec)
                current_prong = None
            elif line.startswith("# "):
                continue
            else:
                if current_prong is not None:
                    current_prong["notes"] += (line + "\n")
                elif current_sec is not None:
                    current_sec["notes"] += (line + "\n")
        return sections

    def _sync(cursor) -> dict[str, Any]:
        wid = resolve_workspace_id(cursor, workspace_id)
        board, current = _load_board(cursor, wid)
        _require_version(current, expected_updated_at, board)
        draft = find_draft(board, side, draft_id)
        old_titles = [
            s.get("title") for s in draft.get("sections") or [] if isinstance(s, dict)
        ]
        if isinstance(outline, str):
            new_sections = _parse_markdown_outline(outline)
        elif isinstance(outline, list):
            new_sections = outline
        elif isinstance(outline, dict) and isinstance(outline.get("sections"), list):
            new_sections = outline["sections"]
        else:
            raise McpToolError(
                "invalid",
                "outline must be markdown, a sections list, or {sections: [...]}.",
            )
        # Ensure ids on nested objects.
        for section in new_sections:
            if not isinstance(section, dict):
                raise McpToolError("invalid", "Each section must be an object.")
            section.setdefault("id", new_id("sec"))
            section.setdefault("notes", "")
            section.setdefault("prongs", [])
            for prong in section["prongs"]:
                if not isinstance(prong, dict):
                    raise McpToolError("invalid", "Each prong must be an object.")
                prong.setdefault("id", new_id("pr"))
                prong.setdefault("notes", "")
        new_titles = [s.get("title") for s in new_sections]
        diff = {
            "old_section_titles": old_titles,
            "new_section_titles": new_titles,
            "old_section_count": len(old_titles),
            "new_section_count": len(new_titles),
        }
        if dry_run:
            return {"dry_run": True, "diff": diff}
        require_confirm(
            dry_run=dry_run, confirm=confirm, action="replace_draft_outline"
        )
        next_board = deep_copy_board(board)
        target = find_draft(next_board, side, draft_id)
        target["sections"] = new_sections
        result = _save_board(
            cursor,
            wid,
            next_board,
            tool_name="replace_draft_outline",
            expected_updated_at=expected_updated_at,
        )
        return {"dry_run": False, "diff": diff, **result}

    return await run_db_async(_sync)
