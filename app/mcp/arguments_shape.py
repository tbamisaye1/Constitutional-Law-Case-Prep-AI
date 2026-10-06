"""
Shape validation for the Arguments board (validation only, no seeding).

Mirrors the structural expectations of normalizeArgumentsBoard without
inventing outline or note content.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from app.mcp.errors import McpToolError
from app.mcp.ids import new_id

_SIDES = ("petitioner", "respondent")


def validate_arguments_board(data: Any) -> dict[str, Any]:
    """
    Ensure the board has draftsBySide with draft/section/prong objects.

    Preserves unknown top-level keys. Raises McpToolError on invalid shape.
    """
    if not isinstance(data, dict):
        raise McpToolError("invalid", "Arguments board must be an object.")

    drafts = data.get("draftsBySide")
    if not isinstance(drafts, dict):
        raise McpToolError("invalid", "draftsBySide is required.")

    for side in _SIDES:
        side_drafts = drafts.get(side)
        if not isinstance(side_drafts, list) or not side_drafts:
            raise McpToolError(
                "invalid",
                f"draftsBySide.{side} must be a non-empty list.",
            )
        for draft in side_drafts:
            _validate_draft(draft, side)

    # Preserve every key the MCP does not understand.
    return data


def _validate_draft(draft: Any, side: str) -> None:
    if not isinstance(draft, dict):
        raise McpToolError("invalid", f"Draft on {side} must be an object.")
    if not isinstance(draft.get("id"), str) or not draft["id"]:
        raise McpToolError("invalid", f"Draft on {side} needs a string id.")
    if not isinstance(draft.get("name"), str):
        raise McpToolError("invalid", f"Draft {draft.get('id')} needs a name.")
    if not isinstance(draft.get("notes"), str):
        draft["notes"] = draft.get("notes") or ""
        if not isinstance(draft["notes"], str):
            raise McpToolError("invalid", "Draft notes must be a string.")
    sections = draft.get("sections")
    if not isinstance(sections, list):
        raise McpToolError("invalid", f"Draft {draft['id']} needs sections[].")
    for section in sections:
        _validate_section(section)


def _validate_section(section: Any) -> None:
    if not isinstance(section, dict):
        raise McpToolError("invalid", "Section must be an object.")
    if not isinstance(section.get("id"), str) or not section["id"]:
        raise McpToolError("invalid", "Section needs a string id.")
    if not isinstance(section.get("title"), str):
        raise McpToolError("invalid", f"Section {section.get('id')} needs a title.")
    if "notes" in section and section["notes"] is not None and not isinstance(
        section["notes"], str
    ):
        raise McpToolError("invalid", "Section notes must be a string.")
    prongs = section.get("prongs")
    if prongs is None:
        section["prongs"] = []
        prongs = section["prongs"]
    if not isinstance(prongs, list):
        raise McpToolError("invalid", "Section prongs must be a list.")
    for prong in prongs:
        if not isinstance(prong, dict):
            raise McpToolError("invalid", "Prong must be an object.")
        if not isinstance(prong.get("id"), str) or not prong["id"]:
            raise McpToolError("invalid", "Prong needs a string id.")
        if not isinstance(prong.get("title"), str):
            raise McpToolError("invalid", f"Prong {prong.get('id')} needs a title.")
        if "notes" in prong and prong["notes"] is not None and not isinstance(
            prong["notes"], str
        ):
            raise McpToolError("invalid", "Prong notes must be a string.")


def find_draft(board: dict[str, Any], side: str, draft_id: str) -> dict[str, Any]:
    if side not in _SIDES:
        raise McpToolError("invalid", "side must be petitioner or respondent.")
    for draft in board.get("draftsBySide", {}).get(side) or []:
        if isinstance(draft, dict) and draft.get("id") == draft_id:
            return draft
    raise McpToolError(
        "not_found",
        "Draft not found.",
        {"side": side, "draft_id": draft_id},
    )


def find_section(draft: dict[str, Any], section_id: str) -> dict[str, Any]:
    for section in draft.get("sections") or []:
        if isinstance(section, dict) and section.get("id") == section_id:
            return section
    raise McpToolError(
        "not_found",
        "Section not found.",
        {"section_id": section_id},
    )


def find_prong(section: dict[str, Any], prong_id: str) -> dict[str, Any]:
    for prong in section.get("prongs") or []:
        if isinstance(prong, dict) and prong.get("id") == prong_id:
            return prong
    raise McpToolError("not_found", "Prong not found.", {"prong_id": prong_id})


def joined_markdown(draft: dict[str, Any]) -> str:
    """Render a draft as Intro → 1. Section → a. Prong outline."""
    lines: list[str] = []
    name = draft.get("name") or "Draft"
    lines.append(f"# {name}")
    notes = (draft.get("notes") or "").strip()
    if notes:
        lines.append("")
        lines.append(notes)
    for index, section in enumerate(draft.get("sections") or [], start=1):
        if not isinstance(section, dict):
            continue
        lines.append("")
        lines.append(f"## {index}. {section.get('title') or 'Section'}")
        sec_notes = (section.get("notes") or "").strip()
        if sec_notes:
            lines.append("")
            lines.append(sec_notes)
        for p_index, prong in enumerate(section.get("prongs") or []):
            if not isinstance(prong, dict):
                continue
            letter = chr(ord("a") + p_index) if p_index < 26 else str(p_index + 1)
            lines.append("")
            lines.append(f"### {letter}. {prong.get('title') or 'Prong'}")
            pr_notes = (prong.get("notes") or "").strip()
            if pr_notes:
                lines.append("")
                lines.append(pr_notes)
    return "\n".join(lines).strip() + "\n"


def deep_copy_board(data: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(data)


def empty_main_draft(side: str) -> dict[str, Any]:
    """Minimal Main draft with one empty section (no seed text)."""
    return {
        "id": f"{side}-main",
        "name": "Main",
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
