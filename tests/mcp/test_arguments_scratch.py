"""Scratch helpers for the Arguments page (pure, no database)."""

import pytest

from app.mcp.arguments_shape import append_scratch_html, validate_arguments_board
from app.mcp.errors import McpToolError


def _board(draft):
    return {
        "draftsBySide": {
            "petitioner": [draft],
            "respondent": [{"id": "respondent-main", "name": "Main", "notes": "", "sections": []}],
        }
    }


def test_append_to_missing_scratch():
    draft = {"id": "d", "name": "D", "notes": "", "sections": []}
    append_scratch_html(draft, "<p>one</p>")
    assert draft["scratch"] == "<p>one</p>"


def test_append_treats_empty_editor_doc_as_empty():
    draft = {"id": "d", "name": "D", "notes": "", "sections": [], "scratch": "<p></p>"}
    append_scratch_html(draft, "<p>two</p>")
    assert draft["scratch"] == "<p>two</p>"


def test_append_keeps_existing_text_first():
    draft = {"id": "d", "name": "D", "notes": "", "sections": [], "scratch": "<p>a</p>"}
    append_scratch_html(draft, "<p>b</p>")
    assert draft["scratch"] == "<p>a</p><p>b</p>"


def test_append_rejects_blank():
    draft = {"id": "d", "name": "D", "notes": "", "sections": []}
    with pytest.raises(McpToolError):
        append_scratch_html(draft, "   ")


def test_board_validation_preserves_scratch():
    draft = {"id": "d", "name": "D", "notes": "", "sections": [], "scratch": "<p>keep</p>"}
    board = validate_arguments_board(_board(draft))
    assert board["draftsBySide"]["petitioner"][0]["scratch"] == "<p>keep</p>"
