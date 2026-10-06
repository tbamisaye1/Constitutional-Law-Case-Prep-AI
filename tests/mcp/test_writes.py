"""MCP write path: sync visibility, conflicts, revisions, isolation, dry runs."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from tests.conftest import requires_database

pytestmark = requires_database


@pytest.fixture
def ws(cursor, workspace_id):
    from app.db.repository import touch_workspace

    touch_workspace(cursor, workspace_id)
    return workspace_id


def test_upsert_prong_visible_via_sync(cursor, ws):
    from app.db.repository import pull_changes, to_epoch_ms
    from app.mcp.arguments_shape import (
        deep_copy_board,
        empty_main_draft,
        find_draft,
        find_section,
    )
    from app.mcp.ids import new_id
    from app.mcp.tools.arguments import _load_board, _save_board
    from app.mcp.writes import write_library_data

    before = to_epoch_ms(datetime.now(timezone.utc)) - 1
    board = {
        "draftsBySide": {
            "petitioner": [empty_main_draft("petitioner")],
            "respondent": [empty_main_draft("respondent")],
        },
        "activeDraftBySide": {
            "petitioner": "petitioner-main",
            "respondent": "respondent-main",
        },
    }
    write_library_data(
        cursor,
        ws,
        kind="arguments",
        record_id="main",
        data=board,
        tool_name="test_seed",
        expected_updated_at=None,
    )
    loaded, current = _load_board(cursor, ws)
    next_board = deep_copy_board(loaded)
    draft = find_draft(next_board, "petitioner", "petitioner-main")
    section = find_section(draft, draft["sections"][0]["id"])
    section.setdefault("prongs", []).append(
        {
            "id": new_id("pr"),
            "title": "New prong from MCP",
            "notes": "<p>R. 12 says the cameras ran for 93 days.</p>",
        }
    )
    saved = _save_board(
        cursor,
        ws,
        next_board,
        tool_name="upsert_prong",
        expected_updated_at=current,
    )
    assert "error" not in saved
    changes = pull_changes(cursor, ws, before)
    lib = changes.get("library_records") or []
    args_rows = [
        r for r in lib if r.get("kind") == "arguments" and r.get("id") == "main"
    ]
    assert args_rows, "sync pull must include MCP arguments write"
    prongs = args_rows[0]["data"]["draftsBySide"]["petitioner"][0]["sections"][0][
        "prongs"
    ]
    assert any(p.get("title") == "New prong from MCP" for p in prongs)


def test_conflict_on_stale_expected(cursor, ws):
    from app.mcp.errors import McpToolError
    from app.mcp.writes import write_library_data

    first = write_library_data(
        cursor,
        ws,
        kind="facts",
        record_id="main",
        data={"title": "v1"},
        tool_name="t1",
        expected_updated_at=None,
    )
    write_library_data(
        cursor,
        ws,
        kind="facts",
        record_id="main",
        data={"title": "v2"},
        tool_name="t2",
        expected_updated_at=first["updated_at"],
    )
    with pytest.raises(McpToolError) as exc:
        write_library_data(
            cursor,
            ws,
            kind="facts",
            record_id="main",
            data={"title": "v3"},
            tool_name="t3",
            expected_updated_at=first["updated_at"],
        )
    assert exc.value.code == "conflict"
    assert "current" in exc.value.details


def test_revision_source_mcp(cursor, ws):
    from app.mcp.writes import write_library_data

    first = write_library_data(
        cursor,
        ws,
        kind="facts",
        record_id="main",
        data={"title": "before"},
        tool_name="t1",
        expected_updated_at=None,
    )
    second = write_library_data(
        cursor,
        ws,
        kind="facts",
        record_id="main",
        data={"title": "after"},
        tool_name="t2",
        expected_updated_at=first["updated_at"],
    )
    assert second.get("revision_id")
    cursor.execute(
        "SELECT source FROM library_record_revisions WHERE id = %s",
        (second["revision_id"],),
    )
    assert cursor.fetchone()["source"] == "mcp"


def test_workspace_isolation(cursor, ws):
    from app.db.repository import touch_workspace
    from app.mcp.writes import write_library_data
    from app.mcp.workspace import resolve_workspace_id

    other = str(uuid.uuid4())
    touch_workspace(cursor, other)
    write_library_data(
        cursor,
        other,
        kind="facts",
        record_id="main",
        data={"secret": True},
        tool_name="t",
        expected_updated_at=None,
    )
    resolved = resolve_workspace_id(cursor, ws)
    assert resolved == ws
    cursor.execute(
        """
        SELECT data FROM library_records
        WHERE workspace_id = %s AND kind = 'facts' AND id = 'main'
        """,
        (ws,),
    )
    assert cursor.fetchone() is None


def test_unknown_keys_survive_upsert_prong(cursor, ws):
    from app.mcp.arguments_shape import (
        deep_copy_board,
        empty_main_draft,
        find_draft,
        find_section,
    )
    from app.mcp.ids import new_id
    from app.mcp.tools.arguments import _load_board, _save_board
    from app.mcp.writes import write_library_data

    board = {
        "draftsBySide": {
            "petitioner": [empty_main_draft("petitioner")],
            "respondent": [empty_main_draft("respondent")],
        },
        "activeDraftBySide": {
            "petitioner": "petitioner-main",
            "respondent": "respondent-main",
        },
        "customFutureKey": {"keep": True},
        "removedOutlineIdsByDraft": {"petitioner-main": ["x"]},
    }
    write_library_data(
        cursor,
        ws,
        kind="arguments",
        record_id="main",
        data=board,
        tool_name="seed",
        expected_updated_at=None,
    )
    loaded, current = _load_board(cursor, ws)
    next_board = deep_copy_board(loaded)
    draft = find_draft(next_board, "petitioner", "petitioner-main")
    section = find_section(draft, draft["sections"][0]["id"])
    section.setdefault("prongs", []).append(
        {"id": new_id("pr"), "title": "P", "notes": ""}
    )
    _save_board(
        cursor,
        ws,
        next_board,
        tool_name="upsert_prong",
        expected_updated_at=current,
    )
    loaded2, _ = _load_board(cursor, ws)
    assert loaded2["customFutureKey"] == {"keep": True}
    assert loaded2["removedOutlineIdsByDraft"] == {"petitioner-main": ["x"]}


def test_dry_run_delete_case_writes_nothing(cursor, ws):
    from app.mcp.writes import require_confirm, write_entity
    from app.mcp.errors import McpToolError

    write_entity(
        cursor,
        ws,
        entity="cases",
        key={"id": "case-1"},
        values={"name": "Test Case", "cite": "1 U.S. 1"},
        tool_name="seed",
        expected_updated_at=None,
    )
    cursor.execute(
        "SELECT count(*) AS n FROM cases WHERE workspace_id = %s AND deleted_at IS NULL",
        (ws,),
    )
    before = cursor.fetchone()["n"]

    # Dry-run confirm gate must not write.
    require_confirm(dry_run=True, confirm=False, action="delete_case")
    with pytest.raises(McpToolError):
        require_confirm(dry_run=False, confirm=False, action="delete_case")

    cursor.execute(
        "SELECT count(*) AS n FROM cases WHERE workspace_id = %s AND deleted_at IS NULL",
        (ws,),
    )
    assert cursor.fetchone()["n"] == before
