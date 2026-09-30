"""
Sync semantics: conflict resolution, tombstones, cursors, isolation.

These are the rules a second device depends on, and none of them are visible
from reading a single function, so they are tested against real SQL.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tests.conftest import requires_database


def _annotation(row_id: str, updated_at_ms: int, text: str, deleted: bool = False) -> dict:
    return {
        "id": row_id,
        "caseId": "case-at-bar",
        "fileId": "pdf-1",
        "page": 3,
        "kind": "page",
        "quote": "",
        "text": text,
        "rects": None,
        "updatedAt": updated_at_ms,
        "deleted": deleted,
    }


def _push(cursor, workspace_id: str, changes: dict, now: datetime | None = None) -> dict:
    from app.db.repository import push_changes, touch_workspace

    touch_workspace(cursor, workspace_id)
    return push_changes(cursor, workspace_id, changes, now or datetime.now(timezone.utc))


@requires_database
def test_push_then_pull_round_trips_an_annotation(cursor, workspace_id):
    from app.db.repository import pull_changes

    _push(cursor, workspace_id, {"annotations": [_annotation("a-1", 1_000, "pole camera")]})

    changes = pull_changes(cursor, workspace_id, 0)
    annotations = changes["annotations"]

    assert len(annotations) == 1
    assert annotations[0]["id"] == "a-1"
    assert annotations[0]["text"] == "pole camera"
    assert annotations[0]["page"] == 3
    assert annotations[0]["deleted"] is False
    assert annotations[0]["fileId"] == "pdf-1"


@requires_database
def test_document_pull_includes_stored_after_blob_upload(cursor, workspace_id):
    """
    Readers only re-fetch PDF bytes when filesMeta.stored is true.

    Sync must expose that flag even though blob_pathname itself stays off the
    client wire (so one workspace cannot point at another's Blob object).
    """
    from app.db.repository import pull_changes, touch_workspace, upsert_document_blob

    touch_workspace(cursor, workspace_id)
    now = datetime.now(timezone.utc)
    upsert_document_blob(
        cursor,
        workspace_id,
        document_id="pdf-article-1",
        case_id="corpus-articles",
        name="Summary.pdf",
        size_bytes=1200,
        content_type="application/pdf",
        blob_pathname=f"case-law-agent/workspaces/{workspace_id}/pdf-article-1/secret.pdf",
        blob_url="https://example.public.blob.vercel-storage.com/secret.pdf",
        now=now,
    )

    documents = pull_changes(cursor, workspace_id, 0)["documents"]

    assert len(documents) == 1
    assert documents[0]["id"] == "pdf-article-1"
    assert documents[0]["stored"] is True
    assert "blob_pathname" not in documents[0]


@requires_database
def test_newer_push_overwrites_older_row(cursor, workspace_id):
    from app.db.repository import pull_changes

    _push(cursor, workspace_id, {"annotations": [_annotation("a-1", 1_000, "first")]})
    _push(cursor, workspace_id, {"annotations": [_annotation("a-1", 2_000, "second")]})

    annotations = pull_changes(cursor, workspace_id, 0)["annotations"]

    assert len(annotations) == 1
    assert annotations[0]["text"] == "second"


@requires_database
def test_older_push_is_rejected(cursor, workspace_id):
    """
    A device that was offline must not clobber a newer edit when it reconnects.

    This is the whole point of the WHERE clause on ON CONFLICT DO UPDATE, and
    it is the easiest thing to break while refactoring the statement builder.
    """
    from app.db.repository import pull_changes

    _push(cursor, workspace_id, {"annotations": [_annotation("a-1", 5_000, "newer edit")]})
    written = _push(cursor, workspace_id, {"annotations": [_annotation("a-1", 1_000, "stale")]})

    annotations = pull_changes(cursor, workspace_id, 0)["annotations"]

    assert written["annotations"] == 0, "stale row should not count as written"
    assert annotations[0]["text"] == "newer edit"


@requires_database
def test_delete_comes_back_as_a_tombstone(cursor, workspace_id):
    from app.db.repository import pull_changes

    _push(cursor, workspace_id, {"annotations": [_annotation("a-1", 1_000, "note")]})
    _push(cursor, workspace_id, {"annotations": [_annotation("a-1", 2_000, "note", deleted=True)]})

    annotations = pull_changes(cursor, workspace_id, 0)["annotations"]

    assert len(annotations) == 1
    assert annotations[0]["deleted"] is True


@requires_database
def test_id_only_annotation_tombstone_inserts(cursor, workspace_id):
    """
    Client delete tombstones historically sent only {id, deleted, updatedAt}.

    That used to 500 on NotNullViolation for case_id and block every later sync
    retry in the browser. Defaults must let the INSERT succeed.
    """
    from app.db.repository import pull_changes

    written = _push(
        cursor,
        workspace_id,
        {
            "annotations": [
                {"id": "a-sparse-tombstone", "updatedAt": 1_000, "deleted": True},
            ]
        },
    )

    assert written["annotations"] == 1
    annotations = pull_changes(cursor, workspace_id, 0)["annotations"]
    assert annotations[0]["id"] == "a-sparse-tombstone"
    assert annotations[0]["deleted"] is True
    assert annotations[0]["caseId"] == ""


@requires_database
def test_future_client_clock_is_clamped_to_server_time(cursor, workspace_id):
    """
    A device whose clock runs fast must not win every future conflict.

    Without the clamp, one bad clock pins a row permanently: every later edit
    made on a correct clock looks older and gets rejected.
    """
    from app.db.repository import pull_changes, to_epoch_ms

    now = datetime.now(timezone.utc)
    far_future_ms = to_epoch_ms(now + timedelta(days=3_650))

    _push(cursor, workspace_id, {"annotations": [_annotation("a-1", far_future_ms, "fast clock")]}, now)
    stored = pull_changes(cursor, workspace_id, 0)["annotations"][0]

    assert stored["updatedAt"] <= to_epoch_ms(now)

    # A normal edit a moment later still wins.
    later = now + timedelta(seconds=1)
    _push(cursor, workspace_id, {"annotations": [_annotation("a-1", to_epoch_ms(later), "correct clock")]}, later)

    assert pull_changes(cursor, workspace_id, 0)["annotations"][0]["text"] == "correct clock"


@requires_database
def test_pull_cursor_excludes_unchanged_rows(cursor, workspace_id):
    from app.db.repository import pull_changes, to_epoch_ms

    first = datetime.now(timezone.utc)
    _push(cursor, workspace_id, {"annotations": [_annotation("a-1", to_epoch_ms(first), "old")]}, first)

    cursor_ms = to_epoch_ms(first + timedelta(seconds=1))
    assert pull_changes(cursor, workspace_id, cursor_ms)["annotations"] == []

    second = first + timedelta(seconds=2)
    _push(cursor, workspace_id, {"annotations": [_annotation("a-2", to_epoch_ms(second), "new")]}, second)

    changed = pull_changes(cursor, workspace_id, cursor_ms)["annotations"]
    assert [row["id"] for row in changed] == ["a-2"]


@requires_database
def test_workspaces_cannot_see_each_other(cursor, workspace_id):
    """Without a login, workspace scoping is the only separation there is."""
    import uuid

    from app.db.repository import pull_changes

    other_workspace = str(uuid.uuid4())

    _push(cursor, workspace_id, {"annotations": [_annotation("a-1", 1_000, "mine")]})
    _push(cursor, other_workspace, {"annotations": [_annotation("a-2", 1_000, "theirs")]})

    mine = pull_changes(cursor, workspace_id, 0)["annotations"]
    theirs = pull_changes(cursor, other_workspace, 0)["annotations"]

    assert [row["id"] for row in mine] == ["a-1"]
    assert [row["id"] for row in theirs] == ["a-2"]


@requires_database
def test_notes_are_keyed_by_case_and_layer(cursor, workspace_id):
    """Two layers of the same case are separate rows, not one overwriting the other."""
    from app.db.repository import pull_changes

    _push(
        cursor,
        workspace_id,
        {
            "notes": [
                {"caseId": "case-at-bar", "layerId": "overview", "html": "<p>record</p>", "updatedAt": 1_000},
                {"caseId": "case-at-bar", "layerId": "holding", "html": "<p>holding</p>", "updatedAt": 1_000},
            ]
        },
    )

    notes = pull_changes(cursor, workspace_id, 0)["notes"]

    assert {(row["caseId"], row["layerId"]) for row in notes} == {
        ("case-at-bar", "overview"),
        ("case-at-bar", "holding"),
    }


@requires_database
def test_jsonb_fields_survive_the_round_trip(cursor, workspace_id):
    """Highlight rectangles and matter issues must come back unchanged."""
    from app.db.repository import pull_changes

    rects = [{"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.04}]
    issues = [{"id": "q1", "label": "Fourth Amendment", "short": "Pole camera"}]

    _push(
        cursor,
        workspace_id,
        {
            "annotations": [
                {
                    "id": "a-1",
                    "caseId": "case-at-bar",
                    "fileId": "pdf-1",
                    "page": 2,
                    "kind": "highlight",
                    "quote": "reasonable expectation",
                    "text": "",
                    "rects": rects,
                    "updatedAt": 1_000,
                }
            ],
            "matters": [
                {"id": "bronner-2026", "title": "Bronner", "season": "AMCA", "issues": issues, "updatedAt": 1_000}
            ],
        },
    )

    changes = pull_changes(cursor, workspace_id, 0)

    assert changes["annotations"][0]["rects"] == rects
    assert changes["matters"][0]["issues"] == issues


@requires_database
def test_unknown_collections_are_ignored(cursor, workspace_id):
    """
    An older backend must not reject a newer client outright.

    Sync is the one endpoint where a version mismatch should degrade instead of
    failing, so the client can keep saving everything else.
    """
    written = _push(
        cursor,
        workspace_id,
        {
            "annotations": [_annotation("a-1", 1_000, "kept")],
            "argument_chains": [{"id": "chain-1", "updatedAt": 1_000}],
        },
    )

    assert written["annotations"] == 1
    assert "argument_chains" not in written


@requires_database
def test_library_records_separate_kinds_sharing_an_id(cursor, workspace_id):
    """Seed ids repeat across collections, so kind has to be part of the key."""
    from app.db.repository import pull_changes

    _push(
        cursor,
        workspace_id,
        {
            "library_records": [
                {"kind": "cites", "id": "shared-1", "data": {"label": "a cite"}, "updatedAt": 1_000},
                {"kind": "timeline", "id": "shared-1", "data": {"label": "an event"}, "updatedAt": 1_000},
            ]
        },
    )

    records = pull_changes(cursor, workspace_id, 0)["library_records"]

    assert len(records) == 2
    assert {row["kind"] for row in records} == {"cites", "timeline"}
