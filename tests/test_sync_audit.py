"""Durable sync_audit_log rows for every /sync round-trip."""

from __future__ import annotations

from tests.conftest import requires_database

WORKSPACE_HEADER = "X-Workspace-Id"


def _arguments_board(notes: str, updated_at_ms: int = 5_000) -> dict:
    return {
        "id": "main",
        "kind": "arguments",
        "updatedAt": updated_at_ms,
        "data": {
            "id": "main",
            "draftsBySide": {
                "petitioner": [
                    {
                        "id": "d1",
                        "name": "Test draft",
                        "notes": notes,
                        "sections": [
                            {
                                "id": "s1",
                                "prongs": [
                                    {
                                        "id": "c3-s1-a",
                                        "title": "2.1",
                                        "notes": notes,
                                    }
                                ],
                            }
                        ],
                    }
                ],
                "respondent": [],
            },
            "activeDraftBySide": {"petitioner": "d1", "respondent": None},
        },
    }


@requires_database
def test_empty_heartbeat_is_audited(client, workspace_id):
    response = client.post(
        "/sync",
        headers={WORKSPACE_HEADER: workspace_id},
        json={"since": 0, "changes": {}},
    )
    assert response.status_code == 200, response.text

    audit = client.get(
        "/sync/audit",
        headers={WORKSPACE_HEADER: workspace_id},
        params={"limit": 5},
    ).json()

    assert audit["workspaceId"] == workspace_id
    assert audit["entries"], "expected at least one audit row"
    latest = audit["entries"][0]
    assert latest["event"] == "push_then_pull"
    assert latest["emptyPush"] is True
    assert latest["note"] and "empty_inbound" in latest["note"]


@requires_database
def test_arguments_push_stores_snapshot_and_flags_change(client, workspace_id):
    first = client.post(
        "/sync",
        headers={WORKSPACE_HEADER: workspace_id},
        json={
            "since": 0,
            "changes": {
                "library_records": [_arguments_board("first notes", 10_000)]
            },
        },
    )
    assert first.status_code == 200, first.text
    assert first.json()["written"].get("library_records", 0) >= 1

    second = client.post(
        "/sync",
        headers={WORKSPACE_HEADER: workspace_id},
        json={
            "since": first.json()["serverTime"],
            "changes": {
                "library_records": [_arguments_board("first notes", 10_000)]
            },
        },
    )
    assert second.status_code == 200, second.text

    third = client.post(
        "/sync",
        headers={WORKSPACE_HEADER: workspace_id},
        json={
            "since": second.json()["serverTime"],
            "changes": {
                "library_records": [_arguments_board("changed notes", 20_000)]
            },
        },
    )
    assert third.status_code == 200, third.text

    audit = client.get(
        "/sync/audit",
        headers={WORKSPACE_HEADER: workspace_id},
        params={"limit": 10},
    ).json()["entries"]

    # Newest first: third (changed), second (identical), first (insert).
    changed = next(
        row
        for row in audit
        if row.get("argumentsUnchanged") is False and row.get("hasArgumentsSnapshot")
    )
    identical = next(
        row
        for row in audit
        if row.get("argumentsUnchanged") is True and row.get("hasArgumentsSnapshot")
    )
    assert changed["emptyPush"] is False
    assert identical["emptyPush"] is False
    assert any(
        s.get("kind") == "arguments" and s.get("c3s1aNotesBytes", 0) > 0
        for s in changed["rowSummaries"]
    )
