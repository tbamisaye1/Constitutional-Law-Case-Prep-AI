"""HTTP behaviour of the sync endpoints, including the header contract."""

from __future__ import annotations

from tests.conftest import requires_database

WORKSPACE_HEADER = "X-Workspace-Id"


def _annotation(row_id: str, updated_at_ms: int, text: str) -> dict:
    return {
        "id": row_id,
        "caseId": "case-at-bar",
        "fileId": "pdf-1",
        "page": 1,
        "kind": "page",
        "quote": "",
        "text": text,
        "rects": None,
        "updatedAt": updated_at_ms,
    }


@requires_database
def test_missing_workspace_header_is_rejected(client):
    response = client.get("/sync")

    assert response.status_code == 422, response.text


@requires_database
def test_malformed_workspace_header_is_rejected(client):
    """
    A non-UUID key must fail rather than being coerced.

    The value goes straight into a uuid query parameter, so rejecting it early
    is what keeps that safe.
    """
    response = client.get("/sync", headers={WORKSPACE_HEADER: "not-a-uuid"})

    assert response.status_code == 400
    assert WORKSPACE_HEADER in response.json()["detail"]


@requires_database
def test_push_then_pull_returns_the_row_and_a_server_cursor(client, workspace_id):
    response = client.post(
        "/sync",
        headers={WORKSPACE_HEADER: workspace_id},
        json={"since": 0, "changes": {"annotations": [_annotation("a-1", 1_000, "pole camera")]}},
    )

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["written"]["annotations"] == 1
    assert body["serverTime"] > 0
    assert [row["id"] for row in body["changes"]["annotations"]] == ["a-1"]


@requires_database
def test_server_cursor_stops_the_client_re_downloading(client, workspace_id):
    """
    The response cursor must exclude what the client already has.

    A cursor that kept replaying old rows would make every sync grow, which is
    the failure mode that makes local-first sync feel broken.
    """
    first = client.post(
        "/sync",
        headers={WORKSPACE_HEADER: workspace_id},
        json={"since": 0, "changes": {"annotations": [_annotation("a-1", 1_000, "first")]}},
    ).json()

    second = client.post(
        "/sync",
        headers={WORKSPACE_HEADER: workspace_id},
        json={"since": first["serverTime"], "changes": {}},
    ).json()

    assert second["changes"]["annotations"] == []


@requires_database
def test_oversized_push_is_refused_with_a_useful_message(client, workspace_id):
    from app.db.repository import MAX_ROWS_PER_ENTITY

    too_many = [_annotation(f"a-{i}", 1_000, "x") for i in range(MAX_ROWS_PER_ENTITY + 1)]

    response = client.post(
        "/sync",
        headers={WORKSPACE_HEADER: workspace_id},
        json={"since": 0, "changes": {"annotations": too_many}},
    )

    assert response.status_code == 413
    assert "smaller batches" in response.json()["detail"]


@requires_database
def test_status_counts_live_rows_only(client, workspace_id):
    client.post(
        "/sync",
        headers={WORKSPACE_HEADER: workspace_id},
        json={
            "since": 0,
            "changes": {
                "annotations": [
                    _annotation("a-1", 1_000, "kept"),
                    {**_annotation("a-2", 1_000, "gone"), "deleted": True},
                ]
            },
        },
    )

    body = client.get("/sync/status", headers={WORKSPACE_HEADER: workspace_id}).json()

    assert body["counts"]["annotations"] == 1
    assert body["workspaceId"] == workspace_id


@requires_database
def test_matters_falls_back_to_the_seed_without_a_workspace(client):
    """A plain call with no key still has to answer with the moot problem."""
    body = client.get("/matters").json()

    assert [matter["id"] for matter in body["matters"]] == ["bronner-2026"]


@requires_database
def test_workspace_matter_overrides_the_seed(client, workspace_id):
    client.post(
        "/sync",
        headers={WORKSPACE_HEADER: workspace_id},
        json={
            "since": 0,
            "changes": {
                "matters": [
                    {
                        "id": "bronner-2026",
                        "title": "Bronner, my edit",
                        "season": "AMCA 2026-27",
                        "issues": [],
                        "updatedAt": 2_000,
                    }
                ]
            },
        },
    )

    body = client.get("/matters", headers={WORKSPACE_HEADER: workspace_id}).json()
    titles = {matter["id"]: matter["title"] for matter in body["matters"]}

    assert titles["bronner-2026"] == "Bronner, my edit"
