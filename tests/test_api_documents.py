"""
Document upload, download, and delete.

Blob is stubbed. What matters here is our own contract: the size guard that
keeps Vercel's platform 413 from replacing our readable error, the redirect
that keeps large PDFs out of the response body, and the tombstone that tells
other devices the file is gone.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import requires_database

WORKSPACE_HEADER = "X-Workspace-Id"


@pytest.fixture
def blob_stub(monkeypatch):
    """Capture Blob writes and deletes instead of making network calls."""
    written: dict[str, bytes] = {}
    deleted: list[str] = []

    def fake_put(pathname: str, data: bytes, content_type: str) -> dict:
        written[pathname] = data
        return {"url": f"https://example.public.blob.vercel-storage.com/{pathname}"}

    monkeypatch.setattr("app.api.documents.put_blob", fake_put)
    monkeypatch.setattr("app.api.documents.delete_blob", lambda pathname: deleted.append(pathname))
    monkeypatch.setattr("app.api.documents.blob_configured", lambda: True)

    return {"written": written, "deleted": deleted}


def _pdf_bytes(size: int = 1024) -> bytes:
    """Minimal bytes that pass the .pdf name check; content is never parsed here."""
    return b"%PDF-1.7\n" + b"0" * size


@requires_database
def test_upload_stores_bytes_and_records_the_document(client, workspace_id, blob_stub):
    response = client.post(
        "/documents",
        headers={WORKSPACE_HEADER: workspace_id},
        files={"file": ("bronner-record.pdf", _pdf_bytes(), "application/pdf")},
        data={"document_id": "pdf-1", "case_id": "case-at-bar"},
    )

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["id"] == "pdf-1"
    assert body["caseId"] == "case-at-bar"
    assert body["name"] == "bronner-record.pdf"
    assert body["stored"] is True
    assert len(blob_stub["written"]) == 1


@requires_database
def test_blob_pathname_is_scoped_to_the_workspace(client, workspace_id, blob_stub):
    """
    One workspace's bytes must not land in another's prefix.

    The pathname also carries a random segment, because the resulting URL is
    unguessable rather than access-controlled.
    """
    client.post(
        "/documents",
        headers={WORKSPACE_HEADER: workspace_id},
        files={"file": ("record.pdf", _pdf_bytes(), "application/pdf")},
        data={"document_id": "pdf-1", "case_id": "case-at-bar"},
    )

    pathname = next(iter(blob_stub["written"]))

    assert workspace_id in pathname
    assert pathname != f"case-law-agent/workspaces/{workspace_id}/documents/pdf-1.pdf"


@requires_database
def test_oversized_upload_explains_the_vercel_limit(client, workspace_id, blob_stub):
    from app.api.documents import MAX_UPLOAD_BYTES

    response = client.post(
        "/documents",
        headers={WORKSPACE_HEADER: workspace_id},
        files={"file": ("huge.pdf", _pdf_bytes(MAX_UPLOAD_BYTES + 1), "application/pdf")},
        data={"document_id": "pdf-1", "case_id": "case-at-bar"},
    )

    assert response.status_code == 413
    detail = response.json()["detail"]
    assert "MB" in detail
    assert blob_stub["written"] == {}, "nothing should reach Blob when the file is too big"


@requires_database
def test_non_pdf_is_refused(client, workspace_id, blob_stub):
    response = client.post(
        "/documents",
        headers={WORKSPACE_HEADER: workspace_id},
        files={"file": ("notes.txt", b"hello", "text/plain")},
        data={"document_id": "pdf-1", "case_id": "case-at-bar"},
    )

    assert response.status_code == 400


@requires_database
def test_download_redirects_rather_than_streaming(client, workspace_id, blob_stub):
    """
    Bytes must not pass through the function.

    Vercel caps a response body at 4.5 MB, so proxying a record PDF would fail
    for exactly the files this feature exists to hold.
    """
    client.post(
        "/documents",
        headers={WORKSPACE_HEADER: workspace_id},
        files={"file": ("record.pdf", _pdf_bytes(), "application/pdf")},
        data={"document_id": "pdf-1", "case_id": "case-at-bar"},
    )

    response = client.get(
        "/documents/pdf-1/file",
        headers={WORKSPACE_HEADER: workspace_id},
        follow_redirects=False,
    )

    assert response.status_code == 307
    assert response.headers["location"].startswith("https://")


@requires_database
def test_another_workspace_cannot_read_the_document(client, workspace_id, blob_stub):
    import uuid

    client.post(
        "/documents",
        headers={WORKSPACE_HEADER: workspace_id},
        files={"file": ("record.pdf", _pdf_bytes(), "application/pdf")},
        data={"document_id": "pdf-1", "case_id": "case-at-bar"},
    )

    response = client.get(
        "/documents/pdf-1",
        headers={WORKSPACE_HEADER: str(uuid.uuid4())},
    )

    assert response.status_code == 404


@requires_database
def test_delete_tombstones_the_row_and_removes_the_bytes(client, workspace_id, blob_stub):
    client.post(
        "/documents",
        headers={WORKSPACE_HEADER: workspace_id},
        files={"file": ("record.pdf", _pdf_bytes(), "application/pdf")},
        data={"document_id": "pdf-1", "case_id": "case-at-bar"},
    )

    response = client.delete("/documents/pdf-1", headers={WORKSPACE_HEADER: workspace_id})

    assert response.status_code == 200
    assert response.json()["blobRemoved"] is True
    assert len(blob_stub["deleted"]) == 1

    # The tombstone has to reach other devices through sync.
    synced = client.get("/sync", headers={WORKSPACE_HEADER: workspace_id}).json()
    documents = synced["changes"]["documents"]

    assert [row["id"] for row in documents] == ["pdf-1"]
    assert documents[0]["deleted"] is True


@requires_database
def test_upload_is_refused_when_blob_is_not_configured(client, workspace_id, monkeypatch):
    """Local development without a Blob token should say so, not 500."""
    monkeypatch.setattr("app.api.documents.blob_configured", lambda: False)

    response = client.post(
        "/documents",
        headers={WORKSPACE_HEADER: workspace_id},
        files={"file": ("record.pdf", _pdf_bytes(), "application/pdf")},
        data={"document_id": "pdf-1", "case_id": "case-at-bar"},
    )

    assert response.status_code == 503


@requires_database
def test_blob_client_upload_mints_a_workspace_scoped_token(client, workspace_id, monkeypatch):
    monkeypatch.setenv(
        "BLOB_READ_WRITE_TOKEN",
        "vercel_blob_rw_store_testhost_secretsegment",
    )

    response = client.post(
        "/documents/blob-client-upload",
        headers={WORKSPACE_HEADER: workspace_id},
        json={
            "type": "blob.generate-client-token",
            "payload": {
                "pathname": "ignored/by/server.pdf",
                "clientPayload": json.dumps(
                    {"documentId": "pdf-big", "caseId": "youngstown", "name": "Youngstown.pdf"}
                ),
                "multipart": False,
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["type"] == "blob.generate-client-token"
    assert body["clientToken"].startswith("vercel_blob_client_")
    assert workspace_id in body["pathname"]
    assert body["pathname"].startswith("case-law-agent/workspaces/")
    assert body["pathname"].endswith(".pdf")


@requires_database
def test_complete_direct_upload_records_the_blob(client, workspace_id, blob_stub):
    pathname = f"case-law-agent/workspaces/{workspace_id}/pdf-big/secret.pdf"
    response = client.post(
        "/documents/complete",
        headers={WORKSPACE_HEADER: workspace_id},
        json={
            "document_id": "pdf-big",
            "case_id": "youngstown",
            "name": "Youngstown.pdf",
            "size_bytes": 4_500_000,
            "blob_pathname": pathname,
            "blob_url": f"https://example.public.blob.vercel-storage.com/{pathname}",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == "pdf-big"
    assert body["stored"] is True
    assert body["size"] == 4_500_000


@requires_database
def test_complete_rejects_another_workspaces_pathname(client, workspace_id):
    response = client.post(
        "/documents/complete",
        headers={WORKSPACE_HEADER: workspace_id},
        json={
            "document_id": "pdf-big",
            "case_id": "youngstown",
            "name": "Youngstown.pdf",
            "size_bytes": 1000,
            "blob_pathname": "case-law-agent/workspaces/other-id/pdf-big/secret.pdf",
            "blob_url": "https://example.public.blob.vercel-storage.com/x.pdf",
        },
    )

    assert response.status_code == 400
    assert "browser-only" in response.json()["detail"]
